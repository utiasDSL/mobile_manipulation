import numpy as np
import casadi as cs
import time
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver

from mmseq_utils.math import wrap_pi_array
from mmseq_utils.casadi_struct import casadi_sym_struct
from mmseq_control_new.MPCConstraints import HierarchicalTrackingConstraint
from mmseq_control_new.MPCCostFunctions import CostFunctions
import mobile_manipulation_central as mm

from mmseq_control_new.MPC import MPC, INF

class HTMPC(MPC):

    def __init__(self, config):
        super().__init__(config)
        self.EEPos3LexConstraint = HierarchicalTrackingConstraint(self.EEPos3Cost, "_".join([self.EEPos3Cost.name, "Lex"]))
        self.BasePos2LexConstraint = HierarchicalTrackingConstraint(self.BasePos2Cost, "_".join([self.BasePos2Cost.name, "Lex"]))
        common_cost_fcns = [cost for cost in self.collisionSoftCsts.values()]
        ocp1, ocp_solver1, p_struct1 = self._construct([self.EEPos3Cost, self.RegularizationCost, self.CtrlEffCost] + common_cost_fcns, [], 1, "EEPos3")
        ocp2, ocp_solver2, p_struct2 = self._construct([self.BasePos2Cost, self.CtrlEffCost] + common_cost_fcns, [self.EEPos3LexConstraint], 1, "BasePos2")
        
        self.stmpcs = [ocp1, ocp2]
        self.stmpc_solvers = [ocp_solver1, ocp_solver2]
        self.stmpc_p_structs = [p_struct1, p_struct2]

    def control(self, t, robot_states, planners, map=None):
        task_num = len(planners)
        self.py_logger.debug("control time {}".format(t))
        self.curr_control_time = t
        q, v = robot_states
        q[2:9] = wrap_pi_array(q[2:9])
        xo = np.hstack((q, v))

        # 0.1 Get warm start point
        self.u_bar[:-1] = self.u_bar[1:]
        self.u_bar[-1] = 0
        self.x_bar = self._predictTrajectories(xo, self.u_bar)            

        # 0.2 Get ref, sdf map,
        r_bar_map = {}
        self.ree_bar = []
        self.rbase_bar = []

        t1 = time.perf_counter()
        if map is not None and self.params["sdf_collision_avoidance_enabled"]:
            self.model_interface.sdf_map.update_map(*map)
        t2 = time.perf_counter()
        self.log["time_map_update"][:] = t2 - t1

        for pid, planner in enumerate(planners):
            r_bar = [planner.getTrackingPoint(t + k * self.dt, (self.x_bar[k, :self.DoF], self.x_bar[k, self.DoF:]))[0]
                        for k in range(self.N + 1)]
            acceptable_ref = False
            if planner.type == "EE" and  planner.ref_data_type == "Vec3":
                acceptable_ref = True
                r_bar_map[self.EEPos3Cost.name] = r_bar
            elif planner.type == "base" and planner.ref_data_type == "Vec2":
                acceptable_ref = True
                r_bar_map[self.BasePos2Cost.name] = r_bar

            if not acceptable_ref:
                self.py_logger.warning(f"unknown cost type {planner.ref_data_type}, planner {planner.name}")
            
            if planner.type == "EE":
                self.ree_bar = r_bar 
            elif planner.type == "base":
                self.rbase_bar = r_bar

        # Optimal tracking error
        e_p_bar_map = {}

        # for logging
        self.log = self._get_log(task_num=task_num)

        tracking_cost_fcns = []
        ps = []
        for task_id, (stmpc, stmpc_solver, p_struct) in enumerate(zip(self.stmpcs, self.stmpc_solvers, self.stmpc_p_structs)):
            tracking_cost_fcn_name = stmpc.name
            tracking_cost_fcn = self.EEPos3Cost if tracking_cost_fcn_name == "EEPos3" else self.BasePos2Cost
            tracking_cost_fcns.append(tracking_cost_fcn)

            if self.params["sdf_collision_avoidance_enabled"]:
                sdf_params = self.model_interface.sdf_map.get_params()

            t1 = time.perf_counter()
            curr_p_map_bar = []
            for k in range(self.N+1):
                # set parameters
                curr_p_map = p_struct(0)
                
                # Set regularization
                if task_id == 0:
                    curr_p_map["eps_Regularization"] = self.params["cost_params"]["Regularization"]["eps"]

                # SDF parameters
                if self.params["sdf_collision_avoidance_enabled"]:
                    curr_p_map["x_grid_sdf"] = sdf_params[0]
                    curr_p_map["y_grid_sdf"] = sdf_params[1]
                    if self.model_interface.sdf_map.dim == 3:
                        curr_p_map["z_grid_sdf"] = sdf_params[2]
                        curr_p_map["value_sdf"] = sdf_params[3]
                    else:
                        curr_p_map["value_sdf"] = sdf_params[2]

                # set initial guess
                stmpc_solver.set(k, 'x', self.x_bar[k])
                if k != self.N:
                    stmpc_solver.set(k, 'u', self.u_bar[k])

                # set parameters for tracking cost functions
                p_keys = p_struct.keys()
                p_name_r = "_".join(["r", tracking_cost_fcn_name])

                if p_name_r in p_keys:
                    r_k = r_bar_map[tracking_cost_fcn_name][k]
                    # set reference
                    curr_p_map[p_name_r] = r_k
                    p_name_W = "_".join(["W", tracking_cost_fcn_name])

                    # Set weight matricies, assuming identity matrix with identical diagonal terms
                    if k == self.N:
                        curr_p_map[p_name_W] = self.params["cost_params"][tracking_cost_fcn_name]["P"] * np.eye(r_k.size)
                    else:
                        curr_p_map[p_name_W] = self.params["cost_params"][tracking_cost_fcn_name]["Qk"] * np.eye(r_k.size)

                else:
                    self.py_logger.warning(f"unknown p name {p_name_r} in {tracking_cost_fcn_name}'s p_struct")
                
                for p in range(task_id):
                    # set parameters for lexicographic optimality constraints
                    name = self.stmpcs[p].name
                    p_name_e_p = "_".join(["e_p", name, "Lex"])
                    curr_p_map[p_name_e_p] = e_p_bar_map[name][k]

                    p_name_r = "_".join(["r", name])
                    r_k = r_bar_map[name][k]
                    # set reference
                    curr_p_map[p_name_r] = r_k

                stmpc_solver.set(k, 'p', curr_p_map.cat.full().flatten())
                curr_p_map_bar.append(curr_p_map)
            t2 = time.perf_counter()
            self.log["time_stmpc_set_params"][task_id] = t2 - t1

            # Log: initial cost
            self.log["cost_iter"][task_id, 0] = self.evaluate_cost_function(tracking_cost_fcn, self.x_bar, self.u_bar, curr_p_map_bar)

            # Solve stmpc
            t1 = time.perf_counter()
            stmpc_solver.solve_for_x0(xo, fail_on_nonzero_status=False)
            t2 = time.perf_counter()
            self.log["time_stmpc_solve"][task_id] = t2 - t1

            stmpc_solver.print_statistics() # encapsulates: stat = ocp_solver.get_stats("statistics")

            if stmpc_solver.status !=0:
                for i in range(self.N):
                    print(f"stage {i}: x: {stmpc_solver.get(i, 'x')}")
                    print(f"stage {i}: u: {stmpc_solver.get(i, 'u')}")
                            
                for i in range(self.N):
                    print(f"stage {i}: lam: {stmpc_solver.get(i, 'lam')}")
                
                for i in range(self.N):
                    print(f"stage {i}: pi: {stmpc_solver.get(i, 'pi')}")

                if self.params["raise_exception_on_failure"]:
                    raise Exception(f'acados acados_ocp_solver returned status {stmpc_solver.status}')
                
            # get solution
            for i in range(self.N):
                self.x_bar[i,:] = stmpc_solver.get(i, "x")
                self.u_bar[i,:] = stmpc_solver.get(i, "u")
            self.x_bar[self.N,:] = stmpc_solver.get(self.N, "x")

            # get e_p_bar
            e_p_bar_map[tracking_cost_fcn_name] = []
            for k in range(self.N+1):
                # Relaxed Lex Constraints |e_k| \leq |e^*_k| + eps
                e_p_bar_map[tracking_cost_fcn_name].append(np.abs(tracking_cost_fcn.get_e(self.x_bar[k],
                                                            [],
                                                            r_bar_map[tracking_cost_fcn_name][k])) + self.params["hierarchy_const_tol"])
                print(f"task:{tracking_cost_fcn_name}, time{k}: e_p {e_p_bar_map[tracking_cost_fcn_name][-1]}")
            
            e_p_bar_map[tracking_cost_fcn_name] = np.array(e_p_bar_map[tracking_cost_fcn_name])
            print(f"cost {tracking_cost_fcn_name}: {stmpc_solver.get_cost()}")

            # Log: solver status, step size, cost after stmpc
            self.log["solver_status"][task_id] = stmpc_solver.status
            self.log["step_size"][task_id] = np.mean(stmpc_solver.get_stats('alpha'))
            self.log["cost_iter"][task_id, 1] = self.evaluate_cost_function(tracking_cost_fcn, self.x_bar, self.u_bar, curr_p_map_bar)
            self.log["sqp_iter"][task_id] = stmpc_solver.get_stats('sqp_iter')
            self.log["qp_iter"][task_id] = sum(stmpc_solver.get_stats('qp_iter'))
            ps.append(curr_p_map_bar)

        # Log: final cost for each task after all stmpcs have been solved
        for i, tracking_cost_fcn in enumerate(tracking_cost_fcns):
            self.log["cost_final"][i] = self.evaluate_cost_function(tracking_cost_fcn, 
                                                             self.x_bar,
                                                             self.u_bar,
                                                             ps[i])

        self.v_cmd = self.x_bar[0][self.robot.DoF:].copy()
        self.ee_bar, self.base_bar = self._getEEBaseTrajectories(self.x_bar)

        return self.v_cmd, self.u_bar[0].copy(), self.u_bar.copy(), self.x_bar[:, 9:].copy()
    
    def _get_log(self, task_num=2):
        log = {"cost_iter": np.zeros((task_num, 2)),
        "cost_final": np.zeros(task_num),
        "solver_status": np.zeros(task_num),
        "step_size": np.zeros(task_num),
        "sqp_iter": np.zeros(task_num),
        "qp_iter": np.zeros(task_num),
        "time_map_update": np.zeros(task_num),
        "time_stmpc_set_params": np.zeros(task_num),
        "time_stmpc_solve": np.zeros(task_num),
        }
        return log
