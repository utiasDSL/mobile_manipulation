import numpy as np
import casadi as cs
import time
from typing import Optional, List, Dict, Tuple, Union
from numpy.typing import NDArray
from scipy.interpolate import interp1d

from mmseq_plan.PlanBaseClass import Planner,TrajectoryPlanner
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
        common_csts = []
        common_cost_fcns = []
        for name in self.collision_link_names:
            if name in self.model_interface.scene.collision_link_names["static_obstacles"]:
                softened = self.params["collision_constraints_softend"]["static_obstacles"]
            else:
                softened = self.params["collision_constraints_softend"][name]

            if softened:
                common_cost_fcns.append(self.collisionSoftCsts[name])
            else:
                common_csts.append(self.collisionCsts[name])
        self.stmpc_cost_fcns = []
        self.stmpc_cost_fcns.append([self.EEPos3Cost, self.RegularizationCost, self.CtrlEffCost] + common_cost_fcns)
        self.stmpc_cost_fcns.append([self.BasePoseSE2Cost, self.BaseVel3Cost, self.CtrlEffCost] + common_cost_fcns)

        ocp1, ocp_solver1, p_struct1 = self._construct(self.stmpc_cost_fcns[0], [] + common_csts, 1, "EEPos3")
        ocp2, ocp_solver2, p_struct2 = self._construct(self.stmpc_cost_fcns[1], [self.EEPos3LexConstraint] + common_csts, 1, "BasePoseSE2")
        
        self.stmpcs = [ocp1, ocp2]
        self.stmpc_solvers = [ocp_solver1, ocp_solver2]
        self.stmpc_p_structs = [p_struct1, p_struct2]

    def control(self, 
                t: float, 
                robot_states: Tuple[NDArray[np.float64], NDArray[np.float64]], 
                planners: List[Union[Planner, TrajectoryPlanner]], 
                map=None):
        
        task_num = len(planners)
        self.py_logger.debug("control time {}".format(t))
        self.curr_control_time = t
        q, v = robot_states
        q[2:9] = wrap_pi_array(q[2:9])
        xo = np.hstack((q, v))

        # for logging
        self.log = self._get_log(task_num=task_num)

        # 0.1 Get warm start point
        # self.u_bar[:-1] = self.u_bar[1:]
        # self.u_bar[-1] = 0
        # self.x_bar = self._predictTrajectories(xo, self.u_bar)            
        if self.t_bar is not None:
            self.u_t = interp1d(self.t_bar, self.u_bar, axis=0, 
                                bounds_error=False, fill_value="extrapolate")
            t_bar_new = t + np.arange(self.N)* self.dt
            self.u_bar = self.u_t(t_bar_new)
            self.x_bar = self._predictTrajectories(xo, self.u_bar)
        else:
            self.u_bar = np.zeros_like(self.u_bar)
            self.x_bar = self._predictTrajectories(xo, self.u_bar)

        x_bar_initial = self.x_bar.copy()
        u_bar_initial = self.u_bar.copy()

        # 0.2 Get ref, sdf map,
        r_bar_map = {}
        self.ree_bar = []
        self.rbase_bar = []

        for planner in planners:
            # get tracking points from planner, tracking points are tuples of desired position and velocity (p, v) 
            # for planners without velocity reference, none should be given (p, None) 
            if planner.ref_type == "path":
                p_bar, v_bar = planner.getTrackingPointArray((xo[:self.DoF], xo[self.DoF:]), self.N+1, self.dt)
            else:
                r_bar = [planner.getTrackingPoint(t + k * self.dt, (self.x_bar[k, :self.DoF], self.x_bar[k, self.DoF:]))
                            for k in range(self.N + 1)]
                p_bar = [r[0] for r in r_bar]
                v_bar = [r[1] for r in r_bar]
            contains_none = any(item is None for item in v_bar)
            velocity_ref_available = not contains_none
            
            acceptable_ref = True
            if planner.type == "EE":
                if planner.ref_data_type == "Vec3":
                    if planner.frame_id == "base_link":
                        r_bar_map["EEPos3BaseFrame"] = p_bar
                    else:
                        r_bar_map["EEPos3"] = p_bar
                        if velocity_ref_available:
                            # if planner generates velocity reference as well
                            r_bar_map["EEVel3"] = v_bar
                elif planner.ref_data_type == "SE3":
                    if planner.frame_id == "base_link":
                        r_bar_map["EEPoseBaseFrame"] = p_bar
                    else:
                        # assuming world frame
                        r_bar_map["EEPose"] = p_bar
                else:
                    acceptable_ref = False
            elif planner.type == "base":
                if planner.ref_data_type == "Vec2":
                    r_bar_map["BasePos2"] = p_bar
                    if velocity_ref_available:
                        r_bar_map["BaseVel2"] = v_bar
                elif planner.ref_data_type == "Vec3":
                    r_bar_map["BasePos3"] = p_bar
                    if velocity_ref_available:
                        r_bar_map["BaseVel3"] = v_bar
                elif planner.ref_data_type == "SE2":
                    r_bar_map["BasePoseSE2"] = p_bar
                    if velocity_ref_available:
                        r_bar_map["BaseVel3"] = v_bar
                else:
                    acceptable_ref = False

            if not acceptable_ref:
                self.py_logger.warning(f"unknown cost type {planner.ref_data_type}, planner {planner.name}")
            
            if planner.type == "EE":
                self.ree_bar = p_bar 
            elif planner.type == "base":
                self.rbase_bar = p_bar

        t1 = time.perf_counter()
        if map is not None and self.params["sdf_collision_avoidance_enabled"]:
            self.model_interface.sdf_map.update_map(*map)
        t2 = time.perf_counter()
        self.log["time_map_update"] = t2 - t1

        # Optimal tracking error
        e_p_bar_map = {}
        tracking_cost_fcns = []
        curr_p_map_bars = []
        map_params = self.model_interface.sdf_map.get_params()

        for task_id, (stmpc, stmpc_solver, p_struct, stmpc_cost_fcns) in enumerate(zip(self.stmpcs, self.stmpc_solvers, self.stmpc_p_structs, self.stmpc_cost_fcns)):
            tracking_cost_fcn_name = stmpc.name
            tracking_cost_fcn = stmpc_cost_fcns[0]
            tracking_cost_fcns.append(tracking_cost_fcn)


            t1 = time.perf_counter()
            curr_p_map_bar = []
            for k in range(self.N+1):
                # set parameters
                curr_p_map = p_struct(0)
                if self.params["sdf_collision_avoidance_enabled"]:
                    # params = self.model_interface.sdf_map.get_params()
                    curr_p_map["x_grid_sdf"] = map_params[0]
                    curr_p_map["y_grid_sdf"] = map_params[1]
                    if self.model_interface.sdf_map.dim == 3:
                        curr_p_map["z_grid_sdf"] = map_params[2]
                        curr_p_map["value_sdf"] = map_params[3]
                    else:
                        curr_p_map["value_sdf"] = map_params[2]

                # Set regularization
                if task_id == 0:
                    curr_p_map["eps_Regularization"] = self.params["cost_params"]["Regularization"]["eps"]

                for name in self.collision_link_names:
                    cbf_cst_type = False

                    if name in self.model_interface.scene.collision_link_names["static_obstacles"]:
                        if self.params["collision_constraint_type"]["static_obstacles"] == "SignedDistanceConstraintCBF":
                            cbf_cst_type = True
                    else:
                        if self.params["collision_constraint_type"][name] == "SignedDistanceConstraintCBF":
                            cbf_cst_type = True

                    if cbf_cst_type:
                        p_name = "_".join(["gamma", name])
                        curr_p_map[p_name] = self.params["collision_cbf_gamma"][name]

                # set initial guess
                stmpc_solver.set(k, 'x', x_bar_initial[k])
                if k < self.N:
                    stmpc_solver.set(k, 'u', u_bar_initial[k])

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
                        curr_p_map[p_name_W] = np.diag(self.params["cost_params"][tracking_cost_fcn_name]["P"])
                    else:
                        curr_p_map[p_name_W] = np.diag(self.params["cost_params"][tracking_cost_fcn_name]["Qk"])

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

                curr_p_map["Qqa_ControlEffort"] = self.params["cost_params"]["Effort"]["Qqa"]
                curr_p_map["Qqb_ControlEffort"] = self.params["cost_params"]["Effort"]["Qqb"]
                curr_p_map["Qva_ControlEffort"] = self.params["cost_params"]["Effort"]["Qva"]
                curr_p_map["Qvb_ControlEffort"] = self.params["cost_params"]["Effort"]["Qvb"]
                curr_p_map["Qua_ControlEffort"] = self.params["cost_params"]["Effort"]["Qua"]
                curr_p_map["Qub_ControlEffort"] = self.params["cost_params"]["Effort"]["Qub"]

                stmpc_solver.set(k, 'p', curr_p_map.cat.full().flatten())
                curr_p_map_bar.append(curr_p_map)

            t2 = time.perf_counter()
            self.log["time_ocp_set_params"][task_id] = t2 - t1

            # Log: initial cost
            self.log["cost_iter"][task_id, 0] = self.evaluate_cost_function(tracking_cost_fcn, self.x_bar, self.u_bar, curr_p_map_bar)

            # Solve stmpc
            t1 = time.perf_counter()
            stmpc_solver.solve_for_x0(xo, fail_on_nonzero_status=False)
            t2 = time.perf_counter()
            self.log["time_ocp_solve"][task_id] = t2 - t1

            stmpc_solver.print_statistics() # encapsulates: stat = ocp_solver.get_stats("statistics")

            if stmpc_solver.status !=0:

                x_bar = []
                u_bar = []
                print(f"xo: {xo}")
                for i in range(self.N):
                    print(f"stage {i}: x: {stmpc_solver.get(i, 'x')}")
                    x_bar.append(stmpc_solver.get(i, 'x'))
                    print(f"stage {i}: u: {stmpc_solver.get(i, 'u')}")
                    u_bar.append(stmpc_solver.get(i, 'u'))
                
                x_bar.append(stmpc_solver.get(self.N, 'x'))
                    
                for i in range(self.N):
                    print(f"stage {i}: lam: {stmpc_solver.get(i, 'lam')}")
                
                for i in range(self.N):
                    print(f"stage {i}: pi: {stmpc_solver.get(i, 'pi')}")

                for i in range(self.N):
                    print(f"stage {i}: sl: {stmpc_solver.get(i, 'sl')}")
                for i in range(self.N):
                    print(f"stage {i}: su: {stmpc_solver.get(i, 'su')}")
                    # v = self.evaluate_constraints(self.collisionCsts['sdf'], 
                    #                                     x_bar, u_bar, curr_p_map_bar)
                    # h = self.evaluate_sdf_h_fcn(self.collisionCsts['sdf'], 
                    #                             x_bar, u_bar, curr_p_map_bar)
                    # xdot = self.evaluate_sdf_xdot_fcn(self.collisionCsts['sdf'], 
                    #                             x_bar, u_bar, curr_p_map_bar)
                    # for i in range(self.N):
                    #     print(f"stage {i}: t: {stmpc_solver.get(i, 't')}")
                    #     print(f"state {i}: sdf: {v[i]}")
                    #     print(f"state {i}: h: {h[i]}")
                    #     print(f"state {i}: xdot: {xdot[i]}")
                

                self.log["iter_snapshot"][task_id] = {"t": t,
                                                    "xo": xo,
                                                    "p_map_bar": [p.cat.full().flatten() for p in curr_p_map_bar],
                                                    "x_bar_init": x_bar_initial,
                                                    "u_bar_init": u_bar_initial,
                                                    "x_bar": x_bar,
                                                    "u_bar": u_bar,
                                                    }

                # get iterate:
                solution = self.log["iter_snapshot"][task_id]

                lN = len(str(self.N+1))
                for i in range(self.N+1):
                    i_string = f'{i:0{lN}d}'
                    solution['x_'+i_string] = stmpc_solver.get(i,'x')
                    solution['u_'+i_string] = stmpc_solver.get(i,'u')
                    solution['z_'+i_string] = stmpc_solver.get(i,'z')
                    solution['lam_'+i_string] = stmpc_solver.get(i,'lam')
                    solution['t_'+i_string] = stmpc_solver.get(i, 't')
                    solution['sl_'+i_string] = stmpc_solver.get(i, 'sl')
                    solution['su_'+i_string] = stmpc_solver.get(i, 'su')
                    if i < self.N:
                        solution['pi_'+i_string] = stmpc_solver.get(i,'pi')

                # for k in list(solution.keys()):
                #     if len(solution[k]) == 0:
                #         del solution[k]

                # stmpc_solver.store_iterate(filename=str(self.output_dir / "iter_{:.2f}.json".format(t)))
                if self.params["acados"]["raise_exception_on_failure"]:
                    raise Exception(f'acados acados_ocp_solver returned status {self.solver_status}')
            else:
                self.log["iter_snapshot"][task_id] = None

            # get solution
            for i in range(self.N):
                x_bar_initial[i,:] = stmpc_solver.get(i, "x")
                u_bar_initial[i,:] = stmpc_solver.get(i, "u")
            x_bar_initial[self.N,:] = stmpc_solver.get(self.N, "x")

            # get e_p_bar for the hierarchy constraints
            e_p_bar = []
            for k in range(self.N+1):
                # Relaxed Lex Constraints |e_k| \leq |e^*_k| + eps
                rhs = np.abs(tracking_cost_fcn.get_e(x_bar_initial[k],[],r_bar_map[tracking_cost_fcn_name][k]))
                rhs += self.params["hierarchy_const_tol"]
                e_p_bar.append(rhs)
                self.py_logger.debug(f"task:{tracking_cost_fcn_name}, time{k}: e_p + eps {rhs}")
            
            e_p_bar_map[tracking_cost_fcn_name] = np.array(e_p_bar)
            self.py_logger.debug(f"cost {tracking_cost_fcn_name}: {stmpc_solver.get_cost()}")

            # Log: solver status, step size, cost after stmpc
            self.log["solver_status"][task_id] = stmpc_solver.status
            if stmpc.solver_options.nlp_solver_type != "SQP_RTI":
                self.log["step_size"][task_id] = np.mean(stmpc_solver.get_stats('alpha'))
            else:
                self.log["step_size"][task_id] = -1
            self.log["cost_iter"][task_id, 1] = self.evaluate_cost_function(tracking_cost_fcn, x_bar_initial, u_bar_initial, curr_p_map_bar)
            self.log["sqp_iter"][task_id] = stmpc_solver.get_stats('sqp_iter')
            self.log["qp_iter"][task_id] = sum(stmpc_solver.get_stats('qp_iter'))
            self.log["_".join(["ocp_param", str(task_id)])]=[p.cat.full().flatten() for p in curr_p_map_bar]
            self.log["x_bar"][task_id] =x_bar_initial
            self.log["u_bar"][task_id] = u_bar_initial


            curr_p_map_bars.append(curr_p_map_bar)

        self.t_bar = t + np.arange(self.N) * self.dt
        self.u_prev = self.u_bar[0].copy()
        self.x_bar = x_bar_initial.copy()
        self.u_bar = u_bar_initial.copy()

        self.ee_bar, self.base_bar = self._getEEBaseTrajectories(self.x_bar)
        self.sdf_bar["EE"] = self.model_interface.sdf_map.query_val(self.ee_bar[:, 0],self.ee_bar[:, 1],self.ee_bar[:, 2]).flatten()
        self.sdf_grad_bar["EE"] = self.model_interface.sdf_map.query_grad(self.ee_bar[:, 0],self.ee_bar[:, 1],self.ee_bar[:, 2]).reshape((3,-1))
        
        self.sdf_bar["base"] = self.model_interface.sdf_map.query_val(self.base_bar[:, 0], self.base_bar[:, 1], np.ones(self.N+1)*0.2)
        self.sdf_grad_bar["base"] = self.model_interface.sdf_map.query_grad(self.base_bar[:, 0], self.base_bar[:, 1], np.ones(self.N+1)*0.2).reshape((3,-1))

        # Log: final cost for each task after all stmpcs have been solved
        for i, tracking_cost_fcn in enumerate(tracking_cost_fcns):
            self.log["cost_final"][i] = self.evaluate_cost_function(tracking_cost_fcn, 
                                                             self.x_bar,
                                                             self.u_bar,
                                                             curr_p_map_bars[i])

        self.v_cmd = self.x_bar[0][self.robot.DoF:].copy()

        return self.v_cmd, self.u_prev, self.u_bar.copy(), self.x_bar[:, 9:].copy()
    
    def _get_log(self, task_num=2):
        log = {"cost_iter": np.zeros((task_num, 2)),
        "cost_final": np.zeros(task_num),
        "solver_status": np.zeros(task_num),
        "step_size": np.zeros(task_num),
        "sqp_iter": np.zeros(task_num),
        "qp_iter": np.zeros(task_num),
        "time_map_update": np.zeros(task_num),
        "time_ocp_set_params": np.zeros(task_num),
        "time_ocp_solve": np.zeros(task_num),
        "time_ocp_set_params_map" : np.zeros(task_num),
        "time_ocp_set_params_set_x" : np.zeros(task_num),
        "time_ocp_set_params_tracking" : np.zeros(task_num),
        "time_ocp_set_params_setp" : np.zeros(task_num),
        "x_bar": np.zeros((task_num, self.x_bar.shape[0], self.x_bar.shape[1])),
        "u_bar": np.zeros((task_num, self.u_bar.shape[0], self.u_bar.shape[1])),
        "iter_snapshot":[{},{}]
        }

        for i in range(task_num):
            log["_".join(["ocp_param", str(i)])] = []

        return log
    
    def reset(self):
        super().reset()
        for solver in self.stmpc_solvers:
            solver.reset()

if __name__ == "__main__":
    # robot mdl
    from mmseq_utils import parsing
    path_to_config = parsing.parse_ros_path({"package": "mmseq_run",
                                             "path":"config/self_collision_avoidance.yaml"})
    config = parsing.load_config(path_to_config)

    HTMPC(config["controller"])
