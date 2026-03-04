"""Metrics collection and analysis for MPSF experiments.

This module provides functions and classes for collecting, computing, and printing
metrics during MPSF (Model Predictive Shared Framework) experiments.
"""

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from mm_utils.math import wrap_pi_scalar


def compute_jerkiness(velocities, dt):
    """Compute jerkiness metric as the rate of change of velocity commands.

    Args:
        velocities (np.ndarray): Velocity commands, shape (N, nu).
        dt (float): Time step in seconds.

    Returns:
        float: Jerkiness metric. Returns 0.0 if len(velocities) < 2.
    """
    if len(velocities) < 2:
        return 0.0
    velocity_changes = np.diff(velocities, axis=0)
    jerkiness = np.mean(np.linalg.norm(velocity_changes, axis=1)) / dt
    return jerkiness


def compute_rmse(actual, reference, mask=None):
    """Compute Root Mean Square Error between actual and reference.

    Args:
        actual (np.ndarray): Actual values, shape (N, dim) or (dim,).
        reference (np.ndarray): Reference values, shape (N, dim) or (dim,).
        mask (np.ndarray, optional): Mask to apply, shape (dim,). Defaults to None.

    Returns:
        float: RMSE value. Returns 0.0 if actual or reference is None.
    """
    if actual is None or reference is None:
        return 0.0

    # Handle 1D case
    if actual.ndim == 1:
        actual = actual.reshape(1, -1)
    if reference.ndim == 1:
        reference = reference.reshape(1, -1)

    # Ensure same shape
    min_len = min(len(actual), len(reference))
    actual = actual[:min_len]
    reference = reference[:min_len]

    # Apply mask if provided
    if mask is not None:
        actual = actual * mask
        reference = reference * mask

    errors = actual - reference
    mse = np.mean(errors**2)
    rmse = np.sqrt(mse)
    return rmse


def extract_robot_states(robot, robot_states):
    """Extract base and end-effector pose and velocity from robot states.

    Args:
        robot: Robot instance with link_pose() and link_velocity() methods.
        robot_states (tuple): (joint_positions, joint_velocities) from robot.joint_states().

    Returns:
        dict: Dictionary with "base" and "EE" keys, each containing "pose" and
            "velocity" arrays. Base pose/vel shape (3,), EE pose/vel shape (6,).
    """
    ee_curr_pos, ee_cur_orn = robot.link_pose()
    ee_euler = Rot.from_quat(ee_cur_orn).as_euler("xyz")
    ee_pose = np.hstack([ee_curr_pos, ee_euler])

    ee_lin_vel, ee_ang_vel = robot.link_velocity()
    ee_vel = np.hstack([ee_lin_vel, ee_ang_vel])

    base_pose = robot_states[0][:3]
    base_vel = robot_states[1][:3]

    return {
        "base": {"pose": base_pose, "velocity": base_vel},
        "EE": {"pose": ee_pose, "velocity": ee_vel},
    }


def get_constraint_violations(controller, robot_states):
    """Extract constraint violations from measured robot states and controller log.

    Args:
        controller: Controller instance with log, collision_link_names, and robot attributes.
        robot_states (tuple): (q, v) tuple from robot.joint_states(), where q is positions
            and v is velocities.

    Returns:
        dict: Dictionary with constraint names as keys. Values are dicts with:
            - "max": Maximum violation value
            - "violations": Number of timesteps with violation > threshold
            - "max_per_dim" (optional): Per-dimension max violations for state/control
    """
    violations = {}
    violation_threshold = 1e-3  # 1mm or 0.001rad for measured violations
    collision_violation_threshold = 1e-3

    # Collision constraints from log
    for name in controller.collision_link_names:
        constraint_key = "_".join([name, "constraint"])
        constraint_vals = controller.log.get(constraint_key)
        if (
            constraint_vals
            and isinstance(constraint_vals, list)
            and len(constraint_vals) > 0
        ):
            # Each element in constraint_vals is a timestep
            all_values = []
            timesteps_with_violation = 0
            for v in constraint_vals:
                # Convert CasADi objects to numpy arrays
                if hasattr(v, "full"):
                    v = v.full()
                if not isinstance(v, np.ndarray):
                    v = np.array(v)

                v_flat = v.flatten()
                all_values.extend(v_flat)
                if np.any(v_flat > collision_violation_threshold):
                    timesteps_with_violation += 1

            # Max value (closest to boundary, can be negative if satisfied)
            max_value = np.max(all_values) if all_values else 0.0

            violations[name] = {
                "max": max_value,
                "violations": timesteps_with_violation,
            }

    # Measured state constraints - evaluate from actual robot states
    q_meas, v_meas = robot_states
    q_meas_wrapped = q_meas.copy()
    for i in range(2, len(q_meas)):  # Wrap revolute joints (base yaw + arm joints)
        q_meas_wrapped[i] = wrap_pi_scalar(q_meas[i])
    x_meas = np.hstack([q_meas_wrapped, v_meas])

    # Compute violations: positive means violation
    vio_upper = x_meas - controller.robot.ub_x
    vio_lower = controller.robot.lb_x - x_meas
    vio_per_dim = np.maximum(vio_upper, vio_lower)

    max_vio = float(np.max(vio_per_dim)) if vio_per_dim.size else 0.0
    has_violation = max_vio > violation_threshold

    # Print violation details if significant
    if has_violation:
        argmax_vio = int(np.argmax(vio_per_dim))
        nq = len(q_meas)
        dim_name = f"q[{argmax_vio}]" if argmax_vio < nq else f"v[{argmax_vio - nq}]"
        print(
            "EXPERIMENT - "
            f"State bound violation: max={max_vio:.6f} at {dim_name} "
            f"(x={x_meas[argmax_vio]:.6f}, "
            f"lb={controller.robot.lb_x[argmax_vio]:.6f}, "
            f"ub={controller.robot.ub_x[argmax_vio]:.6f})"
        )

    violations["state"] = {
        "max": max_vio,
        "violations": 1 if has_violation else 0,
        "max_per_dim": vio_per_dim,
    }

    # Control constraints - evaluate from u_bar if available (still useful for MPC diagnostics)
    if (
        hasattr(controller, "controlCst")
        and hasattr(controller, "x_bar")
        and hasattr(controller, "u_bar")
    ):
        try:
            nlp_p_map_bar = controller.log.get("ocp_param", [])
            if not nlp_p_map_bar:
                nlp_p_map_bar = [{}] * (controller.N + 1)

            control_vals = controller.evaluate_constraints(
                controller.controlCst, controller.x_bar, controller.u_bar, nlp_p_map_bar
            )

            if control_vals and len(control_vals) > 0:
                # Each element is a timestep
                nu = controller.robot.ssSymMdl["nu"]
                all_values = []
                timesteps_with_violation = 0
                per_dim_max = np.full(
                    nu, -np.inf
                )  # Start with -inf to get max correctly

                for v in control_vals:
                    # Convert CasADi objects to numpy arrays
                    if hasattr(v, "full"):
                        v = v.full()
                    v_arr = np.array(v).flatten()
                    all_values.extend(v_arr)

                    # Check if this timestep has any violation
                    if np.any(v_arr > violation_threshold):
                        timesteps_with_violation += 1

                    upper_values = v_arr[:nu]
                    lower_values = v_arr[nu:]
                    per_dim_max = np.maximum(
                        per_dim_max, np.maximum(upper_values, lower_values)
                    )

                # Max value (closest to boundary, can be negative if satisfied)
                max_value = np.max(all_values) if all_values else 0.0

                violations["control"] = {
                    "max": max_value,
                    "violations": timesteps_with_violation,
                    "max_per_dim": per_dim_max,
                }
        except Exception:
            pass  # Skip if evaluation fails

    return violations


class MPSFMetricsCollector:
    """Collects and manages MPSF experiment metrics over time.

    This class provides a clean interface for collecting metrics during MPSF
    experiments, whether running in simulation or on a real robot via ROS.
    """

    def __init__(self):
        """Initialize an empty metrics collector."""
        self.metrics = {
            "base_rmses": [],
            "ee_rmses": [],
            "jerks": [],
            "control_efforts": [],
            "constraint_violations": [],
        }
        self.u_prev = None

    def update(
        self,
        references,
        states,
        u,
        desired_base_vel,
        desired_ee_vel,
        controller,
        robot_states,
        sim_timestep,
    ):
        """Update metrics with current timestep data.

        Args:
            references (dict): Reference trajectories with optional "base_pose" and
                "ee_pose" keys, shape (N+1, dim).
            states (dict): Current robot states with "base" and "EE" keys.
            u (np.ndarray): Current velocity command, shape (nu,).
            desired_base_vel (np.ndarray or None): Desired base velocity, shape (3,).
            desired_ee_vel (np.ndarray or None): Desired EE velocity, shape (6,).
            controller: Controller instance with mask attributes and log.
            robot_states (tuple): (q, v) tuple from robot.joint_states().
            sim_timestep (float): Simulation timestep in seconds.
        """
        # RMSE
        base_pose_ref = references.get("base_pose")
        if base_pose_ref is not None:
            base_rmse = compute_rmse(
                states["base"]["pose"], base_pose_ref[0], controller.base_mask
            )
            self.metrics["base_rmses"].append(base_rmse)

        ee_pose_ref = references.get("ee_pose")
        if ee_pose_ref is not None:
            ee_rmse = compute_rmse(
                states["EE"]["pose"], ee_pose_ref[0], controller.ee_mask
            )
            self.metrics["ee_rmses"].append(ee_rmse)

        # Jerkiness
        if self.u_prev is not None:
            jerkiness = compute_jerkiness(np.vstack([self.u_prev, u]), sim_timestep)
            self.metrics["jerks"].append(jerkiness)

        # Control effort (L2 norm of velocity command)
        control_effort = np.linalg.norm(u)
        self.metrics["control_efforts"].append(control_effort)

        # Constraint violations (using measured states)
        violations = get_constraint_violations(controller, robot_states)
        self.metrics["constraint_violations"].append(violations)

        # Update previous velocity for next iteration
        self.u_prev = u.copy()

    def reset(self):
        """Reset all collected metrics."""
        self.metrics = {
            "base_rmses": [],
            "ee_rmses": [],
            "jerks": [],
            "control_efforts": [],
            "constraint_violations": [],
        }
        self.u_prev = None

    def print_summary(self):
        """Print formatted summary of collected MPSF experiment metrics."""
        print("\n" + "=" * 80)
        print("MPSF EXPERIMENT METRICS SUMMARY")
        print("=" * 80)

        # RMSE
        if self.metrics["base_rmses"]:
            print(
                f"Base Pose RMSE (average): {np.mean(self.metrics['base_rmses']):.4f}"
            )
        else:
            print("Base Pose RMSE: N/A (no base pose references)")

        if self.metrics["ee_rmses"]:
            print(f"EE Pose RMSE (average): {np.mean(self.metrics['ee_rmses']):.4f}")
        else:
            print("EE Pose RMSE: N/A (no EE pose references)")

        # Jerkiness
        if self.metrics["jerks"]:
            print(f"Mean Jerkiness: {np.mean(self.metrics['jerks']):.4f} m/s³")

        # Control effort
        if self.metrics["control_efforts"]:
            print(
                f"Mean Control Effort: {np.mean(self.metrics['control_efforts']):.4f} m/s/s"
            )
            print(
                f"Max Control Effort: {np.max(self.metrics['control_efforts']):.4f} m/s/s"
            )

        # Constraint violations
        if any(self.metrics["constraint_violations"]):
            print("\nConstraint Violations Summary:")
            all_names = set()
            for violations in self.metrics["constraint_violations"]:
                all_names.update(violations.keys())

            for name in sorted(all_names):
                # Collect all violations for this constraint across all timesteps
                constraint_data = [
                    v.get(name)
                    for v in self.metrics["constraint_violations"]
                    if name in v and isinstance(v.get(name), dict)
                ]

                if not constraint_data:
                    continue

                # Get max across all timesteps
                max_v = max([d.get("max", 0.0) for d in constraint_data])
                # Count how many simulation timesteps had violations
                timesteps_with_violation = sum(
                    [1 for d in constraint_data if d.get("violations", 0) > 0]
                )
                total_steps = len(constraint_data)

                # Check if per-dimension data exists
                has_per_dim = any("max_per_dim" in d for d in constraint_data)

                if has_per_dim:
                    # Get max per dimension across all timesteps
                    per_dim_arrays = [
                        d.get("max_per_dim")
                        for d in constraint_data
                        if "max_per_dim" in d
                    ]
                    if per_dim_arrays:
                        per_dim_max = np.max(per_dim_arrays, axis=0)
                        per_dim_str = ", ".join([f"{v:.4f}" for v in per_dim_max])
                        print(
                            f"  {name}: max={max_v:.6f}, violations={timesteps_with_violation}/{total_steps}, per_dim_max=[{per_dim_str}]"
                        )
                    else:
                        print(
                            f"  {name}: max={max_v:.6f}, violations={timesteps_with_violation}/{total_steps}"
                        )
                else:
                    print(
                        f"  {name}: max={max_v:.6f}, violations={timesteps_with_violation}/{total_steps}"
                    )

        print("=" * 80 + "\n")
