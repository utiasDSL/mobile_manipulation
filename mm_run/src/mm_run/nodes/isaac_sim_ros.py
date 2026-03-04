import argparse
import datetime
import os
import sys

import numpy as np
import rospy
from omni.isaac.kit import SimulationApp

from mm_utils import parsing
from mm_utils.logging import DataLogger

# URDF import, configuration and simulation sample
simulation_app = SimulationApp({"headless": False})

from omni.isaac.core.utils import extensions  # noqa: E402

extensions.enable_extension("omni.isaac.ros_bridge")
from mm_sim_isaac.isaac_sim_env import IsaacSimEnv  # noqa: E402
from omni.isaac.core.utils.rotations import euler_to_rot_matrix  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    argv = rospy.myargv(argv=sys.argv)

    parser.add_argument(
        "-c", "--config", required=True, help="Path to configuration file."
    )
    parser.add_argument(
        "--logging_sub_folder",
        type=str,
        help="save data in a sub folder of logging directory",
    )

    args = parser.parse_args(argv[1:])

    config = parsing.load_config(args.config)

    if args.logging_sub_folder != "default":
        config["logging"]["log_dir"] = os.path.join(
            config["logging"]["log_dir"], args.logging_sub_folder
        )

    sim_config = config["simulation"]

    rospy.init_node("isaac_sim_ros")

    # Create shared timestamp for logging (format: YYYY-MM-DD_HH-MM-SS)
    timestamp = datetime.datetime.now()
    session_timestamp = timestamp.strftime("%Y-%m-%d_%H-%M-%S")

    # Set ROS parameter so control node can use the same timestamp
    rospy.set_param("/experiment_timestamp", session_timestamp)

    sim = IsaacSimEnv(sim_config)
    robot = sim.robot
    world = sim.world
    robot_ros_interface = sim.robot_ros_interface

    # disable gravity to use joint velocity control
    robot.disable_gravity()

    # init logger
    logger = DataLogger(config, name="sim")
    logger.add("sim_timestep", sim.timestep)
    logger.add("duration", sim.duration)

    logger.add("nq", sim_config["robot"]["dims"]["q"])
    logger.add("nv", sim_config["robot"]["dims"]["v"])
    logger.add("nx", sim_config["robot"]["dims"]["x"])
    logger.add("nu", sim_config["robot"]["dims"]["u"])

    # if no cmd_vel comes, brake
    while not robot_ros_interface.ready() and simulation_app.is_running():
        sim.publish_feedback()
        sim.apply_joint_velocities()

        sim.step(render=True)
        sim.publish_ros_topics()

    t0 = world.current_time
    t = world.current_time
    while (
        not rospy.is_shutdown()
        and t - t0 <= sim.duration
        and simulation_app.is_running()
    ):
        t = world.current_time

        q = robot.get_joint_positions()
        v = robot.get_joint_velocities()

        # convert base cmd_vel from base frame to world frame
        base_cmd_vel_b = robot_ros_interface.base.cmd_vel
        R_wb = euler_to_rot_matrix([0, 0, q[2]], extrinsic=False)
        base_cmd_vel = R_wb @ base_cmd_vel_b
        cmd_vel_world = np.concatenate((base_cmd_vel, robot_ros_interface.arm.cmd_vel))
        sim.apply_joint_velocities(cmd_vel_world)

        robot_ros_interface.publish_feedback(t=t, q=q, v=v)

        # log
        (r_ew_w, Q_we), (v_ew_w, ω_ew_w) = robot.tool_state()
        logger.append("ts", t)
        logger.append("xs", np.hstack((q, v)))
        logger.append("cmd_vels", cmd_vel_world)
        logger.append("r_ew_ws", r_ew_w)
        logger.append("Q_wes", Q_we)
        logger.append("v_ew_ws", v_ew_w)
        logger.append("ω_ew_ws", ω_ew_w)

        logger.append("r_bw_ws", q[:2])
        logger.append("yaw_bw_ws", q[2])
        logger.append("v_bw_ws", v[:2])
        logger.append("ω_bw_ws", v[2])

        sim.step(render=True)
        sim.publish_ros_topics()

    logger.save(session_timestamp=session_timestamp)
    simulation_app.close()


if __name__ == "__main__":
    main()
