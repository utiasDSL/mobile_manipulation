import argparse
import datetime
import time

import numpy as np

from mm_simulator import simulation
from mm_utils import parsing
from mm_utils.logging import DataLogger
from mm_utils.plotting import DataPlotter


def main():
    np.set_printoptions(precision=3, suppress=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to configuration file.")
    parser.add_argument(
        "--jointId", type=int, default=0, help="Joint index to simulate step response."
    )
    parser.add_argument(
        "--video",
        nargs="?",
        default=None,
        const="",
        help="Record video. Optionally specify prefix for video directory.",
    )
    args = parser.parse_args()

    # load configuration
    config = parsing.load_config(args.config)
    sim_config = config["simulation"]

    # start the simulation
    timestamp = datetime.datetime.now()
    sim = simulation.BulletSimulation(
        config=sim_config, timestamp=timestamp, cli_args=args
    )
    robot = sim.robot
    robot.command_velocity(np.zeros(9))
    sim.settle(5.0)

    # initial time, state, input
    t = 0.0
    vd = np.zeros(robot.nv)
    vd[args.jointId] = -0.0

    # init logger
    logger = DataLogger(config, name="simple_sim")

    logger.add("sim_timestep", sim.timestep)
    logger.add("duration", sim.duration)

    logger.add("nq", sim_config["robot"]["dims"]["q"])
    logger.add("nv", sim_config["robot"]["dims"]["v"])
    logger.add("nx", sim_config["robot"]["dims"]["x"])
    logger.add("nu", sim_config["robot"]["dims"]["u"])
    log = True
    while t <= sim.duration:
        q, v = sim.robot.joint_states(add_noise=False)
        x = np.hstack((q, v))
        robot.command_velocity(vd)

        # log sim stuff
        if log:
            logger.append("ts", t)
            logger.append("xs", x)
            logger.append("cmd_vels", vd)

        t, _ = sim.step(t)
        time.sleep(sim.timestep)

    plotter = DataPlotter.from_logger(logger)
    plotter.plot_cmd_vs_real_vel()
    plotter.show()


if __name__ == "__main__":
    main()
