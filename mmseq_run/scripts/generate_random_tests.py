import argparse
import os
import yaml
import numpy.random as random
import numpy as np

from mmseq_plan.EEPlanner import *
from mmseq_plan.BasePlanner import *

REACHABLE_RADIUS = 1.83
#EE position under different arm configuration
#home: [1.48985, 0.17431, 0.705131]
#straight front: [1.8366, 0.173895, 0.67636]
#upright: [0.385042, 0.17387, 2.35772]


def generate_waypoint_tasks(args):
    random.seed(args.seed)
    num_case = args.number
    configs = []

    for i in range(num_case):
        planner_config = {"sot_type": "SoTStatic"}
        ee_task_config = EESimplePlanner.getDefaultParams()
        base_task_config = BaseSingleWaypoint.getDefaultParams()

        ree = random.rand() * 2
        theta = random.rand() * np.pi * 2
        h = random.rand() * 2.0 - 0.5
        ee_target_pos = [ree *  np.cos(theta),
                         ree * np.sin(theta),
                         h]
        ee_target_pos = [float(x) for x in ee_target_pos]
        ee_task_config["target_pos"] = ee_target_pos

        base_task_config["frame_id"] = "EE"
        rbase = random.rand() * 2. + REACHABLE_RADIUS
        theta = random.rand() * np.pi * 2
        base_target_pos = [ee_target_pos[0] + rbase * np.cos(theta),
                           ee_target_pos[1] + rbase * np.sin(theta)]
        base_target_pos = [float(x) for x in base_target_pos]
        base_task_config["target_pos"] = base_target_pos

        planner_config["tasks"] = [ee_task_config, base_task_config]

        configs.append({"planner": planner_config})

    for cid, config in enumerate(configs):
        config_path = os.path.join(args.folder, "test_{}.yaml".format(cid))
        with open(config_path, "w") as f:
            yaml.safe_dump(config, stream=f, default_flow_style=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', "--folder", required=True, help="Path to save data.")
    parser.add_argument('-n', "--number", type=int, default=10, help="Number of test cases.")
    parser.add_argument('-s', "--seed", type=int, default=0, help="Seed number.")
    parser.add_argument("--waypoint", action="store_true", help="Generate hierarchical waypoint tasks")

    args = parser.parse_args()
    if args.waypoint:
        generate_waypoint_tasks(args)