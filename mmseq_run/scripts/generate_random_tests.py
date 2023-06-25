import argparse
import os
import yaml
import numpy.random as random
import matplotlib.pyplot as plt

from mmseq_plan.EEPlanner import *
from mmseq_plan.BasePlanner import *

REACHABLE_RADIUS = 1.83
EE_POS_HOME = [1.48985, 0.17431, 0.705131]
#EE position under different arm configuration
#home: [1.48985, 0.17431, 0.705131]
#straight front: [1.8366, 0.173895, 0.67636]
#upright: [0.385042, 0.17387, 2.35772]

def save_configs(configs, folder_path):
    for cid, config in enumerate(configs):
        config_path = os.path.join(folder_path, "test_{}.yaml".format(cid))
        with open(config_path, "w") as f:
            yaml.safe_dump(config, stream=f, default_flow_style=False)

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

    save_configs(configs, args.folder)


def generate_squarewave_tasks(args):
    random.seed(args.seed)
    num_case = args.number
    configs = []
    base_peaks = []

    for i in range(num_case):
        planner_config = {"sot_type": "SoTStatic"}
        ee_task_config = EESimplePlanner.getDefaultParams()
        base_task_config = BasePosTrajectorySqaureWave.getDefaultParams()

        while True:
            rbase = random.rand() * 0.25 + 0.25 + REACHABLE_RADIUS
            theta = random.rand() * np.pi * 2

            base_target_pos = np.array([rbase * np.cos(theta),
                               rbase * np.sin(theta)])
            base_target_pos += EE_POS_HOME[:2]

            # base waypoint can't be too far since HTIDKC tend to violate acceleration constraints
            if np.linalg.norm(base_target_pos) < 3:
                break
        base_task_config["frame_id"] = "base"
        base_task_config["peak_pos"] = [float(x) for x in base_target_pos]
        base_task_config["period"] = 16
        base_task_config["round"] = 1

        planner_config["tasks"] = [ee_task_config, base_task_config]

        configs.append({"planner": planner_config})
        base_peaks.append(base_target_pos)

    save_configs(configs, os.path.join(args.folder, "test_cases"))
    base_peaks = np.array(base_peaks)

    f = plt.figure()
    ax = plt.gca()
    plt.plot(base_peaks[:, 0], base_peaks[:, 1], 'o', markersize=5, label="Base Target (Peak)")
    plt.plot(0,0, '.', markersize=8, label="Base Home (Valley)")
    plt.plot(EE_POS_HOME[0], EE_POS_HOME[1], '.', markersize=8, label="EE Home")
    reachable_area = plt.Circle(EE_POS_HOME[:2], radius=REACHABLE_RADIUS, facecolor="green", alpha=0.5, label="Reachable Area")
    ax.add_patch(reachable_area)
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.legend(loc="upper right")
    ax.set_aspect("equal")

    plt.show()





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', "--folder", required=True, help="Path to save data.")
    parser.add_argument('-n', "--number", type=int, default=10, help="Number of test cases.")
    parser.add_argument('-s', "--seed", type=int, default=0, help="Seed number.")
    parser.add_argument("--waypoint", action="store_true", help="Generate hierarchical waypoint tasks")
    parser.add_argument("--squarewave", action="store_true", help="Generate hierarchical waypoint tasks")

    args = parser.parse_args()
    if args.waypoint:
        generate_waypoint_tasks(args)
    elif args.squarewave:
        generate_squarewave_tasks(args)