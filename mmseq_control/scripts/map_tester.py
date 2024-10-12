import rospy
import numpy as np
import time
from mobile_manipulation_central.ros_interface import MapInterface, MapInterfaceNew, MapGridInterface
from mmseq_control.map import SDF2D, SDF3D, SDF2DNew, SDF3DNew
import matplotlib.pyplot as plt

def test_3d(config):

    map_ros_interface = MapGridInterface(config["controller"])
    sdf = SDF3DNew(config["controller"])
    rate = rospy.Rate(100)

    while not map_ros_interface.ready() and not rospy.is_shutdown():
        rate.sleep()

    while not rospy.is_shutdown():
        is_map_updated, map = map_ros_interface.get_map()
        if is_map_updated:
            sdf.update_map(*map)
            # tsdf, tsdf_vals = map_ros_interface.tsdf, map_ros_interface.tsdf_vals

            if False:
                #use tsdf range
                pts = np.around(np.array([np.array([p.x,p.y]) for p in tsdf]), 2).reshape((len(tsdf),2))
                xs = pts[:,0]
                ys = pts[:,1]
                x_lim = [min(xs), max(xs)]
                y_lim = [min(ys), max(ys)]
                print(f"using tsdf range")
                print(f"x:{x_lim}, y:{y_lim}")
            else:
                # use robot local map range
                params = sdf.get_params()
                x_lim = [min(params[0]), max(params[0])]
                y_lim = [min(params[1]), max(params[1])]
                z_lim = [min(params[2]), max(params[2])]
                print(f"using local map range")
                print(f"x:{x_lim}, y:{y_lim}, z:{z_lim}")

            # x-y
            sdf.vis(x_lim=x_lim,
                    y_lim=y_lim,
                    z_lim=[0.1, 0.1],
                    block=False)
            sdf.vis(x_lim=x_lim,
                    y_lim=y_lim,
                    z_lim=[0.7, 0.7],
                    block=True)
            # sdf.vis(x_lim=x_lim,
            #         y_lim=y_lim,
            #         z_lim=[1.3, 1.3],
            #         block=False)
            # y-z
            # sdf.vis(x_lim=[0, 0],
            #         y_lim=y_lim,
            #         z_lim=[0.1, 1.4],
            #         block=False)
            # sdf.vis(x_lim=[0.5, 0.5],
            #         y_lim=y_lim,
            #         z_lim=[0.1, 1.4],
            #         block=False)
            # sdf.vis(x_lim=[1.0, 1.0],
            #         y_lim=y_lim,
            #         z_lim=[0.1, 1.4],
            #         block=False)
            # # x-z
            # sdf.vis(x_lim=x_lim,
            #         y_lim=[0,0],
            #         z_lim=[0.1, 1.4],
            #         block=False)
            # sdf.vis(x_lim=x_lim,
            #         y_lim=[0.5,0.5],
            #         z_lim=[0.1, 1.4],
            #         block=False)
            # sdf.vis(x_lim=x_lim,
            #         y_lim=[-0.5, -0.5],
            #         z_lim=[0.1, 1.4],
            #         block=True)
            # sdf.vis3d(x_lim=x_lim,
            #           y_lim=y_lim,
            #           z_lim=[0.1, 1.5],
            #           block=True)
                

        rate.sleep()

def test_2d(config):
    map_ros_interface = MapInterfaceNew(config["controller"])
    sdf = SDF2DNew(config["controller"])
    rate = rospy.Rate(100)

    while not map_ros_interface.ready() and not rospy.is_shutdown():
        rate.sleep()

    while not rospy.is_shutdown():
        _, map = map_ros_interface.get_map()
        sdf.update_map(*map)
        tsdf, tsdf_vals = map_ros_interface.tsdf, map_ros_interface.tsdf_vals
        pts = np.around(np.array([np.array([p.x,p.y]) for p in tsdf]), 2).reshape((len(tsdf),2))
        xs = pts[:,0]
        ys = pts[:,1]
        x_lim = [min(xs), max(xs)]
        y_lim = [min(ys), max(ys)]
        if map_ros_interface.valid:
            sdf.vis(x_lim=x_lim,
                    y_lim=y_lim,
                    block=True)
        else:
            print("map invalid")

        rate.sleep()

if __name__ == "__main__":
    rospy.init_node("map_tester")
    from mmseq_utils import parsing
    import argparse
    import sys
    config_path = parsing.parse_ros_path({"package": "mmseq_run",
                                          "path": "config/3d_collision_sdf.yaml"})
    config = parsing.load_config(config_path)

    argv = rospy.myargv(argv=sys.argv)
    parser = argparse.ArgumentParser()
    parser.add_argument("--map2d", action="store_true",
                        help="Test SDF3DNew")
    parser.add_argument("--map3d", action="store_true",
                        help="Test SDF2DNew")

    args = parser.parse_args(argv[1:])

    if args.map2d:
        test_2d(config)
    elif args.map3d:
        test_3d(config)