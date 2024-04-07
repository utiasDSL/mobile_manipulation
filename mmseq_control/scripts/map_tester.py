import rospy
import numpy as np

from mobile_manipulation_central.ros_interface import MapInterface
from mmseq_control.map import SDF2D, SDF3D

def test_3d():

    map_ros_interface = MapInterface(topic_name="/pocd_slam_node/occupied_ef_dist_nodes")
    # sdf = SDF2D()
    sdf = SDF3D()
    rate = rospy.Rate(100)

    while not map_ros_interface.ready() and not rospy.is_shutdown():
        rate.sleep()

    while not rospy.is_shutdown():
        is_map_updated, map = map_ros_interface.get_map()
        if is_map_updated:
            sdf.update_map(*map)
            print(f"{rospy.get_time()} map updated")
            tsdf, _ = map
            pts = np.around(np.array([np.array([p.x,p.y]) for p in tsdf]), 2).reshape((len(tsdf),2))
            xs = pts[:,0]
            ys = pts[:,1]
            x_lim = [min(xs), max(xs)]
            y_lim = [min(ys), max(ys)]
            if sdf.valid:
                sdf.vis(x_lim=x_lim,
                        y_lim=y_lim,
                        z_lim=[0.5, 0.5],
                        block=True)
            else:
                print("map invalid")

        rate.sleep()

def test_2d():

    map_ros_interface = MapInterface(topic_name="/pocd_slam_node/occupied_ef_dist_nodes")
    sdf = SDF2D()
    rate = rospy.Rate(100)

    while not map_ros_interface.ready() and not rospy.is_shutdown():
        rate.sleep()

    while not rospy.is_shutdown():
        is_map_updated, map = map_ros_interface.get_map()
        if is_map_updated:
            sdf.update_map(*map)
            print(f"{rospy.get_time()} map updated")
            tsdf, _ = map
            pts = np.around(np.array([np.array([p.x,p.y]) for p in tsdf]), 2).reshape((len(tsdf),2))
            xs = pts[:,0]
            ys = pts[:,1]
            x_lim = [min(xs), max(xs)]
            y_lim = [min(ys), max(ys)]
            if sdf.valid:
                sdf.vis(x_lim=x_lim,
                        y_lim=y_lim,
                        block=True)
            else:
                print("map invalid")

        rate.sleep()

if __name__ == "__main__":
    rospy.init_node("map_tester")
    test_3d()
    # test_2d()