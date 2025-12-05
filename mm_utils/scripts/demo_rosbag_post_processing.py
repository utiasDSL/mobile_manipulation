import argparse

import rosbag
import rospy
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker

RBY_COLORS = [[1, 0, 0, 1], [0, 0, 1, 1], [1, 1, 0, 1]]


def make_marker(marker_type, id, rgba, scale, lifetime=0.1):
    m = Marker()
    m.id = id
    m.type = marker_type
    m.action = Marker.ADD

    m.scale.x = scale[0]
    m.scale.y = scale[1]
    m.scale.z = scale[2]
    m.color.r = rgba[0]
    m.color.g = rgba[1]
    m.color.b = rgba[2]
    m.color.a = rgba[3]
    m.lifetime = rospy.Duration.from_sec(lifetime)

    m.pose.orientation.w = 1

    return m


def update_plan_visualization_msg(msg):
    id = msg.id
    index = id // 2

    c = RBY_COLORS[index]
    msg.color.r = c[0]
    msg.color.g = c[1]
    msg.color.b = c[2]
    msg.color.a = c[3]


def add_human_visualization(swarm_state_msg):
    marker_msgs = []
    for i, pose in enumerate(swarm_state_msg.poses):
        marker = make_marker(
            Marker.CYLINDER, i, RBY_COLORS[i], [0.2, 0.2, pose.position.z], lifetime=0.1
        )
        marker.pose.position = Point(
            pose.position.x, pose.position.y, pose.position.z / 2
        )
        marker_msgs.append(marker)

        marker.header.stamp = swarm_state_msg.header.stamp
        marker.header.frame_id = swarm_state_msg.header.frame_id

    return marker_msgs


def process_bag(path_to_input_bag, output_file_name):
    bag_in = rosbag.Bag(path_to_input_bag, mode="r")

    path_to_folder = path_to_input_bag.split("/")[:-1]
    path_to_output_bag = "/".join(path_to_folder + [output_file_name])
    bag_out = rosbag.Bag(path_to_output_bag, mode="w")

    topics = [
        "/ur10/joint_states",
        "/ridgeback/joint_states",
        "/controller_visualization",
        "/swarm_state",
        "/current_plan_visualization",
        "/plan_visualization",
    ]

    for topic, msg, t in bag_in.read_messages(topics=topics):
        # This also replaces tf timestamps under the assumption
        # that all transforms in the message share the same timestamp
        if topic == "/plan_visualization":
            update_plan_visualization_msg(msg)
        if topic == "/swarm_state":
            marker_msgs = add_human_visualization(msg)
            for marker_msg in marker_msgs:
                bag_out.write("/human_visualization", marker_msg, t)

        bag_out.write(topic, msg, t)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="Path to input bag file.")
    parser.add_argument("-o", "--output", required=True, help="output file name.")
    args = parser.parse_args()
    process_bag(args.input, args.output)
