import argparse

import rosbag
import rospy


def extract_last_message(bag_file, topic_name):
    last_msg = None
    last_msg_time = None

    with rosbag.Bag(bag_file, "r") as bag:
        for topic, msg, t in bag.read_messages(topics=[topic_name]):
            last_msg = msg
            last_msg_time = t

    return last_msg, last_msg_time


def publish_message_at_rate(topic_name, message, rate_hz):
    rospy.init_node("last_message_publisher", anonymous=True)
    pub = rospy.Publisher(topic_name, type(message), queue_size=10)
    rate = rospy.Rate(rate_hz)

    rospy.loginfo(f"Publishing to {topic_name} at {rate_hz} Hz")

    while not rospy.is_shutdown():
        pub.publish(message)
        rate.sleep()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract and publish the last message of a topic from a ROS bag file."
    )
    parser.add_argument("bag_file", type=str, help="Path to the ROS bag file")
    parser.add_argument(
        "topic_name", type=str, help="The topic name to extract and publish"
    )
    parser.add_argument(
        "rate_hz",
        type=float,
        default=10,
        help="The rate (in Hz) at which to publish the message",
    )

    args = parser.parse_args()

    bag_file = args.bag_file
    topic_name = args.topic_name
    rate_hz = args.rate_hz

    last_message, last_message_time = extract_last_message(bag_file, topic_name)

    if last_message:
        rospy.loginfo(f"Last message from {topic_name} at {last_message_time}")
        publish_message_at_rate(topic_name, last_message, rate_hz)
    else:
        rospy.logwarn(f"No messages found in topic {topic_name} in the given bag file.")
