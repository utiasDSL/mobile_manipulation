import argparse
import rosbag
from mmseq_utils.logging import ROSBagPlotter

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', "--bagfiles", nargs='+', required=True, help="path to bag files")
    parser.add_argument("--htmpc", action="store_true", help="plot htmpc info")
    parser.add_argument("--tracking", action="store_true",
                        help="plot tracking data")
    parser.add_argument("--robot", action="store_true",
                        help="plot tracking data")

    args = parser.parse_args()

    plotter1 = ROSBagPlotter(args.bagfiles[0])
    # plotter2 = ROSBagPlotter(args.bagfiles[1])
    plotter1.plot_joint_states()
    plotter1.plot_joint_vel_tracking()
    plotter1.plot_tracking()
    # plotter2.plot_tracking(subscript="_damped")
    plotter1.plot_show()

if __name__ == "__main__":
    main()

