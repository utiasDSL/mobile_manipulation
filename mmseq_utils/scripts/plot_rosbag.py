import argparse
import rosbag
from mmseq_utils.plotting import ROSBagPlotter



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', "--bagfiles", nargs='+', required=True, help="path to bag files")

    parser.add_argument("--robot", action="store_true",
                        help="plot robot data")
    parser.add_argument("--estimation", action="store_true",
                        help="plot estimation data")
    parser.add_argument("--topichz", action="store_true",
                        help="plot topic frequency")
  
    args = parser.parse_args()

    plotter1 = ROSBagPlotter(args.bagfiles[0])
    if args.estimation:
        plotter1.plot_estimation()
    if args.topichz:
        plotter1.plot_topic_frequency()
    if args.robot:
        plotter1.plot_joint_states()
        plotter1.plot_joint_vel_tracking()
        plotter1.plot_tracking()

    plotter1.plot_show()

if __name__ == "__main__":
    main()

