import argparse
import rosbag
from mmseq_utils.logging import ROSBagPlotter

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', "--bagfile", required=True, help="path to bag files")

    args = parser.parse_args()

    plotter = ROSBagPlotter(args.bagfile)
    plotter.plot_model_vs_groundtruth()
    plotter.plot_show()

if __name__ == "__main__":
    main()

