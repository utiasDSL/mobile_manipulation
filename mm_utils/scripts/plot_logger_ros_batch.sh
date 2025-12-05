#!/bin/bash
# NOTE : Quote it else use array to avoid problems #
while getopts f: flag
do
    case "${flag}" in
        f) path_to_folder=${OPTARG};;
        *) echo "usage: $0 [-f] path to folder" >&2
           exit 1 ;;
    esac
done

FILES="${path_to_folder}/*"
for f in $FILES
do
  if [[ -d $f ]]; then
    rosrun mm_utils plot_logger_ros.py --folder $f --compare
  fi
done
