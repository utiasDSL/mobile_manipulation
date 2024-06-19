#!/bin/bash
while getopts n:c:f: flag
do
    case "${flag}" in
        n) test_num=${OPTARG};;
        f) folder_name=${OPTARG};;
        *) echo "usage: $0 [-n] number of cases [-f] data folder name" >&1
           exit 1 ;;
    esac
done

use_mpc=true
config="$(rospack find mmseq_run)/config/ral/self_collision_avoidance.yaml"

for (( tid=0; tid<$test_num; tid++ ))
do
  roslaunch mmseq_run simple_demo_sim.launch config:=$config use_mpc:=$use_mpc logging_sub_folder:="test_$tid/$folder_name"
done
