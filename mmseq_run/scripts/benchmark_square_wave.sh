#!/bin/bash
while getopts n:c:f: flag
do
    case "${flag}" in
        n) file_num=${OPTARG};;
        c) controller_type=${OPTARG};;
        f) path_to_folder=${OPTARG};;
        *) echo "usage: $0 [-n] number of cases [-c] controller type HTMPC/HTIDKC [-f] path to folder" >&2
           exit 1 ;;
    esac
done
use_mpc=false
if [ "$controller_type" = "MPC" ]; then
  use_mpc=true
fi

test_files=()
for (( fid=0; fid<file_num; fid++ ))
do
  test_files+=("$path_to_folder""/test_cases/test_""$fid"".yaml")
done

for i in "${!test_files[@]}"
do
  roslaunch mmseq_run simple_demo_sim.launch config:="$path_to_folder""/base.yaml" ctrl_config:=$(rospack find mmseq_run)/config/controller/"$controller_type".yaml planner_config:="${test_files[$i]}" use_mpc:="$use_mpc" logging_sub_folder:="test_$i/HTMPC_NewTol2"
done

#for i in "${!test_files[@]}"
#do
#  roslaunch mmseq_run simple_demo_sim.launch config:="$path_to_folder""/base.yaml" ctrl_config:="${test_files[$i]}" use_mpc:="$use_mpc" logging_sub_folder:="test_$i"
#done
#
