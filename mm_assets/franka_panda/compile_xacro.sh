#!/bin/sh

# Compile xacro versions of URDFs to regular versions for consumption by libraries.
# This is mainly used to resolve ROS package paths without hardcoding.
mkdir -p urdf
xacro xacro/panda.urdf.xacro -o urdf/panda.urdf
xacro xacro/panda_tray.urdf.xacro -o urdf/panda_tray.urdf
