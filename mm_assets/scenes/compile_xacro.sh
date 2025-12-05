#!/bin/sh

# Compile xacro versions of URDFs to regular versions for consumption by libraries.
# This is mainly used to resolve ROS package paths without hardcoding.
mkdir -p urdf
xacro xacro/restaurant.urdf.xacro -o urdf/restaurant.urdf
