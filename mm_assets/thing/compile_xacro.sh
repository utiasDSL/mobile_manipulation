#!/bin/sh

# Compile xacro versions of URDFs to regular versions for consumption by libraries.
# This is mainly used to resolve ROS package paths without hardcoding.
mkdir -p urdf
xacro xacro/test.urdf.xacro -o urdf/test.urdf
