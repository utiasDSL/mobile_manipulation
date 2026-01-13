"""Utilities for parsing general configuration dictionaries."""

import os
import re
import subprocess
from pathlib import Path

import numpy as np
import rospkg
import xacro
import yaml


def recursive_dict_update(default, custom):
    """Recursively merge custom dictionary into default dictionary.

    Args:
        default (dict): Default dictionary (will be modified in-place).
        custom (dict): Custom dictionary with overrides.

    Returns:
        dict: Merged dictionary (same object as default).
    """
    if not isinstance(default, dict) or not isinstance(custom, dict):
        raise TypeError("Params of recursive_update should be dicts")

    for key in custom:
        if isinstance(custom[key], dict) and isinstance(default.get(key), dict):
            default[key] = recursive_dict_update(default[key], custom[key])
        else:
            default[key] = custom[key]

    return default


def load_config(path, depth=0, max_depth=5):
    """Load configuration file with support for included files.

    `depth` and `max_depth` arguments are provided to protect against
    unexpectedly deep or infinite recursion through included files.

    Args:
        path (str): Path to configuration file.
        depth (int): Current recursion depth (used internally).
        max_depth (int): Maximum allowed recursion depth.

    Returns:
        dict: Merged configuration dictionary.

    Raises:
        Exception: If maximum inclusion depth is exceeded.
    """
    if depth > max_depth:
        raise Exception(f"Maximum inclusion depth {max_depth} exceeded.")

    with open(parse_path(path)) as f:
        d = yaml.safe_load(f)

    # get the includes while also removing them from the dict
    includes = d.pop("include", [])

    # construct a dict of everything included
    includes_dict = {}
    for include in includes:
        path = parse_ros_path(include)
        include_dict = load_config(path, depth=depth + 1)

        # nest the include under `key` if specified
        if "key" in include:
            include_dict = {include["key"]: include_dict}

        # update the includes dict and reassign
        includes_dict = recursive_dict_update(includes_dict, include_dict)

    # now add in the info from this file
    d = recursive_dict_update(includes_dict, d)
    return d


def parse_number(x, dtype=float):
    """Parse a number from the config.

    If the number can be converted to a float, then it is and is returned.
    Otherwise, check if it ends with "pi" and convert it to a float that is a
    multiple of pi.

    Args:
        x (str or number): Number or string to parse (e.g., "1.5", "2pi").
        dtype (type): Target data type (default: float).

    Returns:
        float or dtype: Parsed number.

    Raises:
        ValueError: If the string cannot be parsed as a number.
    """
    try:
        return dtype(x)
    except ValueError:
        if isinstance(x, str) and x.endswith("pi"):
            return float(x[:-2]) * np.pi
        else:
            raise ValueError(f"Could not parse {x} as a number.")


def parse_array_element(x):
    """Parse a single array element (float, pi-scaled, or repeated value).

    Args:
        x (str): String to parse. Can be:
            - A float (e.g., "1.5")
            - Pi-scaled (e.g., "2pi" for 2*pi)
            - Repeated (e.g., "1.0rep5" for [1.0, 1.0, 1.0, 1.0, 1.0])

    Returns:
        list: Parsed value as a list.

    Raises:
        ValueError: If the string cannot be parsed.
    """
    try:
        return [float(x)]
    except ValueError:
        if x.endswith("pi"):
            return [float(x[:-2]) * np.pi]
        if "rep" in x:
            y, n = x.split("rep")
            return float(y) * np.ones(int(n))
        raise ValueError(f"Could not convert {x} to array element.")


def parse_array(a):
    """Parse a one-dimensional iterable into a numpy array.

    Args:
        a (iterable): One-dimensional iterable of array elements (can include floats, "pi"-scaled values, or repeated values).

    Returns:
        ndarray: Parsed numpy array.
    """
    subarrays = []
    for x in a:
        subarrays.append(parse_array_element(x))
    return np.concatenate(subarrays)


def parse_path(path):
    """Parse and resolve file path, handling ROS package and environment variable references.

    Args:
        path (str): Path string that may contain:
            - $(rospack find <package>) commands
            - Environment variables (e.g., $HOME)

    Returns:
        str or None: Resolved path string, or None if rospack command fails.
    """
    # Regular expression to match the $(rospack find <package>) command
    rospack_pattern = r"\$\((rospack find \w+)\)"

    # Search for the $(rospack find <package>) command
    match = re.search(rospack_pattern, path)
    if match:
        rospack_command = match.group(
            1
        )  # Extract the rospack command (e.g., "rospack find mm_control")

        try:
            # Use subprocess to run the rospack command (e.g., rospack find mm_control)
            package_path = subprocess.check_output(
                rospack_command.split(), text=True
            ).strip()

            # Replace the $(rospack find <package>) part with the actual path
            resolved_path = re.sub(rospack_pattern, package_path, path)

            # Expand any environment variables in the resolved path
            expanded_path = os.path.expandvars(resolved_path)

            return expanded_path
        except subprocess.CalledProcessError as e:
            print(f"Error running command {rospack_command}: {e}")
            return None
    else:
        return os.path.expandvars(path)


def parse_ros_path(d, as_string=True):
    """Resolve full path from a dict containing ROS package and relative path.

    Args:
        d (dict): Dictionary with keys "package" (str) and "path" (str).
        as_string (bool): If True, return path as string; if False, return Path object.

    Returns:
        str or Path: Resolved full path.
    """
    rospack = rospkg.RosPack()
    path = Path(rospack.get_path(d["package"])) / d["path"]
    if as_string:
        path = path.as_posix()
    return path


def xacro_include(path):
    """Generate xacro include directive string.

    Args:
        path (str): Path to file to include.

    Returns:
        str: Xacro include directive XML string.
    """
    return f"""
    <xacro:include filename="{path}" />
    """


def parse_and_compile_urdf(d, max_runs=10, compare_existing=True):
    """Parse and compile a URDF from a xacro'd URDF file.

    Args:
        d (dict): Dictionary with "includes" (list) and optionally "args" (dict) keys. Also used for output path via parse_ros_path.
        max_runs (int): Maximum number of xacro processing iterations.
        compare_existing (bool): If True, compare with existing file before writing.

    Returns:
        str: Path to compiled URDF file.

    Raises:
        ValueError: If URDF file does not converge after max_runs iterations.
    """

    s = """
    <?xml version="1.0" ?>
    <robot name="thing" xmlns:xacro="http://www.ros.org/wiki/xacro">
    """.strip()
    for incl in d["includes"]:
        s += xacro_include(incl)
    s += "</robot>"

    doc = xacro.parse(s)
    s1 = doc.toxml()

    # xacro args
    mappings = d["args"] if "args" in d else {}

    # keep processing until a fixed point is reached
    run = 1
    while run < max_runs:
        xacro.process_doc(doc, mappings=mappings)
        s2 = doc.toxml()
        if s1 == s2:
            break
        s1 = s2
        run += 1

    if run == max_runs:
        raise ValueError("URDF file did not converge.")

    # write the final document to a file for later consumption
    output_path = parse_ros_path(d, as_string=False)

    # make sure path exists
    if not output_path.parent.exists():
        output_path.parent.mkdir()

    text = doc.toprettyxml(indent="  ")

    # if the full path already exists, we can check if the contents are the
    # same to avoid writing it if it hasn't changed. This avoids some race
    # conditions if the file is being compiled by multiple processes
    # concurrently.
    if output_path.exists() and compare_existing:
        with open(output_path) as f:
            text_current = f.read()
        if text_current == text:
            print("URDF files are the same - not writing.")
            return output_path.as_posix()
        else:
            print("URDF files are not the same - writing.")

    with open(output_path, "w") as f:
        f.write(text)

    return output_path.as_posix()
