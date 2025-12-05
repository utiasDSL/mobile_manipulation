# Minor modification based on original implementation by Adam Heins
# ref: https://github.com/utiasDSL/dsl__projects__tray_balance/blob/master/upright_core/src/upright_core/logging.py

from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from mm_utils.parsing import parse_path, parse_ros_path


def get_session_timestamp():
    """Get or create a shared session timestamp via ROS parameter.

    This ensures sim and control nodes save to the same timestamped folder.
    """
    try:
        import rospy

        param_name = "/logging_session_timestamp"
        if rospy.has_param(param_name):
            return rospy.get_param(param_name)
        else:
            # First node to call this creates the timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            rospy.set_param(param_name, timestamp)
            return timestamp
    except Exception:
        # Fallback if ROS not available
        return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


class DataLogger:
    """Log data for later saving and viewing."""

    def __init__(self, config, name="data"):
        """Initialize the data logger.

        Args:
            config: Configuration dictionary with logging settings
            name: Subdirectory name for this logger (e.g., 'sim' or 'control')
        """
        log_dir = config["logging"]["log_dir"]

        # If log_dir is a simple name (not an absolute path or ROS path),
        # save to mm_run/results/<log_dir>
        if not log_dir.startswith("/") and not log_dir.startswith("$"):
            results_base = parse_ros_path({"package": "mm_run", "path": "results"})
            self.base_directory = Path(results_base) / log_dir
        else:
            self.base_directory = Path(parse_path(log_dir))

        self.name = name
        self.config = config
        self.data = {}

    def add(self, key, value):
        """Add a single value named `key`."""
        if key in self.data:
            raise ValueError(f"Key {key} already in the data log.")
        self.data[key] = value

    def append(self, key, value):
        """Append a values to the list named `key`."""
        # copy to an array (also copies if value is already an array, which is
        # what we want)
        a = np.array(value)

        # append to list or start a new list if this is the first value under
        # `key`
        if key in self.data:
            if a.shape != self.data[key][-1].shape:
                raise ValueError("Data must all be the same shape.")
            self.data[key].append(a)
        else:
            self.data[key] = [a]

    def save(self):
        """Save the data and configuration to a timestamped directory.

        Directory structure:
            <base_directory>/<session_timestamp>/<name>/
                data.npz
                config.yaml
        """
        # Get shared session timestamp (coordinated across sim/control nodes)
        session_timestamp = get_session_timestamp()

        # Create directory: <base>/<session_timestamp>/<name>/
        dir_path = self.base_directory / session_timestamp / self.name
        dir_path.mkdir(parents=True, exist_ok=True)

        data_path = dir_path / "data.npz"
        config_path = dir_path / "config.yaml"

        self.data["dir_path"] = str(dir_path)

        # save the recorded data
        np.savez_compressed(data_path, **self.data)

        # save the configuration used for this run
        with open(config_path, "w") as f:
            yaml.safe_dump(self.config, stream=f, default_flow_style=False)

        print(f"Saved data to {dir_path}.")
