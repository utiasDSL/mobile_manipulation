from pathlib import Path

import numpy as np
import yaml

from mm_utils.parsing import parse_path, parse_ros_path


class DataLogger:
    """Log data for later saving and viewing."""

    def __init__(self, config, name="data"):
        """Initialize the data logger.

        Args:
            config (dict): Configuration dictionary with logging settings.
            name (str): Subdirectory name for this logger (e.g., 'sim' or 'control').
        """
        if not isinstance(config, dict) or "logging" not in config:
            raise ValueError("Config must be a dict with 'logging' key")

        log_dir = config["logging"]["log_dir"]

        # If log_dir is a simple name, save to mm_run/results/<log_dir>
        if not log_dir.startswith("/") and not log_dir.startswith("$"):
            results_base = parse_ros_path({"package": "mm_run", "path": "results"})
            self.base_directory = Path(results_base) / log_dir
        else:
            self.base_directory = Path(parse_path(log_dir))

        self.name = str(name)
        self.config = config
        self.data = {}

    def add(self, key, value):
        """Add a single value for the given key.

        Args:
            key (str): Key name.
            value: Value to store.

        Raises:
            ValueError: If key already exists in the data log.
        """
        if key in self.data:
            raise ValueError(f"Key '{key}' already exists in the data log.")
        self.data[key] = value

    def append(self, key, value):
        """Append a value to the list for the given key.

        Args:
            key (str): Key name.
            value: Value to append (will be converted to numpy array).

        Raises:
            ValueError: If shape mismatch with existing values for the key.
        """
        a = np.array(value)

        if key in self.data:
            # Check shape consistency
            if a.shape != self.data[key][-1].shape:
                raise ValueError(
                    f"Shape mismatch for key '{key}': expected {self.data[key][-1].shape}, got {a.shape}"
                )
            self.data[key].append(a)
        else:
            # Start new list
            self.data[key] = [a]

    def save(self, session_timestamp):
        """Save the data and configuration to a timestamped directory.

        Directory structure:
            <base_directory>/<session_timestamp>/<name>/
                data.npz
                config.yaml

        Args:
            session_timestamp (str): Timestamp string in format "%Y-%m-%d_%H-%M-%S".
        """
        dir_path = self.base_directory / session_timestamp / self.name
        dir_path.mkdir(parents=True, exist_ok=True)

        # Save data as compressed numpy archive
        data_path = dir_path / "data.npz"
        config_path = dir_path / "config.yaml"

        self.data["dir_path"] = str(dir_path)

        # save the recorded data
        np.savez_compressed(data_path, **self.data)

        # Save configuration as YAML
        with open(config_path, "w") as f:
            yaml.safe_dump(self.config, stream=f, default_flow_style=False)

        print(f"Saved data to {dir_path}.")
