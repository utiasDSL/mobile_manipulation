# Minor modification based on original implementation by Adam Heins
# ref: https://github.com/utiasDSL/dsl__projects__tray_balance/blob/master/upright_core/src/upright_core/logging.py

import numpy as np
from pathlib import Path
import yaml

from mmseq_utils.parsing import parse_path

class DataLogger:
    """Log data for later saving and viewing."""

    def __init__(self, config):
        self.directory = Path(parse_path(config["logging"]["log_dir"]))
        # self.timestep = config["logging"]["timestep"]
        # self.last_log_time = -np.infty

        self.config = config
        self.data = {}

    # # TODO it may bite me that this is stateful
    # def ready(self, t):
    #     if t >= self.last_log_time + self.timestep:
    #         self.last_log_time = t
    #         return True
    #     return False

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

    def save(self, timestamp, name="data"):
        """Save the data and configuration to a timestamped directory."""
        dir_name = timestamp.strftime("%Y-%m-%d_%H-%M-%S")
        dir_name = name + "_" + dir_name
        dir_path = self.directory / dir_name
        dir_path.mkdir(parents=True)

        data_path = dir_path / "data.npz"
        config_path = dir_path / "config.yaml"

        self.data["dir_path"] = str(dir_path)

        # save the recorded data
        np.savez_compressed(data_path, **self.data)

        # save the configuration used for this run
        with open(config_path, "w") as f:
            yaml.safe_dump(self.config, stream=f, default_flow_style=False)

        print(f"Saved data to {dir_path}.")










