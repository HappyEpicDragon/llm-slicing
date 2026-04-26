import os

import numpy as np


class AssocDataAnalyzer:
    def __init__(self, npz_data):
        self.npz_data = npz_data

    def get_keys(self):
        for key in self.npz_data.keys():
            print(key)

    def get_values(self):
        pass


def analyze_assoc_data(cfg):
    base_dir = cfg["base_dir"]
    eps_list = cfg["eps_list"]
    for ep in eps_list:
        data_path = os.path.join(base_dir, f"ep_{ep}.npz")
        data = np.load(data_path, allow_pickle=True)
        analyzer = AssocDataAnalyzer(data)
        analyzer.get_keys()


def deep_update(original, updates):
    """Recursively update a nested dict."""
    for key, value in updates.items():
        if key in original and isinstance(original[key], dict) and isinstance(value, dict):
            deep_update(original[key], value)
        else:
            original[key] = value
    return original
