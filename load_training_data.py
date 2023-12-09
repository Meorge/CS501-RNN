from json import load
from os.path import join, isfile
from os import listdir

import numpy as np
from feature_select import make_features

try:
    from rich import print
except ImportError:
    pass


def get_all_training_data(folder: str) -> list[tuple[tuple, np.ndarray]]:
    training_data_files = [
        join(folder, name)
        for name in listdir(folder)
        if isfile(join(folder, name)) and name.lower().endswith(".json")
    ]

    # Each segment will be 100 items long
    segments = []
    for filename in training_data_files:
        with open(filename) as f:
            raw_data = load(f)["data"]

        for i in range(0, 10000, 100):
            raw_data_piece = raw_data[i : i + 100]

            in_most_recent, in_rep_hist = make_features(raw_data_piece)
            out_expected = np.array([1.0 if raw_data_piece[-1]["pu_present"] else 0.0])

            segments.append(((in_most_recent, in_rep_hist), out_expected))

    print(f"{len(segments)} segments loaded")
    return segments
