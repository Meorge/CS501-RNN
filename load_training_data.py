from json import load
from os.path import join, isfile
from os import listdir

import numpy as np
from feature_select import make_features

try:
    from rich import print
except ImportError:
    pass

def get_all_training_data() -> list[tuple[tuple, np.ndarray]]:
    TRAINING_DATA_FOLDER = "training_data_v2"

    training_data_files = [
        join(TRAINING_DATA_FOLDER, name)
        for name in listdir(TRAINING_DATA_FOLDER)
        if isfile(join(TRAINING_DATA_FOLDER, name))
        and name.lower().endswith(".json")
    ]


    # Each segment will be 100 items long
    segments = []
    for filename in training_data_files:
        with open(filename) as f:
            raw_data = load(f)["data"]
            
        for i in range(0, 10000, 100):
            raw_data_piece = raw_data[i:i+100]
            
            in_most_recent, in_rep_hist = make_features(raw_data_piece)
            out_expected = np.array([1.0 if raw_data_piece[-1]["pu_present"] else 0.0])
            
            segments.append(((in_most_recent, in_rep_hist), out_expected))
            
    return segments
        
