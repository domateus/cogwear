import os
import pickle

path = f"{os.getcwd()}/results/END_TO_END/muse_eeg/trials/lstm/lstm.pkl"
trials = {}
try:
    if os.path.exists(path):
        with open(path, "rb") as f:
            trials = pickle.load(f)
except EOFError:
    pass

len(trials)
