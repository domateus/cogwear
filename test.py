import os
import pickle

path = f"{os.getcwd()}/results_30/END_TO_END/samsung_ppg/trials/lstm/lstm.pkl"
trials = {}
try:
    if os.path.exists(path):
        with open(path, "rb") as f:
            trials = pickle.load(f)
except EOFError:
    pass

len(trials)
print(trials.best_trial['result']['x'])
