import os
import pickle

path = f"{os.getcwd()}/results_30/END_TO_END/empatica_eda/trials/resnet/resnet.pkl"
trials = {}
try:
    if os.path.exists(path):
        with open(path, "rb") as f:
            trials = pickle.load(f)
except EOFError:
    pass

len(trials)
print(trials.best_trial['result']['x'])
