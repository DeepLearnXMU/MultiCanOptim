import pickle
import sys

with open(sys.argv[1]) as f:
    a=f.readlines()
results = list(map(float, a))
with open(sys.argv[2], "wb") as f:
    pickle.dump(results, f)
