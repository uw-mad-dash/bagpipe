import ujson
import numpy as np
data = open("kaggle_criteo_weekly.txt", "r")
new = open("kaggle_criteo_new.txt", "w")
for line in data:
    l = ujson.loads(line.strip())
    ujson.dump({'label': l['label'], 'sparse': np.array(l['sparse']).reshape(-1, 1).tolist(), 'dense': l['dense']}, new)
    new.write('\n')