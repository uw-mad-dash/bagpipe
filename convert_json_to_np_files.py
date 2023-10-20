import time
import json
import ujson
import numpy as np


def load_data_from_file(in_file_name, data_set_name):
    train_val = []
    count = 0

    with open(in_file_name, "r") as fin:
        in_line = fin.readline()
        while in_line:
            print(in_line)
            print(in_line.strip())
            data = ujson.loads(in_line.strip())
            dense_x = np.array(data["dense"]).astype(np.float32)
            sparse = np.array(data["sparse"]).astype(np.int32)
            target = data["label"]
            if target > 3:
                target = 1.0
            else:
                target = 0
            if dense_x.shape[0] == 0:
                dense_x = np.random.rand(13)
            train_val.append((dense_x, sparse, target))
            in_line = fin.readline()
    train_val = np.array(train_val).astype(np.object)
    np.savez(f"./{data_set_name}.npz", train_val)


if __name__ == "__main__":
    load_data_from_file(
        "/mnt/data/new_data_magnetes/movielen/movielen_data", "movielen_train"
    )
