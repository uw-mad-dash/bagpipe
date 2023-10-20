import ujson
import torch
import load_csv
import numpy as np


data_set_file = "/mnt/data/new_data_magnetes/avazu/avazu_data"
data_set_name = "avazu"
ln_emb = [  # TODO: Length of embedding file
    8,
    8,
    4738,
    7746,
    27,
    8553,
    560,
    37,
    2686409,
    6729487,
    8252,
    6,
    5,
    2627,
    9,
    10,
    436,
    5,
    69,
    173,
    61,
]


def data_parser(data_set_file):
    with open(data_set_file, "r") as fin:
        in_line = fin.readline()
        while in_line:
            data = ujson.loads(in_line.strip())
            dense_x = np.array(data["dense"])
            sparse = np.array(data["sparse"])
            target = np.array(data["label"])
            if target > 3:
                target = 1.0
            else:
                target = 0
            if dense_x.shape[0] == 0:
                dense_x = np.random.rand(13)
            temp_data = (dense_x, sparse, target)
            in_line = fin.readline()
            yield temp_data
        return


train_data = data_parser(data_set_file)
# dataloader = iter(dataiterator)


skew_table = []
total_access = np.zeros(len(ln_emb), dtype=int)
for i in range(len(ln_emb)):
    temp_list = np.zeros((ln_emb[i], 1), dtype=int)
    skew_table.append(temp_list)

# =================== Filling Skew Table ======================
for i, (X, lS_i, T) in enumerate(train_data):
    lS_i = lS_i.squeeze()
    if i % 1000000 == 0:
        print("data loading {}".format(i))
    for j, lS_i_index in enumerate(lS_i):
        total_access[j] = total_access[j] + 1
        skew_table[j][int(lS_i_index)][0] = skew_table[j][int(lS_i_index)][0] + 1

train_data = data_parser(data_set_file)

hot_emb = []  # Hot embedding list
hot_emb_dict = []
for i in range(len(ln_emb)):
    dict = {}
    for j in range(ln_emb[i]):
        if (skew_table[i][j][0] * 100) / total_access[i] > 0.00001:
            E = dlrm.emb_l[i]  # Getting required embedding table in E
            index = torch.tensor([j])  # Converting lS_i into tensor
            offset = torch.tensor([0])  # Converting lS_o into tensor
            V = E(index, offset)  # Getting Required row from E
            V = V.detach().numpy()  # Converting Embedding row into a numpy array
            V = np.insert(V, 0, i)  # Appending Row Index
            V = np.insert(V, 0, j)  # Appending Embedding Table Number
            hot_emb.append(V)
            dict[(i, j)] = len(hot_emb) - 1

    hot_emb_dict.append(dict)

skew_table = None
del skew_table
total_access = None
del total_access

train_hot = []
train_normal = []

for i, train_tuple in enumerate(train_data):
    lS_i = []
    ls_i_tuple = train_tuple[1]
    ls_i_tuple = ls_i_tuple.squeeze()
    if i % 1000000 == 0:
        print("second for loop {}".format(i))
    for j, lS_i_index in enumerate(ls_i_tuple):
        if (j, int(lS_i_index)) in hot_emb_dict[j].keys():
            lS_i.append(hot_emb_dict[j][(j, int(lS_i_index))])
        else:
            break

    if len(lS_i) == len(train_tuple[1]):
        lS_i = np.array(lS_i).astype(np.float32)
        train_hot.append((train_tuple[0], lS_i, train_tuple[2]))
    else:
        train_normal.append(train_tuple)

train_hot = np.array(train_hot).astype(np.object)
train_normal = np.array(train_normal).astype(np.object)
hot_emb_dict = np.array(hot_emb_dict).astype(np.object)
np.savez_compressed(f"./{data_set_name}_hot.npz", train_hot)
np.savez_compressed(f"./{data_set_name}_normal.npz", train_normal)
np.savez_compressed(f"./{data_set_name}_hot_emb_dict.npz", hot_emb_dict)
