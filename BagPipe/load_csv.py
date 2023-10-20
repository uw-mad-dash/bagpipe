import time
import json
import mmap
import queue
import torch
import orjson
import numpy as np
import multiprocessing as mp

def load_data_from_file(dataset_dir, out_file_queue, legacy_queue, rank, batch_size, emb_num, batch_num, ragged):
    dataset_file = open(f"{dataset_dir}/part_{rank}.txt", "r+b")
    counter = 0
    while counter < batch_num:
        loaded_data = list()
        try:
            # acquire_line_start = time.time()
            for _ in range(batch_size):
                ln = next(dataset_file)
                loaded_data.append(orjson.loads(ln))
            # acquire_line_stop = time.time()
            # print(f"Load from Disk {(acquire_line_stop - acquire_line_start) * 1000}ms")
            
            # parse_data_start = time.time()
            out_file_queue.put(parse_data(loaded_data, emb_num, ragged))
            # parse_data_stop = time.time()
            # print(f"Parse Data {(parse_data_stop - parse_data_start) * 1000}ms")
            counter += 1
        except StopIteration:
            dataset_file = open(f"{dataset_dir}/part_{rank}.txt", "r+b")
        
        
def load_data_from_legacy(out_file_queue, legacy_queue, num_process, batch_size, emb_num, ragged):
    # loaded_data = list()
    process_end_count = 0
    while True:
        ln = legacy_queue.get()
        if ln == "end":
            process_end_count += 1
            if process_end_count == num_process:
                break
            out_file_queue.put("end")
            continue
        # loaded_data.append(ln)
        # if len(loaded_data) == batch_size:
        #     out_file_queue.put(parse_data(loaded_data, emb_num, ragged), block=True)
        #     loaded_data = list()


def parse_data(loaded_data, emb_num, ragged):
    dense_x = np.array([l["dense"] for l in loaded_data], dtype=np.float32)
    if ragged:
        sparse = list()
        for _ in range(emb_num):
            sparse.append(list())
        for l in loaded_data:
            sparse_item = l["sparse"]
            for i in range(emb_num):
                sparse[i].append(sparse_item[i])
        sparse = [np.array(s) for s in sparse]
        sparse_u = [np.unique(s, return_counts=True) for s in sparse]
    else:
        sparse = np.array([l["sparse"] for l in loaded_data]).transpose()
        sparse_u = [unique_once(l) for l in sparse]
    target = np.expand_dims(np.array([l["label"] for l in loaded_data], dtype=np.float32), axis=1)
    return (dense_x, sparse, sparse_u, target)


def unique_once(sparse):
    k, v = np.unique(sparse, return_counts=True)
    return k, k[v.__eq__(1)]


def load_data_from_memory(in_file_queue, out_file_queue, batch_size, emb_len):
    print("In load data from memory")
    dataset_location = in_file_queue.get(block=True)
    dataset_file = open(dataset_location, "r")
    print("Opened file")
    dataset_file = dataset_file.readlines()
    dataset_file = [json.loads(ln_val.strip()) for ln_val in dataset_file]
    length_ln_number = len(dataset_file)
    ln_count = 0
    print("Loaded data from memory")
    while True:
        try:
            loaded_data = list()
            end_ln_count = ln_count + batch_size
            if end_ln_count >= length_ln_number:
                raise IndexError
            line_extracted = dataset_file[ln_count:end_ln_count]
            ln_count = end_ln_count
            # loaded_data = [json.loads(ln_val.strip()) for ln_val in line_extracted]
            # for ln_val in line_extracted:
            # pass
            # ln = dataset_file.pop(0)
            # print("Line {}".format(ln))
            # ln_count += 1
            # print(ln_count)
            # loaded_data.append(json.loads(ln_val.strip()))
            dense_x = list()
            sparse = list()
            for _ in range(emb_len):
                sparse.append(list())
            target = list()
            for l in line_extracted:
                dense_x.append(l["dense"])
                sparse_item = l["sparse"]
                for i in range(emb_len):
                    sparse[i].append(sparse_item[i])
                target.append(l["label"])
            dense_x = torch.tensor(dense_x)
            target = torch.tensor(target)
            # dense_x = torch.tensor([l["dense"] for l in loaded_data])
            # sparse = torch.tensor([l["sparse"] for l in loaded_data]).transpose(0, 1)
            # target = torch.tensor([l["label"] for l in loaded_data])
            target.unsqueeze_(1)
            out_file_queue.put((dense_x, sparse, target), block=True)
        except IndexError:
            print("In error")
            out_file_queue.put("end")
            ln_count = 0
            # dataset_location = in_file_queue.get(block=True)
            # dataset_file = open(dataset_location, "r")
            # dataset_file.readlines()


def load_data_from_file_mmap(in_file_queue, out_file_queue, batch_size, emb_len):
    dataset_location = in_file_queue.get(block=True)
    dataset_file = open(dataset_location, "r+b")
    mm = mmap.mmap(dataset_file.fileno(), 0, prot=mmap.PROT_READ)
    # read_iter = iter(mm.readline, b"")
    read_lines = 0
    loaded_data = list()
    while True:
        try:
            ln = mm.readline()
            if ln == b"":
                raise StopIteration
            else:
                read_lines += 1
                ln = ln.decode("utf-8")
                loaded_data.append(json.loads(ln.strip()))

            if read_lines % batch_size == 0:
                dense_x = torch.tensor([l["dense"] for l in loaded_data])
                sparse = list()
                for _ in range(emb_len):
                    sparse.append(list())
                for l in loaded_data:
                    sparse_item = l["sparse"]
                    for i in range(emb_len):
                        sparse[i].append(sparse_item[i])
                target = torch.tensor([l["label"] for l in loaded_data])
                target.unsqueeze_(1)
                # print("Size dense x {}".format(dense_x.shape))
                out_file_queue.put((dense_x, sparse, target), block=True)
                loaded_data = list()

        except StopIteration:
            out_file_queue.put("end")
            dataset_location = in_file_queue.get(block=True)
            dataset_file = open(dataset_location, "r+b")
            mm = mmap.mmap(dataset_file.fileno(), 0, access=mmap.ACCESS_READ)
            # read_iter = iter(mm.readline, b"")


class CSVLoader(object):
    def __init__(self, dataset_dir, batch_size, emb_num, batch_num, num_process=8, ragged=False):
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.emb_num = emb_num
        self.num_process = num_process
        self.ragged = ragged
        self.out_file_queue = mp.SimpleQueue()
        self.legacy_queue = None
        self.res_queue = queue.Queue(maxsize=batch_num)
        
        chunk = batch_num // num_process
        
        for i in range(num_process - 1):
            mp.Process(target=load_data_from_file, args=(self.dataset_dir, self.out_file_queue, self.legacy_queue, i, self.batch_size, self.emb_num, chunk, self.ragged)).start()
        mp.Process(target=load_data_from_file, args=(self.dataset_dir, self.out_file_queue, self.legacy_queue, num_process - 1, self.batch_size, self.emb_num, batch_num - chunk * (num_process - 1), self.ragged)).start()
        
        while not self.res_queue.full():
            self.res_queue.put(self.out_file_queue.get())

    def __next__(self):
        if self.res_queue.empty():
            raise StopIteration
        else:
            val = self.res_queue.get()
            dense_x, sparse, sparse_u, target = val
            if dense_x.shape[1] == 0:
                dense_x = np.random.randn(dense_x.shape[0], 13)
            return (dense_x, sparse, sparse_u, target)


if __name__ == "__main__":
    dataiterator = CSVLoader("../../kaggle_16", 65536, 26, 2200, 16)
    counter = 0
    while True:
        try:
            next_start = time.time() * 1000
            input_batch = next(dataiterator)
            next_stop = time.time() * 1000
            print(f"------------------------------------Next {next_stop - next_start}ms")
            # print(input_batch[0].shape)
            counter += 1
        except StopIteration:
            print(counter)
            break