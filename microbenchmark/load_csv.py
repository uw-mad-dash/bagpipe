import time
import json
import mmap
import torch
import ujson
import multiprocessing as mp

# For now let us run with this, later multiprocess this loading


def load_data_from_file(in_file_queue, out_file_queue, batch_size):
    dataset_location = in_file_queue.get(block=True)
    dataset_file = open(dataset_location, "r")
    while True:
        try:
            loaded_data = list()
            time_read = time.time()
            for _ in range(batch_size):
                ln = next(dataset_file)
                loaded_data.append(ujson.loads(ln.strip()))
            # print("Time read data {}".format(time.time() - time_read))
            time_prep_batch = time.time()
            dense_x = torch.tensor([l["dense"] for l in loaded_data])
            sparse = torch.tensor([l["sparse"] for l in loaded_data]).transpose(0, 1)
            target = torch.tensor([l["label"] for l in loaded_data])
            target.unsqueeze_(1)
            # print("Time to prep batch {}".format(time.time() - time_prep_batch))
            time_start_put = time.time()
            out_file_queue.put((dense_x, sparse, target), block=True)
            # print("Time for calculating GPU {}".format(time.time() - time_start_put))
        except StopIteration:
            out_file_queue.put("end")
            dataset_location = in_file_queue.get(block=True)
            dataset_file = open(dataset_location, "r")


def load_data_from_memory(in_file_queue, out_file_queue, batch_size):
    print("In load data from memory")
    dataset_location = in_file_queue.get(block=True)
    dataset_file = open(dataset_location, "r")
    print("Openedd file")
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
            target = list()
            for l in line_extracted:
                dense_x.append(l["dense"])
                sparse.append(l["sparse"])
                target.append(l["label"])
            dense_x = torch.tensor(dense_x)
            sparse = torch.tensor(sparse).transpose(0, 1)
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


def load_data_from_file_mmap(in_file_queue, out_file_queue, batch_size):
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
                sparse = torch.tensor([l["sparse"] for l in loaded_data]).transpose(
                    0, 1
                )

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
    def __init__(self, dataset_location, batch_size, mode="file"):
        self.dataset_location = dataset_location
        self.batch_size = batch_size
        self.in_file_queue = mp.Queue(maxsize=2)
        self.out_file_queue = mp.Queue(maxsize=500)
        if mode == "memory":
            p = mp.Process(
                target=load_data_from_memory,
                args=(
                    self.in_file_queue,
                    self.out_file_queue,
                    batch_size,
                ),
            )
        if mode == "mmap":
            p = mp.Process(
                target=load_data_from_file_mmap,
                args=(
                    self.in_file_queue,
                    self.out_file_queue,
                    batch_size,
                ),
            )
        if mode == "file":
            p = mp.Process(
                target=load_data_from_file,
                args=(
                    self.in_file_queue,
                    self.out_file_queue,
                    batch_size,
                ),
            )

        p.start()

    def __iter__(self):
        # self.dataset_file = open(self.dataset_location, "r")
        self.in_file_queue.put(self.dataset_location)
        return self

    def __next__(self):
        try:
            # loaded_data = list()
            # for _ in range(self.batch_size):
            # ln = next(self.dataset_file)
            # loaded_data.append(json.loads(ln.strip()))
            # # print(loaded_data)
            # dense_x = torch.tensor([l["dense"] for l in loaded_data])
            # sparse = torch.tensor([l["sparse"] for l in loaded_data]).transpose(0, 1)
            # target = torch.tensor([l["label"] for l in loaded_data])
            # target.unsqueeze_(1)
            time_wait_queue = time.time()
            val = self.out_file_queue.get(block=True)
            # print("Time waiting {}".format(time.time() - time_wait_queue))
            if val == "end":
                self.in_file_queue.put(self.dataset_location)
                raise StopIteration
            else:
                dense_x, sparse, target = val
            # print("Queue size {}".format(self.in_file_queue.qsize()))
            return (dense_x, sparse, target)

        except StopIteration:
            # almost infinite iterator
            # self.dataset_file = open(self.dataset_location, "r")
            raise StopIteration
        # return (dense_x, sparse, target)


if __name__ == "__main__":
    dataiterator = CSVLoader("./kaggle_head_testing.txt", 20)
    dataloader = iter(dataiterator)
    epoch = 0
    while epoch < 4:
        try:
            # print("Try")
            input_batch = next(dataloader)
            print(input_batch)
        except StopIteration:
            epoch = epoch + 1
            print(epoch)
            continue
