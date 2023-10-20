# decides what to cache and does not stop
import os
import time
import copy
import argparse
import numpy as np
from collections import defaultdict
import torch.distributed.rpc as rpc

import torch
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp

import functools

import load_csv
import logging
from subprocess import call


def update_train_queue(input_dict):
    aa = 1
    # comp_intensive_model.train_queue.put(input_dict)
    return 1


def update_prefetch_queue(input_dict):
    aa = 1
    # comp_intensive_model.prefetch_queue.put(input_dict)
    return 1


def exit_worker(input_dict):
    rpc.shutdown()
    return 1


def dash_separated_ints(value):
    vals = value.split("-")
    for val in vals:
        try:
            int(val)
        except ValueError:
            raise argparse.ArgumentTypeError(
                "%s is not a valid dash separated list of ints" % value
            )

    return value


def unpack_batch(b):
    # Experiment with unweighted samples
    return b[0], b[1], b[2], b[3], torch.ones(b[3].size()), None


class OracleCacher(object):
    """
    Oracle Cacher
    """

    def __init__(self, args):
        """
        Args:
            lookahead_value: The number of batches of the place
            worker_addresses: Worker addresses in the
        """
        self.prefetch = args.prefetch  # should look ahead training be performed
        self.cache = args.cache  # should cache elements for training
        self.lookahead_value = args.lookahead_value
        # changing lookahead value based on increasing number of elements
        self.additive_increase_param = args.additive_increase_param
        self.multiplicative_decrease_param = args.multiplicative_decrease_param
        self.args = args
        self.worker_address = args.worker_addresses
        self.cache_size = args.cache_size
        self.elements_in_cache = 0

        self.train_ld = load_csv.CSVLoader(args.processed_csv, args.mini_batch_size)

        self.train_ld_iter = iter(self.train_ld)
        self.nepochs = args.nepochs
        self.ln_emb = args.ln_emb

        # assert (
        # args.mini_batch_size % args.world_size_trainers == 0
        # ), "Batch size should be multiple of number of trainers"
        self.batch_size_on_each_machine = int(
            args.mini_batch_size / args.world_size_trainers
        )

        self.batch_queue_dense = list()
        self.batch_queue_sparse = list()
        self.batch_queue_target = list()
        self.batch_number_queue = list()
        self.batch_emb_union = list()
        # indicates if an element is in cache
        # 1 indicates in cache
        # -1 indicates not in cache
        self.local_cache = {
            k: torch.ones(idx_vals, dtype=torch.long)
            for k, idx_vals in enumerate(self.ln_emb)
        }

        if args.num_features > 26:
            for i in range(26, args.num_features):
                self.local_cache[i] = torch.ones(10, dtype=torch.long)

        self.local_cache_ttl = {
            k: torch.ones(idx_vals, dtype=torch.long)
            for k, idx_vals in enumerate(self.ln_emb)
        }

        if args.num_features > 26:
            for i in range(26, args.num_features):
                self.local_cache_ttl[i] = torch.ones(10, dtype=torch.long)

        self.prefetch_cache = dict()

        # indicates the last time we have seen element
        self.latest_tracker = {
            k: torch.ones(idx_vals, dtype=torch.long)
            for k, idx_vals in enumerate(self.ln_emb)
        }
        if args.num_features > 26:
            for i in range(26, args.num_features):
                self.latest_tracker[i] = torch.ones(10, dtype=torch.long)
        # setting values -1 as needed
        for k in self.local_cache:
            self.local_cache[k][:] = -1

        for k in self.local_cache_ttl:
            self.local_cache_ttl[k][:] = -1

        for k in self.latest_tracker:
            self.latest_tracker[k][:] = -1
        # counters for metric collection
        self.total_access = 0
        self.total_unique_access_in_batch = 0
        self.total_prefetches = 0
        self.total_cache_hits = 0
        self.total_lease_extensions = 0
        self.total_eviction_from_cache = 0
        self.total_addition_to_cache = 0
        self.max_cache_size = 0
        self.worker_id = args.worker_id
        self.worker_name = f"{args.oracle_prefix}"
        self.num_trainers = args.world_size_trainers
        self.trainer_prefix = args.trainer_prefix
        self.world_size = args.world_size
        os.environ["MASTER_ADDR"] = args.master_ip
        os.environ["MASTER_PORT"] = args.master_port
        self.cleanup_interval = max(
            1, int(args.cleanup_proportion * self.lookahead_value)
        )

        self.temp_evicted = 0
        self.total_time_compute = 0
        # self.processed_data_queue = mp.Queue()
        return None

    def train_addition(self):
        """
        Initialize the data partitioning process.
        """
        # rpc setup

        k = 0
        batch_counter = 0
        while k < self.nepochs:

            # fetching batches from train queue
            try:
                time_spent_compute = 0
                start_time_full = time.time()
                while len(self.batch_queue_dense) < self.lookahead_value:
                    # print(len(self.batch_queue_dense))
                    batch_fetch_time = time.time()
                    inputBatch = next(self.train_ld_iter)
                    batch_fetch_time_end = time.time()
                    print(
                        "Batch time taken fetch {}".format(
                            batch_fetch_time_end - batch_fetch_time
                        )
                    )
                    # X, lS_o, lS_i, T, W, CBPP = unpack_batch(inputBatch)
                    X, lS_i, T = inputBatch
                    print(lS_i.shape)
                    lS_i_temp = torch.ones(
                        (self.args.num_features, self.args.mini_batch_size),
                        dtype=torch.long,
                    )
                    for table_id, emb_ids in enumerate(lS_i):
                        lS_i_temp[table_id][:] = emb_ids
                    lS_i = lS_i_temp
                    print(lS_i.shape)
                    self.batch_queue_dense.append(X)
                    self.batch_queue_sparse.append(lS_i)
                    self.batch_queue_target.append(T)
                    self.batch_number_queue.append(batch_counter)
                    # emb_union = set()  # holds unique embeddings in a batch
                    # can be mobed to multiprocess
                    emb_union = dict()
                    start_time_union = time.time()
                    for table_id, emb_ids in enumerate(lS_i):
                        emb_ids = torch.unique(emb_ids).numpy()
                        # for emb_id in emb_ids:
                        # emb_id = emb_id.item()
                        # emb_union.add((table_id, emb_id))
                        emb_union[table_id] = emb_ids
                        # if table_id == 0:
                        # print(emb_ids)
                        self.latest_tracker[table_id][emb_ids] = batch_counter
                    # print("Tracker 3 {}".format(self.latest_tracker[0][3]))
                    self.batch_emb_union.append((emb_union))
                    batch_counter += 1
                    end_time_union = time.time()
                    # time_spent_compute += end_time_union - start_time_union
                    # print(
                    # "Total time for emb union {}".format(
                    # end_time_union - start_time_union
                    # )
                    # )
            except StopIteration:
                k = k + 1
                continue

            current_batch_dense = self.batch_queue_dense.pop(0)
            current_batch_sparse = self.batch_queue_sparse.pop(0)
            current_batch_target = self.batch_queue_target.pop(0)
            current_batch_number = self.batch_number_queue.pop(0)
            emb_union_current_batch = self.batch_emb_union.pop(0)

            if current_batch_number > self.args.times_to_run:
                logger.info(
                    "Average overhead {}".format(
                        self.total_time_compute / float(current_batch_number)
                    )
                )
                call("pkill -9 python", shell=True)
            # self.total_access += (
            # current_batch_sparse.shape[0] * current_batch_sparse.shape[1]
            # )
            # print("Current Batch Number {}".format(current_batch_number))
            print("Cache size {}".format(self.elements_in_cache))
            prefetch_list = list()
            # decide elements to prefetch
            time_prefetch = time.time()

            for table_id in emb_union_current_batch:
                potential_embs_to_fetch = emb_union_current_batch[table_id]
                # if table_id == 0:
                # print(
                # "Potential embs to fetch {}".format(potential_embs_to_fetch)
                # )
                cache_indicator = self.local_cache[table_id]
                cache_indicator_subset = cache_indicator[potential_embs_to_fetch]
                embs_not_in_cache = cache_indicator_subset == -1
                embs_not_in_cache = embs_not_in_cache.nonzero().squeeze()
                embs_to_fetch = potential_embs_to_fetch[embs_not_in_cache]
                if type(embs_to_fetch) == np.int64:
                    embs_to_fetch = np.array([embs_to_fetch])
                if len(embs_to_fetch) > 0:
                    # print("Embs to fetch {}".format(embs_to_fetch))
                    # print("Embs fetch type {}".format(type(embs_to_fetch)))
                    self.elements_in_cache += len(embs_to_fetch)
                    embs_to_fetch = torch.from_numpy(embs_to_fetch)
                    prefetch_list.append(embs_to_fetch)
                    # updating prefetch elemsents as well
                    self.local_cache_ttl[table_id][embs_to_fetch] = current_batch_number
                else:
                    prefetch_list.append(torch.tensor([]))
            time_prefetch_end = time.time()
            time_spent_compute += time_prefetch_end - time_prefetch
            # print("Time for prefetch {}".format(time_prefetch_end - time_prefetch))
            if self.elements_in_cache > self.cache_size:
                print("Prefetch value to large reducing lookahead_value")
                print("Elements in cache {}".format(self.elements_in_cache))
                self.lookahead_value = self.lookahead_value
                # continue
            if self.elements_in_cache < self.cache_size:
                # TODO: Decide additive increase
                self.lookahead_value = self.lookahead_value

            prefetch_dict = {
                current_batch_number: prefetch_list,
                "lookahead_value": self.lookahead_value,
            }

            cache_time_start = time.time()
            elements_to_cache_from_current_batch = dict()

            ttl_to_cache = dict()

            for table_id in emb_union_current_batch:
                # potential indexes to cache
                unique_embs = emb_union_current_batch[table_id]
                # print("Unique embeddings {}".format(unique_embs))
                check_subset = self.latest_tracker[table_id][unique_embs]
                potential_embs_to_cache = (
                    check_subset <= (current_batch_number + self.lookahead_value)
                ) & (check_subset > current_batch_number)
                embs_to_cache = potential_embs_to_cache.nonzero().squeeze()
                embs_ids_to_cache = unique_embs[embs_to_cache]

                elements_to_cache_from_current_batch[table_id] = embs_ids_to_cache
                ttl_to_cache[table_id] = self.latest_tracker[table_id][
                    embs_ids_to_cache
                ]

            cache_time_end = time.time()
            time_spent_compute += cache_time_end - cache_time_start

            # print(
            # "Time for cache matching {}".format(
            # cache_time_end - cache_time_start
            # )
            # )

            # lease extension and cache elements should through the same pipeline.
            # because there is just one cache and it is essentially just updating the ttl
            # lease_extensions = list()

            ttl_idx = elements_to_cache_from_current_batch
            ttl_val = ttl_to_cache

            # update state of the cache
            start_time_local_cache = time.time()
            for table_id in elements_to_cache_from_current_batch:
                elements_to_update = elements_to_cache_from_current_batch[table_id]
                ttl_of_elements_to_cache = ttl_to_cache[table_id]
                self.local_cache[table_id][elements_to_update] = 1
                self.local_cache_ttl[table_id][
                    elements_to_update
                ] = ttl_of_elements_to_cache

            end_time_local_cache = time.time()
            time_spent_compute += end_time_local_cache - start_time_local_cache
            time_cache_update = time.time()
            for table_id in self.local_cache_ttl:
                # if (current_batch_number % self.cleanup_interval == 0) and (
                # current_batch_number > self.lookahead_value
                # ):
                ttl_for_cache = self.local_cache_ttl[table_id]
                elements_to_evict = ttl_for_cache == current_batch_number
                elements_to_evict = elements_to_evict.nonzero().squeeze()
                # print(
                # "Evicted embedding Table {} Embedding {}".format(
                # table_id, elements_to_evict
                # )
                # )
                if len(elements_to_evict.shape) == 0:
                    elements_to_evict = torch.tensor([elements_to_evict.item()])
                # print(f"elements to evict {elements_to_evict}")
                # print(f"elements_to_evict shape {elements_to_evict.shape}")
                # print(f"elements_to_evict shape len {len(elements_to_evict.shape)}")
                self.temp_evicted += len(elements_to_evict)

                if (
                    current_batch_number % self.cleanup_interval == 0
                    and current_batch_number != 0
                ):
                    self.elements_in_cache = self.elements_in_cache - self.temp_evicted
                    self.temp_evicted = 0

                self.local_cache[table_id][elements_to_evict] = -1
                self.local_cache_ttl[table_id][elements_to_evict] = -1
            time_spent_compute += time.time() - time_cache_update
            self.total_time_compute += time_spent_compute
            end_time_total = time.time()
            print("Time spent compute {}".format(time_spent_compute))
            print("Total time {}".format(end_time_total - start_time_full))


def parse_args(parser):
    parser.add_argument(
        "--prefetch",
        action="store_true",
        default=False,
        help="Enable Prefetch for training",
    )

    parser.add_argument(
        "--cache",
        action="store_true",
        default=False,
        help="Enable Logic for caching",
    )
    parser.add_argument(
        "--lookahead-value",
        type=int,
        default=200,
        help="The number of batches further to look ahead for getting cache",
    )

    parser.add_argument(
        "--worker-addresses", type=str, help="Worker IP addresses to perform training"
    )

    parser.add_argument(
        "--ln-emb",
        type=list,
        # type=utils.dash_separated_ints,
        help="embedding table sizes in the right order",
        default=[
            1460,
            583,
            10131227,
            2202608,
            305,
            24,
            12517,
            633,
            3,
            93145,
            5683,
            8351593,
            3194,
            27,
            14992,
            5461306,
            10,
            5652,
            2173,
            4,
            7046547,
            18,
            15,
            286181,
            105,
            142572,
        ],
    )

    # parser.add_argument("")
    parser.add_argument("--arch-mlp-bot", type=dash_separated_ints, default="4-3-2")
    parser.add_argument("--data-size", type=int, default=1)
    parser.add_argument("--num-batches", type=int, default=0)
    parser.add_argument("--data-generation", type=str, default="dataset")
    parser.add_argument("--data-set", type=str, default="kaggle")
    parser.add_argument("--raw-data-file", type=str, default="")
    parser.add_argument("--processed-data-file", type=str, default="")
    parser.add_argument("--data-randomize", type=str, default="total")

    parser.add_argument("--data-trace-enable-padding", type=bool, default=False)
    parser.add_argument("--max-ind-range", type=int, default=-1)
    parser.add_argument("--data-sub-sample-rate", type=float, default=0.0)  # in [0, 1]
    parser.add_argument("--num-indices-per-lookup", type=int, default=10)
    parser.add_argument("--num-indices-per-lookup-fixed", type=bool, default=False)
    parser.add_argument("--memory-map", action="store_true", default=False)
    parser.add_argument("--nepochs", type=int, default=10000)
    parser.add_argument("--mini-batch-size", type=int, default=1)
    parser.add_argument("--test-mini-batch-size", type=int, default=10)
    parser.add_argument(
        "--dataset-multiprocessing",
        action="store_true",
        default=False,
        help="The Kaggle dataset can be multiprocessed in an environment \
                        with more than 7 CPU cores and more than 20 GB of memory. \n \
                        The Terabyte dataset can be multiprocessed in an environment \
                        with more than 24 CPU cores and at least 1 TB of memory.",
    )

    parser.add_argument(
        "--worker-id",
        type=int,
        required=True,
        help="Global Worker ID used for RPC init",
    )

    parser.add_argument("--cache-size", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--master-ip", type=str, default="localhost")
    parser.add_argument("--master-port", type=str, default="18000")

    parser.add_argument(
        "--oracle-prefix",
        type=str,
        default="oracle",
        help="Prefix to name oracle cacher",
    )
    parser.add_argument(
        # "--num-trainers",
        "--world-size-trainers",
        type=int,
        required=True,
        help="Number of trainers",
    )
    parser.add_argument(
        "--trainer-prefix",
        type=str,
        default="worker",
        help="prefix to call the trainer",
    )

    parser.add_argument(
        "--processed-csv", type=str, required=True, help="CSV file name"
    )
    parser.add_argument(
        "--multiplicative-decrease-param",
        type=int,
        default=2,
        help="Multiplicative decrease param when no cache space is available",
    )

    parser.add_argument(
        "--additive-increase-param",
        default=1,
        help="Additive increase param to increase training",
    )

    parser.add_argument(
        "--cleanup-proportion",
        type=float,
        default=0.25,
        help="Cleanup proportion to use",
    )

    parser.add_argument("--num-features", type=int, default=13)

    parser.add_argument("--times-to-run", type=int, default=10)

    parser.add_argument("--logging-prefix", type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args(argparse.ArgumentParser(description="Oracle cacher arguments"))
    log_file_name = f"csize_{args.cache_size}_nfeature_{args.num_features}_batchs_size{args.mini_batch_size}_lookahead_val_{args.lookahead_value}.log"
    logging.basicConfig(filename=log_file_name)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    cacher_instance = OracleCacher(args)
    cacher_instance.train_addition()
