from itertools import product
from math import sqrt

import gurobipy as gp
from gurobipy import GRB

import argparse

import builtins
import datetime
import json
import sys
from time import time
import queue
import numpy

# pytorch dlrm data

import dlrm_data_pytorch as dp

import numpy as np

# import sklearn.metrics

import torch
import threading

from load_csv import CSVLoader
from collections import defaultdict


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


class Simulator(object):
    """
    Simulator time
    """

    def __init__(self, args):
        # self.ln_bot = np.fromstring(args.arch_mlp_bot, dtype=int, sep="-")
        # (
        #     self.train_data,
        #     self.train_ld,
        #     self.test_data,
        #     self.test_ld,
        # ) = dp.make_criteo_data_and_loaders(args)

        # self.table_feature_map = {
        #     idx: idx for idx in range(len(self.train_data.counts))
        # }
        self.train_ld = CSVLoader(args.processed_data_file, args.mini_batch_size, 26, 2)
        self.train_ld = iter(self.train_ld)
        self.queue = queue.Queue()
        self.latest_tracker = {
            k: torch.ones(idx_vals, dtype=torch.long)
            for k, idx_vals in enumerate(args.ln_emb)
        }
        self.lookahead = args.lookahead
        # self.nbatches = args.num_batches if args.num_batches > 0 else len(self.train_ld)
        # self.nbatches_test = len(self.test_ld)
        # self.ln_emb = np.array(self.train_data.counts)
        # self.m_den = self.train_data.m_den
        # self.ln_bot[0] = self.m_den
        self.nepochs = args.nepochs
        self.p_iter = 0
        self.t_iter = 0
        self.cache = {
            k: torch.empty(idx_vals, dtype=torch.long)
            for k, idx_vals in enumerate(args.ln_emb)
        }
        self.cache_size = 10 * 1024 * 1024 * 1024  # 100MB in bytes
        self.emb_size = 786 * 4  # 786 floats in bytes
        self.num_emb_cache = int(self.cache_size / self.emb_size)
        self.cache_hit = 0
        self.total_access = 0
        self.filled = False
        self.current = None
        self.next = None
        self.mini_batch_size = args.mini_batch_size
        self.prefetch()
        self.num_trainers = args.num_trainers
        # threading.Thread(target=self.prefetch).start()
        self.cache_state_per_worker = {
            i: {
                k: torch.empty(idx_vals, dtype=torch.long)
                for k, idx_vals in enumerate(args.ln_emb)
            }
            for i in range(num_trainers)
        }

    def get_partitioned_batch():
        """
        Get partitioned batches for lS_i
        """
        # for now we are going to get a partitioned batch for now we are doing random
        # in future we will get that from the optimization problem

    def prefetch(self):
        while self.p_iter - self.t_iter < self.lookahead:
            inputBatch = next(self.train_ld)
            X, lS_i, T = inputBatch
            get_partitioned_batch = self.get_paritioned_batch(lS_i)
            for emb_id, vals in enumerate(lS_i):
                # partition the batch
                vals = torch.unique(vals)
                self.cache[emb_id][vals] = 1
                self.latest_tracker[emb_id][vals] = self.p_iter
            self.queue.put(inputBatch, block=True)
            self.p_iter += 1

    def simulate_training(self):
        self.current = self.queue.get(block=True)
        ta = 0
        tb = 0
        while self.t_iter < self.nepochs:
            print(self.t_iter)
            self.next = self.queue.get(block=True)
            X, lS_i, T = self.current
            Xn, lS_in, Tn = self.next
            a, b = self.cache_update_new(lS_i, lS_in)
            ta += a
            tb += b
            self.t_iter += 1
            self.current = self.next
        print(self.total_access)
        print(self.cache_hit)
        print(tb / ta)

    def cache_update(self, lS_i, lS_in):
        sync_next_round_w0 = 0
        sync_later_w0 = 0
        sync_next_round_w1 = 0
        sync_later_w1 = 0
        sync_next_round_overlapping = 0
        sync_later_overlapping = 0
        evict = 0
        for emb_id, vals in enumerate(lS_i):
            w0 = vals[: self.mini_batch_size // 2]
            w1 = vals[self.mini_batch_size // 2 :]
            wn0 = lS_in[emb_id][: self.mini_batch_size // 2]
            wn1 = lS_in[emb_id][self.mini_batch_size // 2 :]
            w0 = torch.unique(w0).numpy()
            w1 = torch.unique(w1).numpy()
            wn0 = torch.unique(wn0).numpy()
            wn1 = torch.unique(wn1).numpy()
            w0i = numpy.intersect1d(w0, wn0, True)
            w1i = numpy.intersect1d(w1, wn1, True)
            w0a = numpy.setxor1d(w0, w0i)
            w1a = numpy.setxor1d(w1, w1i)
            sync_next_round_w0 += len(w0i)
            sync_later_w0 += len(w0a)
            sync_next_round_w1 += len(w1i)
            sync_later_w1 += len(w1a)
            sync_next_round_overlapping += len(numpy.intersect1d(w0i, w1i, True))
            sync_later_overlapping += len(numpy.intersect1d(w0a, w1a, True))
            ev = (self.latest_tracker[emb_id][vals] == self.t_iter).nonzero()
            evict += len(ev)
        print("Sync Next Round W0 {}".format(sync_next_round_w0))
        print("Sync Later W0 {}".format(sync_later_w0))
        print("Sync Next Round W1 {}".format(sync_next_round_w1))
        print("Sync Later W1 {}".format(sync_later_w1))
        print("Sync Next Round Overlapping {}".format(sync_next_round_overlapping))
        print("Sync Later Overlapping {}".format(sync_later_overlapping))
        print("Evict {}".format(evict))
        return None

    def cache_update_new(self, lS_i, lS_in):
        sync_next_round_w0 = 0
        sync_later_w0 = 0
        sync_next_round_w1 = 0
        sync_later_w1 = 0
        sync_next_round_overlapping = 0
        sync_later_overlapping = 0
        sync_later_count = 0
        evict = 0
        for emb_id, vals in enumerate(lS_i):
            vals = vals.numpy()
            valsn = lS_in[emb_id].numpy()
            t1 = time() * 1000
            vals_u, counts = np.unique(vals, return_counts=True)
            vlasn_u = np.unique(valsn)
            sync_now = np.intersect1d(vals_u, vlasn_u, assume_unique=True)
            sync_later = np.setxor1d(vals_u, sync_now, assume_unique=True)
            # print("Basic Logic {}ms".format(time() * 1000 - t1))
            t1 = time() * 1000
            non_overlap = vals_u[counts.__eq__(1)]
            sync_later_prune = non_overlap[np.isin(non_overlap, sync_later)]
            ev = (
                self.latest_tracker[emb_id][sync_later_prune] == self.t_iter
            ).nonzero()
            # print("New Logic {}ms".format(time() * 1000 - t1))
            evict += len(ev)
            sync_later_count += len(sync_later)
        print("Sync Later {}".format(sync_later_count))
        print("Evict & Unique in Sync Later {}".format(evict))
        return sync_later_count, evict


def unpack_batch(b):
    # Experiment with unweighted samples
    return b[0], b[1], b[2], b[3], torch.ones(b[3].size()), None


def get_emb_length(in_file):
    with open(in_file, "r") as fin:
        data = fin.readlines()

    data = [int(d) for d in data]
    return data


def parse_args(parser):

    # parser.add_argument("--arch-mlp-bot", type=dash_separated_ints, default="4-3-2")
    # parser.add_argument("--data-size", type=int, default=1)
    # parser.add_argument("--num-batches", type=int, default=4096)
    # parser.add_argument("--data-generation", type=str, default="dataset")
    # parser.add_argument("--data-set", type=str, default="kaggle")
    # parser.add_argument("--raw-data-file", type=str, default="")
    parser.add_argument("--lookahead", type=int, default=200)
    parser.add_argument("--processed-data-file", type=str, default="")
    # parser.add_argument("--data-randomize", type=str, default="total")

    parser.add_argument("--data-trace-enable-padding", type=bool, default=False)
    parser.add_argument("--max-ind-range", type=int, default=-1)
    parser.add_argument("--data-sub-sample-rate", type=float, default=0.0)  # in [0, 1]
    parser.add_argument("--num-indices-per-lookup", type=int, default=10)
    parser.add_argument("--num-indices-per-lookup-fixed", type=bool, default=False)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--test-num-workers", type=int, default=2)
    parser.add_argument("--memory-map", action="store_true", default=False)
    parser.add_argument("--nepochs", type=int, default=50)
    parser.add_argument("--mini-batch-size", type=int, default=256)
    # parser.add_argument("--test-mini-batch-size", type=int, default=10)
    parser.add_argument(
        "--dataset-multiprocessing",
        action="store_true",
        default=False,
        help="The Kaggle dataset can be multiprocessed in an environment \
                        with more than 7 CPU cores and more than 20 GB of memory. \n \
                        The Terabyte dataset can be multiprocessed in an environment \
                        with more than 24 CPU cores and at least 1 TB of memory.",
    )
    parser.add_argument("--emb-info-file", type=str, default=None)
    parser.add_argument(
        "--ln-emb",
        type=list,
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

    args = parser.parse_args()
    if args.emb_info_file is not None:
        args.ln_emb = get_emb_length(args.emb_info_file)
    return args


if __name__ == "__main__":
    args = parse_args(
        argparse.ArgumentParser(description="Argument Parser for simulation")
    )

    sim = Simulator(args)
    # while not sim.filled:
    #     pass
    sim.simulate_training()


def cost_function(current_batch, cache_state):
    """
    cache state:
    """
