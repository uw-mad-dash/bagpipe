# I think this is it. All looks good for now
# training worker, accepts input from the oracle cacher
import os
import sys
import logging
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from time import time
from datetime import datetime

import utils

from s3_utils import uploadFile
from subprocess import call

sys.path.append("..")
sys.path.append(".")

from BagPipe import init_cache


class DeepFactorizationMachineModel(torch.nn.Module):
    """
    A pytorch implementation of DeepFM.
    Reference:
        H Guo, et al. DeepFM: A Factorization-Machine based Neural Network for CTR Prediction, 2017.
    """

    def __init__(self, ln_emb, emb_size, mlp_dims, dropout, device, bp):
        super().__init__()
        self.device = device
        self.bp = bp
        self.linear = FeaturesLinear(ln_emb).to(self.device)
        self.fm = FactorizationMachine(reduce_sum=True).to(self.device)
        self.embed_output_dim = len(ln_emb) * emb_size
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout).to(self.device)
        self.activation = nn.Sigmoid().to(self.device)
        self.loss_fn = torch.nn.BCELoss(reduction="mean").to(self.device)

    def forward(self, X, y, get_emb_start, get_emb_stop):
        """
        :param X: Long tensor of size ``(batch_size, num_fields)``
        """
        get_emb_start.record()
        ly = self.bp.get_emb(X)
        get_emb_stop.record()
        embed_x = torch.stack(ly).transpose(0, 1)
        X = X.transpose(0, 1)
        X = self.linear(X) + self.fm(embed_x) + self.mlp(embed_x.flatten(start_dim=1))
        return self.loss_fn(self.activation(X), y)


class FeaturesLinear(torch.nn.Module):

    def __init__(self, field_dims, output_dim=1):
        super().__init__()
        self.fc = torch.nn.Embedding(sum(field_dims), output_dim)
        self.bias = torch.nn.Parameter(torch.zeros((output_dim,)))
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.int64)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return torch.sum(self.fc(x), dim=1) + self.bias


class FactorizationMachine(torch.nn.Module):

    def __init__(self, reduce_sum=True):
        super().__init__()
        self.reduce_sum = reduce_sum

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        square_of_sum = torch.sum(x, dim=1) ** 2
        sum_of_square = torch.sum(x ** 2, dim=1)
        ix = square_of_sum - sum_of_square
        if self.reduce_sum:
            ix = torch.sum(ix, dim=1, keepdim=True)
        return 0.5 * ix


class MultiLayerPerceptron(torch.nn.Module):

    def __init__(self, input_dim, embed_dims, dropout, output_layer=True):
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, embed_dim)``
        """
        return self.mlp(x)


def main(args):
    os.environ["MASTER_ADDR"] = args.master_ip
    os.environ["MASTER_PORT"] = args.master_port

    bp = init_cache(lookahead_value=args.lookahead_value, 
                    emb_size=args.emb_size, 
                    ln_emb=args.ln_emb, 
                    cache_size=args.cache_size, 
                    device=args.device,
                    trainer_world_size=args.world_size_trainers,
                    cleanup_batch_proportion=args.cleanup_batch_proportion, worker_id=args.worker_id,
                    world_size=args.world_size,
                    training_worker_id=args.dist_worker_id,
                    emb_optim=optim.SGD, emb_optim_params={"lr": 0.01},
                    logger=logger)
    
    mlp_dims = [64, 64, 64]
    dropout = 0.5
    
    dfm = DeepFactorizationMachineModel(args.ln_emb, args.emb_size, mlp_dims, dropout, args.device, bp)

    dist.init_process_group(
        backend=args.dist_backend,
        init_method=f"tcp://{args.dist_master_ip}:{args.master_port_ccom}",
        world_size=args.world_size_trainers,
        rank=args.dist_worker_id,
    )

    dfm.linear = DDP(dfm.linear, device_ids=[0])
    dfm.mlp = DDP(dfm.mlp, device_ids=[0])

    optimizer = optim.SGD(
        [
            {
                "params": dfm.linear.parameters(),
                "lr": 0.01,
            },
            {
                "params": dfm.fm.parameters(),
                "lr": 0.01,
            },
            {
                "params": dfm.mlp.parameters(),
                "lr": 0.01,
            },
        ]
    )

    # train loop

    # this will fill the prefetch cache

    # this is the handle we use for storing all reduce future
    
    forward_start = torch.cuda.Event(enable_timing=True)
    forward_stop = torch.cuda.Event(enable_timing=True)
    backward_start = torch.cuda.Event(enable_timing=True)
    backward_stop = torch.cuda.Event(enable_timing=True)
    dense_optim_start = torch.cuda.Event(enable_timing=True)
    dense_optim_stop = torch.cuda.Event(enable_timing=True)
    next_batch_start = torch.cuda.Event(enable_timing=True)
    next_batch_stop = torch.cuda.Event(enable_timing=True)
    get_emb_start = torch.cuda.Event(enable_timing=True)
    get_emb_stop = torch.cuda.Event(enable_timing=True)
    next_batch_counters = [torch.cuda.Event(enable_timing=True) for _ in range(6)]
    total_forward = 0
    total_get_emb = 0
    total_backward = 0
    total_dense_optim = 0
    total_next_batch = 0
    total_cache_sync_time = 0
    total_cache_waiting_time = 0
    total_sparse_optim_time = 0
    total_start = time() * 1000

    for i in range(args.stop_iter):
        # train example is what is current example
        next_batch_start.record()
        train_example = bp.next_batch(next_batch_counters)
        next_batch_stop.record()

        print(f"Current Iter {i}")
        logger.info(f"Current Iter {i}")
        
        forward_start.record()
        loss = dfm.forward(train_example["sparse_vector"].to(args.device), 
                           train_example["target"].to(args.device), 
                           get_emb_start, 
                           get_emb_stop)
        forward_stop.record()
        
        backward_start.record()
        loss.backward()
        backward_stop.record()

        logger.info("Loss {}".format(loss.item()))
        
        dense_optim_start.record()
        optimizer.step()
        optimizer.zero_grad()
        dense_optim_stop.record()
        
        torch.cuda.synchronize()
        
        logger.info("Next Batch Time {}ms".format(next_batch_start.elapsed_time(next_batch_stop)))
        logger.info("Get Emb Time {}ms".format(get_emb_start.elapsed_time(get_emb_stop)))
        logger.info("Forward Time {}ms".format(forward_start.elapsed_time(forward_stop)))
        logger.info("Backward Time {}ms".format(backward_start.elapsed_time(backward_stop)))
        logger.info("Dense Optim Time {}ms".format(dense_optim_start.elapsed_time(dense_optim_stop)))
        if i != 0:
            logger.info("Sync Now Time {}ms".format(next_batch_counters[0].elapsed_time(next_batch_counters[1])))
            if i != 1 and args.lookahead_value != 1:
                logger.info("Sync Overlap Time {}ms".format(next_batch_counters[2].elapsed_time(next_batch_counters[3])))
            logger.info("Sparse Optim Time {}ms".format(next_batch_counters[4].elapsed_time(next_batch_counters[5])))
        total_forward += forward_start.elapsed_time(forward_stop)
        total_get_emb += get_emb_start.elapsed_time(get_emb_stop)
        total_backward += backward_start.elapsed_time(backward_stop)
        total_dense_optim += dense_optim_start.elapsed_time(dense_optim_stop)
        total_next_batch += next_batch_start.elapsed_time(next_batch_stop)
        if i != 0:
            total_cache_sync_time += next_batch_counters[0].elapsed_time(next_batch_counters[1])
            if i != 1 and args.lookahead_value != 1:
                total_cache_waiting_time += next_batch_counters[2].elapsed_time(next_batch_counters[3])
            total_sparse_optim_time += next_batch_counters[4].elapsed_time(next_batch_counters[5])

    total_stop = time() * 1000
    logger.info("Average Next Batch time: {}ms".format(total_next_batch / args.stop_iter))
    logger.info("Average Forward time: {}ms".format(total_forward / args.stop_iter))
    logger.info("Average Get Emb time: {}ms".format(total_get_emb / args.stop_iter))
    logger.info("Average Backward time: {}ms".format(total_backward / args.stop_iter))
    logger.info("Average Dense Optim time: {}ms".format(total_dense_optim / args.stop_iter))
    logger.info("Average Cache Sync Critial Path time: {}ms".format((total_cache_sync_time - total_cache_waiting_time) / (args.stop_iter - 1)))
    logger.info("Average Cache Waiting time: {}ms".format((total_cache_waiting_time) / (args.stop_iter - 1)))
    logger.info("Average Sparse Optim time: {}ms".format(total_sparse_optim_time / (args.stop_iter - 1)))
    logger.info("Total Time of {} iterations: {}ms".format(args.stop_iter, total_stop - total_start))
    
    if args.s3:
        s3_resource = uploadFile("recommendation-data-bagpipe")
        if args.lookahead_value != 1:
            s3_resource.push_file(filename, f"{args.world_size_trainers}_trainers_wrap_up/{filename}")
        else:
            s3_resource.push_file(filename, f"{args.world_size_trainers}_trainers_baseline/{filename}")
    
    # dist.destroy_process_group()
    # bp.exit_worker()
    call("pkill -9 python", shell=True)


def get_emb_length(in_file):
    with open(in_file, "r") as fin:
        data = fin.readlines()

    data = [int(d) for d in data]
    return data


def parse_args(parser):
    parser.add_argument(
        "--emb-size",
        type=int,
        default=48,
        help="size of the embedding for each sparse feature",
    )

    parser.add_argument(
        "--lookahead-value",
        type=int,
        default=200,
        help="The number of batches further to look ahead for getting cache",
    )

    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--ln-emb",
        type=utils.dash_separated_ints,
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
    parser.add_argument("--cache-size", type=int, required=True)
    parser.add_argument(
        "--worker-id",
        type=int,
        required=True,
        help="Global worker ID, i.e., rank for RPC init",
    )
    parser.add_argument(
        "--world-size", type=int, required=True, help="Global world size"
    )
    parser.add_argument("--master-ip", type=str, default="localhost")
    parser.add_argument("--master-port", type=str, default="18000")
    parser.add_argument(
        "--master-port-ccom",
        type=int,
        default=9988,
        help="Port for collective communication",
    )
    parser.add_argument("--dist-backend", type=str, default="gloo")

    parser.add_argument(
        "--dist-master-ip", type=str, required=True, help="IP of rank 0 training worker"
    )
    parser.add_argument(
        "--dist-worker-id",
        type=int,
        required=True,
        help="Distributed Worker ID, for collective collection library",
    )
    parser.add_argument(
        "--world-size-trainers",
        type=int,
        required=True,
        help="Distributed World Size for collective communication library",
    )

    parser.add_argument(
        "--logging-prefix", type=str, default="test", help="Add for logging"
    )

    parser.add_argument(
        "--cleanup-batch-proportion",
        type=float,
        default=0.25,
        help="The proportion of lookahead value at which we evict and fetch",
    )
    parser.add_argument("--stop-iter", type=int,
                        default=1000, help="Add for logging")
    parser.add_argument("--emb-info-file", type=str, default=None)
    parser.add_argument("--s3", action="store_true", default=False)
    args = parser.parse_args()

    if args.emb_info_file is not None:
        args.ln_emb = get_emb_length(args.emb_info_file)

    return args


if __name__ == "__main__":
    args = parse_args(argparse.ArgumentParser(
        description="Arguments for DeepFM"))
    now = datetime.now().strftime("%H:%M_%B_%d_%Y")
    filename=f"training_worker_{args.dist_worker_id}_{now}_{args.logging_prefix}_bagpipe.log"
    logging.basicConfig(
        filename=filename
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.info(args)
    main(args)
