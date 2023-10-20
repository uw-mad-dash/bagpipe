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


class DistTrainModel(nn.Module):
    def __init__(
            self,
            emb_size=1,
            ln=None,
            device="cuda:0",
            bp=None
    ):
        super(DistTrainModel, self).__init__()
        self.emb_size = emb_size
        self.device = device
        self.bp = bp

        self.deep_mlp = self.create_deep_mlp(ln)
        print(self.deep_mlp)
        self.wide_linear = self.create_wide(ln, False)
        print(self.wide_linear)
        self.deep_mlp.to(self.device)
        self.wide_linear.to(self.device)
        self.loss_fn = torch.nn.BCELoss(reduction="mean")
        self.loss_fn.to(self.device)

        return None

    def create_deep_mlp(self, ln):
        layers = nn.ModuleList()
        for i in range(0, len(ln) - 1):
            n = ln[i]
            m = ln[i + 1]
            LL = nn.Linear(int(n), int(m), bias=True)
            # some xavier stuff the original pytorch code was doing

            # TODO: Change the parameter initializaion?
            mean = 0.0
            std_dev = np.sqrt(2 / (m + n))
            W = np.random.normal(mean, std_dev, size=(m, n)).astype(np.float32)
            std_dev = np.sqrt(1 / m)
            bt = np.random.normal(mean, std_dev, size=m).astype(np.float32)
            LL.weight.data = torch.tensor(W, requires_grad=True)
            LL.bias.data = torch.tensor(bt, requires_grad=True)

            layers.append(LL)

            # add ReLU for all linear layer except the last one
            if i != len(ln) - 2:
                layers.append(nn.ReLU())
        return torch.nn.Sequential(*layers)

    def create_wide(self, ln, cross_transform = False):
        if cross_transform == False:
            n = ln[-1] + 26 * self.emb_size
            m = 1
            LL = nn.Linear(int(n), int(m), bias = True)
            # some xavier stuff the original pytorch code was doing

            # TODO: Change the parameter initializaion?
            mean = 0.0
            std_dev = np.sqrt(2 / (m + n))
            W = np.random.normal(mean, std_dev, size = (m, n)).astype(np.float32)
            std_dev = np.sqrt(1 / m)
            bt = np.random.normal(mean, std_dev, size = m).astype(np.float32)
            LL.weight.data = torch.tensor(W, requires_grad = True)
            LL.bias.data = torch.tensor(bt, requires_grad = True)
            return LL
        return None

    def apply_mlp(self, dense_x, mlp_network):
        """
        Apply MLP on the features
        """
        return mlp_network(dense_x)

    def forward(self, dense_x, lS_i, target, get_emb_start, get_emb_stop):
        """
        Forward pass of the training
        """
        get_emb_start.record()
        ly = self.bp.get_emb(lS_i)
        get_emb_stop.record()
        x = self.apply_mlp(dense_x, self.deep_mlp)
        
        flat_tensor = torch.cat(ly, dim=1).transpose(0, 1).reshape((-1, 26 * self.emb_size))
        
        comb = torch.cat([flat_tensor, x], dim=1)
        
        y = self.wide_linear(comb)

        sigmoid = torch.nn.Sigmoid()
        p = sigmoid(y)
        loss = self.loss_fn(p, target)
        return loss


def main(args):
    ln = [13, 256, 256, 256]

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

    comp_intensive_model = DistTrainModel(
        emb_size=args.emb_size,
        ln=ln,
        device=args.device,
        bp=bp
    )

    dist.init_process_group(
        backend=args.dist_backend,
        init_method=f"tcp://{args.dist_master_ip}:{args.master_port_ccom}",
        world_size=args.world_size_trainers,
        rank=args.dist_worker_id,
    )

    comp_intensive_model.wide_linear = DDP(comp_intensive_model.wide_linear, device_ids=[0])
    comp_intensive_model.deep_mlp = DDP(comp_intensive_model.deep_mlp, device_ids=[0])

    optimizer = optim.SGD(
        [
            {
                "params": comp_intensive_model.wide_linear.parameters(),
                "lr": 0.01,
            },
            {
                "params": comp_intensive_model.deep_mlp.parameters(),
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
        loss = comp_intensive_model.forward(
            train_example["dense_x"].to(comp_intensive_model.device),
            train_example["sparse_vector"].to(comp_intensive_model.device),
            train_example["target"].to(comp_intensive_model.device),
            get_emb_start,
            get_emb_stop
        )
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
    parser.add_argument("--dense-size", type=int, default=13, help="The number of dense parameter")
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
        # "--dist-world-size",
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
        description="Arguments for W&D"))
    now = datetime.now().strftime("%H:%M_%B_%d_%Y")
    filename=f"training_worker_{args.dist_worker_id}_{now}_{args.logging_prefix}_bagpipe.log"
    logging.basicConfig(
        filename=filename
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.info(args)
    main(args)
