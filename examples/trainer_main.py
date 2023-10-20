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
            ln_top=None,
            ln_bot=None,
            sigmoid_bot=-1,
            sigmoid_top=-1,
            feature_interaction="dot",
            interact_itself=False,
            loss_function="bce",
            device="cuda:0",
            bp=None
    ):
        super(DistTrainModel, self).__init__()
        """
        Args:
            emb_size: Size of for each sparse embedding
            ln_top (np.array): Structure of top MLP
            ln_bot (np.array): Structure of bottom MLP
            sigmoid_bot (int): Integer for listing the location of bottom
            sigmoid_top (int): Integer for listing the location of the top
        Returns:
            None
        """
        self.ln_top = ln_top
        self.ln_bot = ln_bot
        self.sigmoid_bot = sigmoid_bot
        self.sigmoid_top = sigmoid_top
        self.feature_interaction = feature_interaction
        self.interact_itself = interact_itself
        self.device = device
        self.bp = bp

        self.bot_mlp = self.create_mlp(self.ln_bot, self.sigmoid_bot)
        self.top_mlp = self.create_mlp(self.ln_top, self.sigmoid_top)
        self.top_mlp.to(self.device)
        self.bot_mlp.to(self.device)
        if loss_function == "bce":
            self.loss_fn = torch.nn.BCELoss(reduction="mean")
        elif loss_function == "mse":
            self.loss_fn = torch.nn.MSELoss(reduction="mean")
        self.loss_fn.to(self.device)

        return None

    def create_mlp(self, ln, sigmoid_layer):
        layers = nn.ModuleList()
        for i in range(0, ln.size - 1):
            n = ln[i]
            m = ln[i + 1]
            LL = nn.Linear(int(n), int(m), bias=True)
            # some xavier stuff the original pytorch code was doing
            mean = 0.0
            std_dev = np.sqrt(2 / (m + n))
            W = np.random.normal(mean, std_dev, size=(m, n)).astype(np.float32)
            std_dev = np.sqrt(1 / m)
            bt = np.random.normal(mean, std_dev, size=m).astype(np.float32)
            LL.weight.data = torch.tensor(W, requires_grad=True)
            LL.bias.data = torch.tensor(bt, requires_grad=True)
            layers.append(LL)
            if i == sigmoid_layer:
                layers.append(nn.Sigmoid())
            else:
                layers.append(nn.ReLU())
        return torch.nn.Sequential(*layers)

    def apply_mlp(self, dense_x, mlp_network):
        """
        Apply MLP on the features
        """
        # print("Shape Dense x {}".format(dense_x.shape))
        return mlp_network(dense_x)

    def interact_features(self, x, ly):
        """
        Interaction between dense and embeddings
        """
        # Copied from interact features function of original code
        if self.feature_interaction == "dot":
            (batch_size, d) = x.shape
            T = torch.cat([x] + ly, dim=1).view((batch_size, -1, d))
            Z = torch.bmm(T, torch.transpose(T, 1, 2))
            _, ni, nj = Z.shape
            offset = 1 if self.interact_itself else 0
            li = torch.tensor([i for i in range(ni)
                              for j in range(i + offset)])
            lj = torch.tensor([j for i in range(nj)
                              for j in range(i + offset)])
            Zflat = Z[:, li, lj]
            R = torch.cat([x] + [Zflat], dim=1)
        elif self.feature_interaction == "cat":
            R = torch.cat([x] + ly, dim=1)
        else:
            sys.exit("Unsupported feature interaction")
        return R

    def forward(self, dense_x, lS_i, target, get_emb_start, get_emb_stop):
        """
        Forward pass of the training
        """
        x = self.apply_mlp(dense_x, self.bot_mlp)
        get_emb_start.record()
        ly = self.bp.get_emb(lS_i)
        get_emb_stop.record()
        z = self.interact_features(x, ly)
        p = self.apply_mlp(z, self.top_mlp)
        loss = self.loss_fn(p, target)
        return loss


def main(args):
    ln_bot = np.fromstring(args.arch_mlp_bot, dtype=int, sep="-")
    arch_mlp_top_adjusted = (
        str(
            utils.get_first_layer_size_top_mlp(
                args.arch_interaction_op,
                args.arch_interaction_itself,
                ln_bot,
                args.ln_emb,
            )
        )
        + "-"
        + args.arch_mlp_top
    )
    ln_top = np.fromstring(arch_mlp_top_adjusted, dtype=int, sep="-")
    print("LN TOP {}".format(ln_top))

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
        ln_bot=ln_bot,
        ln_top=ln_top,
        sigmoid_bot=-1,
        sigmoid_top=ln_top.size - 2,
        loss_function=args.loss_function,
        feature_interaction=args.arch_interaction_op,
        device=args.device,
        bp=bp
    )

    dist.init_process_group(
        backend=args.dist_backend,
        init_method=f"tcp://{args.dist_master_ip}:{args.master_port_ccom}",
        world_size=args.world_size_trainers,
        rank=args.dist_worker_id,
    )

    comp_intensive_model.top_mlp = DDP(
        comp_intensive_model.top_mlp, device_ids=[0])
    comp_intensive_model.bot_mlp = DDP(
        comp_intensive_model.bot_mlp, device_ids=[0])

    optimizer = optim.SGD(
        [
            {
                "params": comp_intensive_model.top_mlp.parameters(),
                "lr": 0.01,
            },
            {
                "params": comp_intensive_model.bot_mlp.parameters(),
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
        "--arch-mlp-bot",
        type=utils.dash_separated_ints,
        help="dimensions of the bottom mlp",
    )
    parser.add_argument(
        "--arch-mlp-top", type=utils.dash_separated_ints, help="dimensions of top mlp"
    )

    parser.add_argument(
        "--emb-size",
        type=int,
        default=48,
        help="size of the embedding for each sparse feature",
    )

    parser.add_argument(
        "--arch-interaction-op", type=str, choices=["dot", "cat"], default="dot"
    )

    parser.add_argument(
        "--lookahead-value",
        type=int,
        default=200,
        help="The number of batches further to look ahead for getting cache",
    )

    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--arch-interaction-itself",
                        action="store_true", default=False)
    parser.add_argument("--loss-function",
                        choices=["mse", "bce"], default="bce")
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
    parser.add_argument(
        "--mini-batch-size", type=int, default=128, help="The batch size to train"
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
        # "--dist-world-size",
        "--world-size-trainers",
        type=int,
        required=True,
        help="Distributed World Size for collective communication library",
    )

    parser.add_argument(
        "--oracle-prefix",
        type=str,
        default="oracle",
        help="Prefix to name oracle cacher",
    )

    parser.add_argument(
        "--trainer-prefix",
        type=str,
        default="worker",
        help="prefix to call the trainer",
    )

    parser.add_argument(
        "--emb-prefix",
        type=str,
        default="emb_worker",
        help="Name of embedding worker Currently I am assuming there is currently only one worker ID",
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
    parser.add_argument("--delay-emb-sync", action="store_true", default=False)
    parser.add_argument("--s3", action="store_true", default=False)
    args = parser.parse_args()

    if args.emb_info_file is not None:
        args.ln_emb = get_emb_length(args.emb_info_file)

    return args


if __name__ == "__main__":
    args = parse_args(argparse.ArgumentParser(
        description="Arguments for DLRM"))
    now = datetime.now().strftime("%H:%M_%B_%d_%Y")
    filename = f"training_worker_{args.dist_worker_id}_{now}_{args.logging_prefix}_bagpipe.log"
    logging.basicConfig(
        filename=filename
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.info(args)
    main(args)