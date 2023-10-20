# I think this is it. All looks good for now
# training worker, accepts input from the oracle cacher
import os
import sys
import logging
import argparse

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

class DeepCrossNetworkModel(nn.Module):
    def __init__(
        self,
        embedding_dim,
        dense_in_features,
        num_sparse_features,
        dense_arch_layer_sizes,
        over_arch_layer_sizes,
        dcn_num_layers,
        dcn_low_rank_dim,
        bp,
        dense_device,
    ):
        super().__init__()

        self.dense_arch = DenseArch(
            in_features=dense_in_features,
            layer_sizes=dense_arch_layer_sizes,
            device=dense_device,
        )

        crossnet = LowRankCrossNet(
            in_features=(num_sparse_features + 1) * embedding_dim,
            num_layers=dcn_num_layers,
            low_rank=dcn_low_rank_dim,
        ).to(dense_device)

        self.inter_arch = InteractionDCNArch(
            num_sparse_features=num_sparse_features,
            crossnet=crossnet,
        )

        over_in_features: int = (num_sparse_features + 1) * embedding_dim

        self.over_arch = OverArch(
            in_features=over_in_features,
            layer_sizes=over_arch_layer_sizes,
            device=dense_device,
        )
        
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.loss_fn.to(dense_device)
        self.bp = bp
        
    def forward(
        self,
        dense_features,
        sparse_features,
        target,
        get_emb_start,
        get_emb_stop
    ) -> torch.Tensor:
        embedded_dense = self.dense_arch(dense_features)
        get_emb_start.record()
        embedded_sparse = self.bp.get_emb(sparse_features)
        get_emb_stop.record()
        concatenated_dense = self.inter_arch(
            dense_features=embedded_dense, sparse_features=embedded_sparse
        )
        logits = self.over_arch(concatenated_dense)
        loss = self.loss_fn(logits, target)
        return loss


class Perceptron(torch.nn.Module):
    def __init__(
        self,
        in_size,
        out_size,
        bias,
        activation,
        device,
    ) -> None:
        super().__init__()
        self._out_size = out_size
        self._in_size = in_size
        self._linear: nn.Linear = nn.Linear(
            self._in_size, self._out_size, bias=bias, device=device
        )
        self._activation_fn = activation

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self._activation_fn(self._linear(input))


class MLP(torch.nn.Module):
    def __init__(
        self,
        in_size,
        layer_sizes,
        bias,
        activation,
        device,
    ) -> None:
        super().__init__()

        activation = nn.ReLU()

        self._mlp: torch.nn.Module = torch.nn.Sequential(
            *[
                Perceptron(
                    layer_sizes[i - 1] if i > 0 else in_size,
                    layer_sizes[i],
                    bias=bias,
                    activation=activation,
                    device=device,
                )
                for i in range(len(layer_sizes))
            ]
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self._mlp(input)


class DenseArch(nn.Module):
    def __init__(
        self,
        in_features,
        layer_sizes,
        device,
    ) -> None:
        super().__init__()
        self.model: nn.Module = MLP(
            in_features, layer_sizes, bias=True, activation="relu", device=device
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.model(features)


class LowRankCrossNet(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        num_layers: int,
        low_rank: int = 1,
    ) -> None:
        super().__init__()
        assert low_rank >= 1, "Low rank must be larger or equal to 1"

        self._num_layers = num_layers
        self._low_rank = low_rank
        self.W_kernels: torch.nn.Module = torch.nn.ParameterList(
            [
                torch.nn.Parameter(
                    torch.nn.init.xavier_normal_(
                        torch.empty(in_features, self._low_rank)
                    )
                )
                for i in range(self._num_layers)
            ]
        )
        self.V_kernels: torch.nn.Module = torch.nn.ParameterList(
            [
                torch.nn.Parameter(
                    torch.nn.init.xavier_normal_(
                        torch.empty(self._low_rank, in_features)
                    )
                )
                for i in range(self._num_layers)
            ]
        )
        self.bias: torch.nn.Module = torch.nn.ParameterList(
            [
                torch.nn.Parameter(torch.nn.init.zeros_(torch.empty(in_features, 1)))
                for i in range(self._num_layers)
            ]
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x_0 = input.unsqueeze(2) 
        x_l = x_0

        for layer in range(self._num_layers):
            xl_w = torch.matmul(
                self.W_kernels[layer],
                torch.matmul(self.V_kernels[layer], x_l),
            )
            x_l = x_0 * (xl_w + self.bias[layer]) + x_l

        return torch.squeeze(x_l, dim=2)


class InteractionDCNArch(nn.Module):
    def __init__(self, num_sparse_features: int, crossnet: nn.Module) -> None:
        super().__init__()
        self.F: int = num_sparse_features
        self.crossnet = crossnet

    def forward(
        self, dense_features, sparse_features
    ) -> torch.Tensor:
        if self.F <= 0:
            return dense_features
        (B, D) = dense_features.shape

        combined_values = torch.cat(
            [dense_features] + sparse_features, dim=1
        )

        return self.crossnet(combined_values.reshape([B, -1]))
    
    
class OverArch(nn.Module):
    def __init__(
        self,
        in_features: int,
        layer_sizes,
        device,
    ) -> None:
        super().__init__()
        if len(layer_sizes) <= 1:
            raise ValueError("OverArch must have multiple layers.")
        self.model: nn.Module = nn.Sequential(
            MLP(
                in_features,
                layer_sizes[:-1],
                bias=True,
                activation="relu",
                device=device,
            ),
            nn.Linear(layer_sizes[-2], layer_sizes[-1], bias=True, device=device),
        )

    def forward(self, features: torch.Tensor):
        return self.model(features)


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
    
    dcn = DeepCrossNetworkModel(args.emb_size, 
                                13, 
                                len(args.ln_emb),
                                dense_arch_layer_sizes=[1024, 512, 256, 64, 48],
                                over_arch_layer_sizes=[1024, 512, 256, 1],
                                dcn_num_layers=2,
                                dcn_low_rank_dim=8,
                                bp=bp,
                                dense_device=args.device)
    
    dist.init_process_group(
        backend=args.dist_backend,
        init_method=f"tcp://{args.dist_master_ip}:{args.master_port_ccom}",
        world_size=args.world_size_trainers,
        rank=args.dist_worker_id,
    )

    dcn.dense_arch = DDP(dcn.dense_arch, device_ids=[0])
    dcn.inter_arch = DDP(dcn.inter_arch, device_ids=[0])
    dcn.over_arch = DDP(dcn.over_arch, device_ids=[0])

    optimizer = optim.SGD(
        [
            {
                "params": dcn.dense_arch.parameters(),
                "lr": 0.01,
            },
            {
                "params": dcn.inter_arch.parameters(),
                "lr": 0.01,
            },
            {
                "params": dcn.over_arch.parameters(),
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
        loss = dcn.forward(train_example["dense_x"].to(args.device), 
                           train_example["sparse_vector"].to(args.device), 
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
        description="Arguments for D&C"))
    now = datetime.now().strftime("%H:%M_%B_%d_%Y")
    filename=f"training_worker_{args.dist_worker_id}_{now}_{args.logging_prefix}_bagpipe.log"
    logging.basicConfig(
        filename=filename
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.info(args)
    main(args)
