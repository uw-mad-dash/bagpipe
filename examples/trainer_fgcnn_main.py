# I think this is it. All looks good for now
# training worker, accepts input from the oracle cacher
import os
import sys
import logging
import argparse
from time import time
from datetime import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import utils

from subprocess import call
from itertools import combinations
from s3_utils import uploadFile

sys.path.append("..")
sys.path.append(".")

from BagPipe import init_cache


class DistTrainModel(nn.Module):
    def __init__(
            self,
            emb_size=1,
            ln_deep=[4096, 2048, 1024, 512, 1],
            sigmoid_bot=-1,
            sigmoid_top=-1,
            feature_interaction="dot",
            interact_itself=False,
            loss_function="bce",
            share_embedding=True,
            channels=[128, 2, 2, 2],
            kernel_heights=[33, 33, 33, 33],
            pooling_sizes=[8, 2, 2, 2],
            recombined_channels=[1, 1, 1, 1],
            conv_activation="Tanh",
            conv_batch_norm=True,
            num_fields=39,
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
        self.emb_size = emb_size
        self.sigmoid_bot = sigmoid_bot
        self.sigmoid_top = sigmoid_top
        self.feature_interaction = feature_interaction
        self.interact_itself = interact_itself
        self.device = device
        self.bp = bp

        self.num_fields = num_fields
        self.share_embedding = share_embedding
        self.ln_deep = ln_deep

        if not self.share_embedding:
            assert False
            pass
        (
            channels,
            kernel_heights,
            pooling_sizes,
            recombined_channels,
        ) = self.validate_input(
            channels, kernel_heights, pooling_sizes, recombined_channels
        )
        self.fgcnn_layer = FGCNN_Layer(
            num_fields,
            emb_size,
            channels=channels,
            kernel_heights=kernel_heights,
            pooling_sizes=pooling_sizes,
            recombined_channels=recombined_channels,
            activation=conv_activation,
            batch_norm=conv_batch_norm,
        )
        self.fgcnn_layer.to(device)
        input_dim, total_features = self.compute_input_dim(
            emb_size, num_fields, channels, pooling_sizes, recombined_channels
        )
        self.inner_product_layer = InnerProductLayer(
            total_features, device=device, output="inner_product"
        )
        self.inner_product_layer.to(device)
        self.dnn = self.create_mlp([input_dim] + self.ln_deep).to(device)
        if loss_function == "bce":
            self.loss_fn = torch.nn.BCELoss(reduction="mean")
        elif loss_function == "mse":
            self.loss_fn = torch.nn.MSELoss(reduction="mean")
        self.loss_fn.to(self.device)

        return None

    def create_mlp(self, ln):
        layers = nn.ModuleList()
        print("ln: " + str(ln))
        for i in range(0, len(ln) - 1):
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

            # add ReLU for all linear layer except the last one
            if i != len(ln) - 2:
                layers.append(nn.ReLU())
        layers.append(nn.Sigmoid())

        return torch.nn.Sequential(*layers)

    def apply_mlp(self, dense_x, mlp_network):
        """
        Apply MLP on the features
        """
        # print("Shape Dense x {}".format(dense_x.shape))
        return mlp_network(dense_x)

    def compute_input_dim(
        self, embedding_dim, num_fields, channels, pooling_sizes, recombined_channels
    ):
        total_features = num_fields
        input_height = num_fields
        for i in range(len(channels)):
            input_height = int(np.ceil(input_height / pooling_sizes[i]))
            total_features += input_height * recombined_channels[i]
        input_dim = (
            int(total_features * (total_features - 1) / 2)
            + total_features * embedding_dim
        )
        return input_dim, total_features

    def validate_input(
        self, channels, kernel_heights, pooling_sizes, recombined_channels
    ):
        if not isinstance(kernel_heights, list):
            kernel_heights = [kernel_heights] * len(channels)
        if not isinstance(pooling_sizes, list):
            pooling_sizes = [pooling_sizes] * len(channels)
        if not isinstance(recombined_channels, list):
            recombined_channels = [recombined_channels] * len(channels)
        if not (
            len(channels)
            == len(kernel_heights)
            == len(pooling_sizes)
            == len(recombined_channels)
        ):
            raise ValueError(
                "channels, kernel_heights, pooling_sizes, and recombined_channels \
                              should have the same length."
            )
        return channels, kernel_heights, pooling_sizes, recombined_channels

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

    def forward(self, lS_i, target, get_emb_start, get_emb_stop):
        """
        Forward pass of the training
        """
        # first we perform bottom MLP
        # print("Dense x shape {}".format(dense_x.shape))
        # start_time = torch.cuda.Event(enable_timing=True)
        # stop_time = torch.cuda.Event(enable_timing=True)
        # start_time.record()

        # X = inputs

        # X = self.apply_emb(lS_i)
        # batch_size = X.size()[0]
        # print(batch_size)
        # TODO: Make the shape and content correct
        # print(lS_i.shape)
        get_emb_start.record()
        feature_emb = self.bp.get_emb(
            lS_i
        )  # Should have (batch_size, num_fields, embedding_length)
        get_emb_stop.record()
        feature_emb = torch.stack(feature_emb)
        feature_emb = torch.swapaxes(feature_emb, 0, 1)
        if not self.share_embedding:
            # Note: not using this. We are resuing embedding table
            assert False
            pass
            # feature_emb2 = self.fg_embedding_layer(X)
            # feature_emb2 = torch.reshape(feature_emb2, (self.num_fields, -1))
            # feature_emb2 = self.apply_emb(torch.flatten(X), self.fg_embedding_layer)
            # feature_emb2 = torch.reshape(feature_emb2, (batch_size, self.num_fields, -1))
        else:
            feature_emb2 = feature_emb
        conv_in = torch.unsqueeze(feature_emb2, 1)  # shape (bs, 1, field, emb)
        new_feature_emb = self.fgcnn_layer(conv_in)

        combined_feature_emb = torch.cat([feature_emb, new_feature_emb], dim=1)

        inner_product_vec = self.inner_product_layer(combined_feature_emb)

        dense_input = torch.cat(
            [combined_feature_emb.flatten(start_dim=1), inner_product_vec], dim=1
        )
        # start.record()
        y_pred = self.dnn(dense_input)
        loss = self.loss_fn(y_pred, target)

        return loss


def get_activation(activation):
    if isinstance(activation, str):
        if activation.lower() == "relu":
            return nn.ReLU()
        elif activation.lower() == "sigmoid":
            return nn.Sigmoid()
        elif activation.lower() == "tanh":
            return nn.Tanh()
        else:
            return getattr(nn, activation)()
    else:
        return activation


class FGCNN_Layer(nn.Module):
    """
    Input X: tensor of shape (batch_size, 1, num_fields, embedding_dim)
    """

    def __init__(
        self,
        num_fields,
        embedding_dim,
        channels=[3],
        device="cpu",
        kernel_heights=[3],
        pooling_sizes=[2],
        recombined_channels=[2],
        activation="Tanh",
        batch_norm=True,
    ):
        super(FGCNN_Layer, self).__init__()
        self.embedding_dim = embedding_dim
        conv_list = []
        recombine_list = []
        self.channels = [1] + channels  # input channel = 1
        input_height = num_fields
        for i in range(1, len(self.channels)):
            in_channel = self.channels[i - 1]
            out_channel = self.channels[i]
            kernel_height = kernel_heights[i - 1]
            pooling_size = pooling_sizes[i - 1]
            recombined_channel = recombined_channels[i - 1]
            conv_layer = (
                [
                    nn.Conv2d(
                        in_channel,
                        out_channel,
                        kernel_size=(kernel_height, 1),
                        padding=(int((kernel_height - 1) / 2), 0),
                    )
                ]
                + ([nn.BatchNorm2d(out_channel)] if batch_norm else [])
                + [
                    get_activation(activation),
                    nn.MaxPool2d(
                        (pooling_size, 1), padding=(input_height % pooling_size, 0)
                    ),
                ]
            )
            conv_list.append(nn.Sequential(*conv_layer))
            input_height = int(np.ceil(input_height / pooling_size))
            input_dim = input_height * embedding_dim * out_channel
            output_dim = input_height * embedding_dim * recombined_channel
            recombine_layer = nn.Sequential(
                nn.Linear(input_dim, output_dim), get_activation(activation)
            )
            recombine_list.append(recombine_layer)
        self.conv_layers = nn.ModuleList(conv_list)
        print(conv_list)
        self.recombine_layers = nn.ModuleList(recombine_list)
        print(recombine_list)
        # print(self.conv_layers)
        # print(self.recombine_layers)

    def forward(self, X):
        # start = torch.cuda.Event(enable_timing = True)
        # end = torch.cuda.Event(enable_timing = True)
        conv_out = X
        new_feature_list = []
        for i in range(len(self.channels) - 1):

            # start.record()
            conv_out = self.conv_layers[i](conv_out)
            # end.record()
            # torch.cuda.synchronize()
            # print("    conv " + str(i) + ": " + str(start.elapsed_time(end)))

            flatten_out = torch.flatten(conv_out, start_dim=1)

            # start.record()
            recombine_out = self.recombine_layers[i](flatten_out)
            # print(self.recombine_layers[i])
            # end.record()
            # torch.cuda.synchronize()
            # print("    recombine " + str(i) + ": " + str(start.elapsed_time(end)))
            new_feature_list.append(
                recombine_out.reshape(X.size(0), -1, self.embedding_dim)
            )
        new_feature_emb = torch.cat(new_feature_list, dim=1)
        return new_feature_emb


class InnerProductLayer(nn.Module):
    """output: product_sum_pooling (bs x 1),
    Bi_interaction_pooling (bs * dim),
    inner_product (bs x f2/2),
    elementwise_product (bs x f2/2 x emb_dim)
    """

    def __init__(self, num_fields=None, device="cpu", output="product_sum_pooling"):
        super(InnerProductLayer, self).__init__()
        self._output_type = output
        if output not in [
            "product_sum_pooling",
            "Bi_interaction_pooling",
            "inner_product",
            "elementwise_product",
        ]:
            raise ValueError(
                "InnerProductLayer output={} is not supported.".format(output)
            )
        if num_fields is None:
            if output in ["inner_product", "elementwise_product"]:
                raise ValueError(
                    "num_fields is required when InnerProductLayer output={}.".format(
                        output
                    )
                )
        else:
            # p, q = zip(*list(combinations(range(num_fields), 2)))
            # self.field_p = nn.Parameter(torch.LongTensor(p).to(device), requires_grad=False)
            # self.field_q = nn.Parameter(torch.LongTensor(q).to(device), requires_grad=False)
            self.interaction_units = int(num_fields * (num_fields - 1) / 2)
            self.upper_triange_mask = nn.Parameter(
                torch.triu(torch.ones(num_fields, num_fields), 1)
                .type(torch.bool)
                .to(device),
                requires_grad=False,
            )

    def forward(self, feature_emb):
        if self._output_type in ["product_sum_pooling", "Bi_interaction_pooling"]:
            sum_of_square = torch.sum(
                feature_emb, dim=1) ** 2  # sum then square
            square_of_sum = torch.sum(
                feature_emb ** 2, dim=1)  # square then sum
            bi_interaction = (sum_of_square - square_of_sum) * 0.5
            if self._output_type == "Bi_interaction_pooling":
                return bi_interaction
            else:
                return bi_interaction.sum(dim=-1, keepdim=True)
        elif self._output_type == "elementwise_product":
            emb1 = torch.index_select(feature_emb, 1, self.field_p)
            emb2 = torch.index_select(feature_emb, 1, self.field_q)
            return emb1 * emb2
        elif self._output_type == "inner_product":
            inner_product_matrix = torch.bmm(
                feature_emb, feature_emb.transpose(1, 2))

            flat_upper_triange = torch.masked_select(
                inner_product_matrix, self.upper_triange_mask
            )
            # print(flat_upper_triange)
            return flat_upper_triange.view(-1, self.interaction_units)


def main(args):
    channels = [10, 12, 14, 16, 18, 20, 22, 24, 26]
    # print("filter range: " + args.filter_range)
    rg = np.fromstring(args.filter_range, dtype=int, sep="-")
    channels = channels[rg[0]: rg[1] + 1]
    kernel_heights = [7] * len(channels)
    pooling_sizes = [2] * len(channels)
    recombined_channels = [2] * len(channels)

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
        loss_function=args.loss_function,
        feature_interaction=args.arch_interaction_op,
        device=args.device,
        channels=channels,
        kernel_heights=kernel_heights,
        pooling_sizes=pooling_sizes,
        recombined_channels=recombined_channels,
        bp=bp
    )

    dist.init_process_group(
        backend=args.dist_backend,
        init_method=f"tcp://{args.dist_master_ip}:{args.master_port_ccom}",
        world_size=args.world_size_trainers,
        rank=args.dist_worker_id,
    )

    comp_intensive_model.fgcnn_layer = DDP(
        comp_intensive_model.fgcnn_layer, device_ids=[0]
    )
    comp_intensive_model.dnn = DDP(comp_intensive_model.dnn, device_ids=[0])

    optimizer = optim.SGD(
        [
            {
                "params": comp_intensive_model.fgcnn_layer.parameters(),
                "lr": 0.01,
            },
            {
                "params": comp_intensive_model.dnn.parameters(),
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
            # train_example["dense_x"].to(comp_intensive_model.device),
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
    # parser.add_argument(
    #     "--arch-mlp-bot",
    #     type=utils.dash_separated_ints,
    #     help="dimensions of the bottom mlp",
    # )
    # parser.add_argument(
    #     "--arch-mlp-top", type=utils.dash_separated_ints, help="dimensions of top mlp"
    # )

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
    parser.add_argument("--filter-range", type=str, default=None)
    parser.add_argument("--delay-emb-sync", action="store_true", default=False)
    parser.add_argument("--s3", action="store_true", default=False)
    args = parser.parse_args()

    if args.emb_info_file is not None:
        args.ln_emb = get_emb_length(args.emb_info_file)

    return args


if __name__ == "__main__":
    args = parse_args(argparse.ArgumentParser(
        description="Arguments for FGCNN"))
    now = datetime.now().strftime("%H:%M_%B_%d_%Y")
    filename=f"training_worker_{args.dist_worker_id}_{now}_{args.logging_prefix}_bagpipe.log"
    logging.basicConfig(
        filename=filename
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.info(args)
    main(args)
