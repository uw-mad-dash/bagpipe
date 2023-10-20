import os
import sys
import logging
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from time import time
from datetime import datetime

from s3_utils import uploadFile
from subprocess import call

sys.path.append("..")
sys.path.append(".")

from BagPipe import init_cache


activation_getter = {
    "iden": lambda x: x,
    "relu": F.relu,
    "tanh": torch.tanh,
    "sigm": torch.sigmoid,
}

class Caser(nn.Module):
    def __init__(self, num_users, num_items, bp):
        super(Caser, self).__init__()
        self.bp = bp
        # init args
        L = 10
        dims = 50
        self.n_h = 64
        self.n_v = 16
        self.drop_ratio = 0.5
        self.ac_conv = activation_getter["relu"]
        self.ac_fc = activation_getter["relu"]

        # vertical conv layer
        self.conv_v = nn.Conv2d(1, self.n_v, (L, 1))

        # horizontal conv layer
        lengths = [i + 1 for i in range(L)]
        self.conv_h = nn.ModuleList(
            [nn.Conv2d(1, self.n_h, (i, dims)) for i in lengths]
        )

        # fully-connected layer
        self.fc1_dim_v = self.n_v * dims
        self.fc1_dim_h = self.n_h * len(lengths)
        fc1_dim_in = self.fc1_dim_v + self.fc1_dim_h
        # W1, b1 can be encoded with nn.Linear
        self.fc1 = nn.Linear(fc1_dim_in, dims)
        # W2, b2 are encoded with nn.Embedding, as we don't need to compute scores for all items
        print(self.fc1)
        # self.w2 = torch.rand(2048, 80, 100, requires_grad=True).to("cuda:0")
        # self.b2 = torch.rand(2048, 80, 1, requires_grad=True).to("cuda:0")
        self.W2 = nn.Embedding(num_items, dims + dims)
        self.b2 = nn.Embedding(num_items, 1)

        # self.W2 = torch.rand(num_items, dims+dims, requires_grad=True).to("cuda:0")
        # self.b2 = torch.rand(num_items, 1, requires_grad=True).to("cuda:0")

        # dropout
        self.dropout = nn.Dropout(self.drop_ratio)

        # weight initialization
        self.W2.weight.data.normal_(0, 1.0 / self.W2.embedding_dim)
        self.b2.weight.data.zero_()

        self.cache_x = None

    def forward(self, ls_i, item_var=None, for_pred=False, get_emb_start=None, get_emb_stop=None):
        # Embedding Look-up
        get_emb_start.record()
        emb = self.bp.get_emb(ls_i)
        get_emb_stop.record()
        item_embs = emb[1].unsqueeze(1)  # use unsqueeze() to get 4-D
        user_emb = emb[0].squeeze(1)

        # Convolutional Layers
        out, out_h, out_v = None, None, None
        # vertical conv layer
        if self.n_v:
            out_v = self.conv_v(item_embs)
            out_v = out_v.view(-1, self.fc1_dim_v)  # prepare for fully connect

        # horizontal conv layer
        out_hs = list()
        if self.n_h:
            for conv in self.conv_h:
                conv_out = self.ac_conv(conv(item_embs).squeeze(3))
                pool_out = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
                out_hs.append(pool_out)
            out_h = torch.cat(out_hs, 1)  # prepare for fully connect

        # Fully-connected Layers
        out = torch.cat([out_v, out_h], 1)
        # apply dropout
        out = self.dropout(out)

        # fully-connected layer
        z = self.ac_fc(self.fc1(out))
        # z = torch.rand(2048, 50, requires_grad=True).to("cuda:0")
        x = torch.cat([z, user_emb], 1)

        w2 = self.W2(item_var)
        b2 = self.b2(item_var)

        if for_pred:
            w2 = w2.squeeze()
            b2 = b2.squeeze()
            res = (x * w2).sum(1) + b2
        else:
            res = torch.baddbmm(b2, w2, x.unsqueeze(2)).squeeze()
        return res


def get_emb_length(in_file):
    with open(in_file, "r") as fin:
        data = fin.readlines()

    data = [int(d) for d in data]
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments for CASER")

    parser.add_argument("--lookahead-value", type=int, default=200)

    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--cache-size", type=int, required=True)
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--master-ip", type=str, default="localhost")
    parser.add_argument("--master-port", type=str, default="18000")
    parser.add_argument("--master-port-ccom", type=int, default=9988)
    parser.add_argument("--dist-backend", type=str, default="gloo")
    parser.add_argument("--dist-master-ip", type=str, required=True)
    parser.add_argument("--dist-worker-id", type=int, required=True)
    parser.add_argument("--world-size-trainers", type=int, required=True)
    parser.add_argument("--stop-iter", type=int,
                        default=1000, help="Add for logging")

    parser.add_argument("--oracle-prefix", type=str, default="oracle")

    parser.add_argument("--trainer-prefix", type=str, default="worker")

    parser.add_argument("--emb-prefix", type=str, default="emb_worker")

    parser.add_argument("--cleanup-batch-proportion", type=float, default=0.25)
    parser.add_argument("--emb-info-file", type=str, default=None)
    parser.add_argument(
        "--logging-prefix", type=str, default="test", help="Add for logging"
    )
    parser.add_argument("--s3", action="store_true", default=False)
    args = parser.parse_args()

    args.ln_emb = get_emb_length(args.emb_info_file)

    now = datetime.now().strftime("%H:%M_%B_%d_%Y")
    
    global logger
    filename=f"training_worker_{args.dist_worker_id}_{now}_{args.logging_prefix}_caser.log"
    logging.basicConfig(
        filename=filename
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.info(args)

    num_users = 71177
    num_items = 36902

    os.environ["MASTER_ADDR"] = args.master_ip
    os.environ["MASTER_PORT"] = args.master_port

    bp = init_cache(
        lookahead_value=args.lookahead_value,
        emb_size=50,
        ln_emb=args.ln_emb,
        cache_size=args.cache_size,
        device=args.device,
        trainer_world_size=args.world_size_trainers,
        cleanup_batch_proportion=args.cleanup_batch_proportion,
        worker_id=args.worker_id,
        world_size=args.world_size,
        training_worker_id=args.dist_worker_id,
        emb_optim=optim.SGD,
        emb_optim_params={"lr": 0.01},
        ragged=True,
        logger=logger
    )

    model = Caser(num_users, num_items, bp).to(args.device)

    dist.init_process_group(
        backend=args.dist_backend,
        init_method=f"tcp://{args.dist_master_ip}:{args.master_port_ccom}",
        world_size=args.world_size_trainers,
        rank=args.dist_worker_id,
    )

    model = DDP(model, device_ids=[0])

    optimizer = optim.Adam(model.parameters(), weight_decay=1e-6, lr=1e-3)

    global total_get_emb
    total_get_emb = 0
    
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
        
        next_batch_start.record()
        example = bp.next_batch(next_batch_counters)
        next_batch_stop.record()
        
        print(f"Current Iter {i}")
        logger.info(f"Current Iter {i}")
        
        forward_start.record()
        batch_targets = example["target"].squeeze().to(args.device)
        batch_negatives = example["dense_x"].to(args.device)
        items_to_predict = (
            torch.cat((batch_targets, batch_negatives), 1).to(args.device).long()
        )
        
        items_prediction = model(example["sparse_vector"], items_to_predict, get_emb_start=get_emb_start, get_emb_stop=get_emb_stop)
        forward_stop.record()

        (targets_prediction, negatives_prediction) = torch.split(
            items_prediction, [batch_targets.size(1), batch_negatives.size(1)], dim=1
        )

        optimizer.zero_grad()
        # compute the binary cross-entropy loss
        positive_loss = -torch.mean(torch.log(torch.sigmoid(targets_prediction)))
        negative_loss = -torch.mean(torch.log(1 - torch.sigmoid(negatives_prediction)))
        loss = positive_loss + negative_loss

        backward_start.record()
        loss.backward()
        backward_stop.record()

        logger.info("Loss {}".format(loss.item()))
        
        dense_optim_start.record()
        optimizer.step()
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
