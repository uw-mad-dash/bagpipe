import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from subprocess import call

import sys

sys.path.append("..")
sys.path.append(".")

from BagPipe import init_cache


class SimpleModel(nn.Module):
    def __init__(
            self,
            ln_emb_len,
            device,
            bp=None
    ):
        super(SimpleModel, self).__init__()
        self.device = device
        self.bp = bp

        bot1 = nn.Linear(13, 64)
        bot2 = nn.Linear(64, 48)
        top1 = nn.Linear((ln_emb_len + 1) * 48, 256)
        top2 = nn.Linear(256, 1)
        self.bot_mlp = torch.nn.Sequential(bot1, bot2, nn.ReLU())
        self.top_mlp = torch.nn.Sequential(top1, top2, nn.Sigmoid())
        self.top_mlp.to(self.device)
        self.bot_mlp.to(self.device)
        self.loss_fn = torch.nn.BCELoss(reduction="mean")
        self.loss_fn.to(self.device)

    def forward(self, dense_x, ls_i, target):
        x = self.bot_mlp(dense_x)
        ly = self.bp.get_emb(ls_i)
        ly = [l.squeeze(1) for l in ly]
        z = torch.cat([x] + ly, dim=1)
        p = self.top_mlp(z)
        loss = self.loss_fn(p, target)
        return loss


def main(rank, rank_d, size, size_d, ip):
    ln_emb = [1460, 583, 10131227, 2202608, 305, 24, 12517, 633, 3, 93145, 5683, 8351593,
              3194, 27, 14992, 5461306, 10, 5652, 2173, 4, 7046547, 18, 15, 286181, 105, 142572]

    os.environ["MASTER_ADDR"] = ip
    os.environ["MASTER_PORT"] = "18000"

    bp = init_cache(emb_size=48, ln_emb=ln_emb, cache_size=5000000, device="cuda:0",
                    cleanup_batch_proportion=0.25, worker_id=rank,
                    world_size=size, training_worker_id=rank_d,
                    trainer_world_size=size_d,
                    emb_optim=optim.SGD, emb_optim_params={"lr": 0.01})

    model = SimpleModel(ln_emb_len=len(ln_emb), device="cuda:0", bp=bp)

    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://{ip}:{9988}",
        world_size=size_d,
        rank=rank_d,
    )

    model.top_mlp = DDP(model.top_mlp, device_ids=[0])
    model.bot_mlp = DDP(model.bot_mlp, device_ids=[0])

    optimizer = optim.SGD(
        [
            {
                "params": model.top_mlp.parameters(),
                "lr": 0.01,
            },
            {
                "params": model.bot_mlp.parameters(),
                "lr": 0.01,
            },
        ]
    )

    # train loop
    # running for 100 iterations
    for _ in range(100):
        train_example = bp.next_batch()

        loss = model.forward(
            train_example["dense_x"].to(model.device),
            train_example["sparse_vector"],
            train_example["target"].to(model.device),
        )

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

    call("pkill -9 python", shell=True)


if __name__ == "__main__":
    size = 4
    size_d = 2

    mp.set_start_method("spawn")
    for rank_d in range(size_d):
        mp.Process(target=main, args=(rank_d + 1, rank_d, size, size_d, "127.0.0.1")).start()
