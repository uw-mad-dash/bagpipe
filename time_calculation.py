import time
import copy
import math
import argparse
import torch
import csv


def parse_args(parser):
    """
    Parse Args
    """
    parser.add_argument("--master-ip", required=True, type=str)
    parser.add_argument("--num-nodes", required=True, type=int)
    parser.add_argument("--rank", required=True, type=int)
    parser.add_argument("--vector-size", required=True, type=int)
    parser.add_argument("--vector-size-file", required=True, type=str)
    args = parser.parse_args()
    return args


def main(args):
    torch.distributed.init_process_group(
        backend="nccl",
        init_method=f"tcp://{args.master_ip}:6585",
        rank=args.rank,
        world_size=args.num_nodes,
    )
    emb_stats_file = open(args.vector_size_file, "r")
    reader = csv.DictReader(emb_stats_file)
    # memory_range = [0.5, 1, 2, 4, 8, 16, 32, 64, 128]
    # memory_range = [6]
    # for m in memory_range:
    total_time_p2p = 0
    total_time_all_reduce = 0
    for row in reader:
        number_of_embeddings_to_read = float(row["average"])
        # mb * 1024 *1024 / (number of floa
        embs_all_reduce = int(row["all_reduce"])
        avg_emb_to_read_write = int(number_of_embeddings_to_read / (args.num_nodes - 1))
        embedding_send = torch.rand(
            (avg_emb_to_read_write, 64), dtype=torch.float32, device="cuda:0"
        )
        embedding_buffer = [
            torch.zeros_like(embedding_send) for _ in range(args.num_nodes - 1)
        ]

        embedding_to_all_reduce = torch.rand(
            (embs_all_reduce, 64), dtype=torch.float32, device="cuda:0"
        )
        start_all_reduce = time.time()
        torch.distributed.all_reduce(embedding_to_all_reduce)
        torch.cuda.synchronize()
        end_all_reduce = time.time()
        # print("Time all reduce {}".format(end_all_reduce - start_all_reduce))

        total_time_all_reduce += end_all_reduce - start_all_reduce
        # sending embeddings
        start_time = time.time()
        count = 0
        futures_list = list()
        for i in range(args.num_nodes):
            if args.rank != i:
                torch.distributed.isend(embedding_send, i)
                request_obj = torch.distributed.irecv(embedding_buffer[count], src=i)
                futures_list.append(request_obj)
                count += 1

        for obj in futures_list:
            obj.wait()
        torch.cuda.synchronize()

        count = 0
        futures_list = list()
        for i in range(args.num_nodes):
            if args.rank != i:
                torch.distributed.isend(embedding_send, i)
                request_obj = torch.distributed.irecv(embedding_buffer[count], src=i)
                futures_list.append(request_obj)
                count += 1

        for obj in futures_list:
            obj.wait()
        torch.cuda.synchronize()
        end_time = time.time()

        # print("Send and recieve {}".format(end_time - start_time))
        total_time_p2p += end_time - start_time
    # print("Emb size {}".format(m))
    print("Total all reduce = {}".format(total_time_all_reduce))
    print("Total time p2p = {}".format(total_time_p2p))
    return None


if __name__ == "__main__":
    args = parse_args(argparse.ArgumentParser(description="Arguments for All reduce"))
    main(args)
