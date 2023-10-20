#!/bin/bash
worker_id=$1 #ID in RPC
world_size=$2 #World Size for rpc
master_ip=$3 #Master IP
dist_worker_id=$4 #workerID for distributed
world_size_trainers=$5 #world size trainers
dist_master_ip=$6 # dist master IP
logging_prefix=$7
cache_size=$8
lookahead=$9
source activate pytorch_latest_p37
python examples/trainer_main.py --arch-mlp-top "1024-1024-1024-256-128-1" --arch-mlp-bot "13-512-256-64-48" --worker-id $worker_id --lookahead-value $lookahead --world-size $world_size --cache-size $cache_size --master-ip $master_ip --dist-worker-id $dist_worker_id --world-size-trainers $world_size_trainers --dist-master-ip $dist_master_ip --device cuda:0 --logging-prefix $logging_prefix --dist-backend nccl --stop-iter 2000 --s3 2>&1 | tee out
