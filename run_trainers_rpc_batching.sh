#!/bin/bash
worker_id=$1 #ID in RPC
world_size=$2 #World Size for rpc
master_ip=$3 #Master IP
dist_worker_id=$4 #workerID for distributed
world_size_trainers=$5 #world size trainers
dist_master_ip=$6 # dist master IP
logging_prefix=$7
stop_iter=$8
cleanup_batch_proportion=$9
lk_val=100


git pull
source activate pytorch_latest_p37
python distributed_trainer_emb_bag.py --arch-mlp-top "1024-1024-1024-512-256-1" --arch-mlp-bot "13-512-256-64-48" --worker-id $worker_id --lookahead-val $lk_val --world-size $world_size --cache-size 6000000 --master-ip $master_ip --dist-worker-id $dist_worker_id --world-size-trainers $world_size_trainers --dist-master-ip $dist_master_ip --device cuda:0 --logging-prefix $logging_prefix --dist-backend gloo --stop-iter $stop_iter --cleanup-batch-proportion $cleanup_batch_proportion  2>&1 | tee out
