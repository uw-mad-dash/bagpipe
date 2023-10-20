#!/bin/bash
worker_id=$1
world_size=$2
master_ip=$3
world_size_trainers=$4
batch_size=$5
cache_size=$6
git pull
source activate pytorch_latest_p37
python OracleCacher/oracle_cacher.py -v --prefetch --lookahead-value 200 --mini-batch-size=$batch_size --dataset-multi-num 16 --worker-id $worker_id --world-size $world_size --cache --master-ip $master_ip --world-size-trainers $world_size_trainers --emb-info-file /home/ubuntu/emb_table_info.txt --processed-csv  /home/ubuntu/fgcnn_16 --cache-size $cache_size 2>&1 | tee out
