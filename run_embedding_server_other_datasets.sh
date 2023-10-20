#!/bin/bash
worker_id=$1
world_size=$2
master_ip=$3
emb_info_file=$4
git pull
source activate pytorch_latest_p37
python embedding_server.py --worker-id $worker_id  --world-size $world_size --master-ip $master_ip --emb-info-file $emb_info_file 2>&1 | tee err_emb.log
