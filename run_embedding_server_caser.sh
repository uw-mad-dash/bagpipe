#!/bin/bash
worker_id=$1
world_size=$2
master_ip=$3
git pull
source activate pytorch_latest_p37
python EmbServer/embedding_server.py --worker-id $worker_id --emb-size 50 --emb-info-file /home/ubuntu/emb_table_info_h.txt --world-size $world_size --master-ip $master_ip 2>&1 | tee out
