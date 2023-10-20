#!/bin/bash
worker_id=$1
world_size=$2
master_ip=$3
world_size_trainers=$4
csv_location=$5
emb_info_file=$6
git pull
source activate pytorch_latest_p37
python oracle_cacher.py --lookahead-value 1 --arch-mlp-bot="13-512-256-64-16" --data-generation=dataset --data-set=kaggle --raw-data-file=/home/saurabh/Work/dlrm_simulator/criteo_kaggle_data/train.txt  --processed-data-file=/home/saurabh/Work/dlrm_simulator/criteo_kaggle_data/kaggleAdDisplayChallenge_processed.npz  --mini-batch-size=16384 --dataset-multiprocessing --worker-id $worker_id --world-size $world_size --master-ip $master_ip --world-size-trainers $world_size_trainers --processed-csv $csv_location --cache-size 6000000 --emb-info-file $emb_info_file 2>&1 | tee out
