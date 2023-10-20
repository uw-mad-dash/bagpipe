source activate pytorch_latest_p37
for csize in 1000000 2000000 3000000 4000000 5000000 6000000 7000000 8000000 9000000
do
	for lk_val in 50 100 200 400 800 1000
	do
		for feature in 30 40 60 100 200
		do
			for bsize in 16384 32768 65536 131072
			do
				python oracle_cacher_benchmark.py --processed-csv /home/ubuntu/kaggle_criteo_weekly.txt --mini-batch-size $bsize --num-features $feature --logging-prefix 10 --worker-id 10 --cache-size $csize --world-size 2 --world-size-trainers 10
			done
		done
	done
done
