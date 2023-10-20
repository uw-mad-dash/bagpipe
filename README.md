## Replicating for SOSP artifact evaluation.
For performing artifact evaluation and reproducing results of Bagpipe. 
We have provided reviewers with a private key and the ip address of a instance. 
The authors can ssh into the instance and then run the following commands.
```
source /home/ubuntu/ptorch/bin/activate
cd bagpipe_reset
python ec2_launcher_six_models.py
```
This script will automatically launch an ec2 cluster with the right AMI and then launch the right scripts on each machine of the cluster.
By default this script will run the DLRM model. The reviewers can modify the specific models to run by changing the arguments around Line 600 in the ec2_launcher_six_models.py. 
Post running this script will automatically terminate the cluster to save on EC2 costs and copy back the log files from the trainers.
The file names will include the prefix training\_worker, followed by a time stamp and then model name, global batch size etc. 
The reviewers can then parse the log file using parse_worker_file.py.
```
python parse_worker_file.py file_name
```
This script will output the Average per iter time and the respective standard deviation.
This scripts will reproduce the numbers for Bagpipe in Figure 9 on the DLRM model by outputing the Average per iter time and the respective standard deviation.


##### Replicating torchrec baseline
To replicate torchrec baseline. Please run - 
```
python ec2_launcher_six_models_torchrec_baseline.py
```
The script launches DLRM model and collects back the right log files. The reviewers can look at the end of log file to view per iteration time.
Please note for Bagpipe the batch size is specified per node, e.g., 2048 on 8 nodes will translate to global batch size 16384.


#### Using the AWS launch script 
All the evaluation performed for Bagpipe uses distributed setup. In order to reduce the effort of allocating resources on AWS, we provide an AWS launch script. 
The AWS launch script given right credentials will automatically launch resources and perform evaluation.

To setup up AWS CLI please look at the [https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-files.html](AWS CLI Configuration) documentation. 
Next setup your account to use EC2 using a key-pair. For details about setting this up please look at [https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-key-pairs.html](EC2 linux key pair setup guide). 

Please make sure that the AWS account is configured to request atleast 8 V100 GPUs and 2 C5.18xlarge. 
#### Using EC2 launch script. 
For reproducing Bagpipe's runtime on six different models please use the script - ```ec2_launcher_six_models.py```. Before you run the script.
Please look at the launch_cfg. Within the launch_cfg you would need to update the following fields with your aws setup details - 

1. key_name
2. key_path
3. region
4. az
5. security_group

Please make sure that the security group has all the ports open and allows traffic from all IP addresses. 

All the data is present in the AMI. However, it is known quirk with AWS that first read from a newly launched instance with AMI will be slow. So we perform a warmup by counting the number of lines. Please be patient with this as this can take time.
#### Reproducing Results of Bagpipe on Different Models

Once you launch ec2_launcher_six_models.py. The script will automatically allocate resources. And launch the training. 
Post training it will scp back the runtime from all the scripts back to your local machine.







# BagPipe API Usage

#### Abstract
BagPipe provides an API for distributed DLRM training. It offers batches and related embeddings for that batch while hiding the process of fetching and syncing the embeddings, making it easier for users to perform DLRM without tweaking the source code of the bagpipe.

#### Introduction
Bagpipe is composed of three components - oracle cacher, embedding server(s), and distributed trainer(s). Oracle cacher and embedding servers are standalone components and are usually configured using command line parameters. 
However, distributed trainers need to interact with the user provided model. To simplify integrating a new recommendation model with Bagpipe we provide a set of well defined APIs. 
Using the API's a user can add any new recommendation model and train the model with Bagpipe.

#### BagCache
BagCache is a class originating from the distributed trainer. It holds a cache for embeddings.

In order to use Bagpipe with a recommendation model the user needs to import the BagCache training class. The BagCache class configures Bagpipe and provides access to the several functionalities provided by Bagpipe. 

```ruby
BagPipe.init_cache(lookahead_value=200, emb_size=1, ln_emb=[], cache_size=25000,
                   cleanup_batch_proportion=0.25, emb_optim=None, emb_optim_params=None, device="cuda:0",
                   worker_id=0, world_size=None, training_worker_id=0, trainer_world_size=None)
```
Returns a new BagCache object. All parameters should be passed as keyword arguments.

##### Parameters
- **lookahead_value** (*int, optional*) - The number of batches ahead of the current batch we are going to analyze to determine what to cache.
- **emb_size** (*int*) - Size of the embedding for each sparse feature.
- **ln_emb** (*List[int]*) - Number of unique categorical variables in a given embedding table.
- **cache_size** (*int*) - Size of the dictionary of embeddings the cache will hold.
- **cleanup_batch_proportion** (*float, optional*) - The proportion of lookahead value at which we evict and fetch (from 0 to 1).
- **device** (*str, optional*) - Device type and optional device ordinal for the device type.
- **worker_id** (*int*) - A global identifier for rpc within oracle cacher, embedding server(s), and trainer(s).
- **world_size** (*int*) - Sum of the number of oracle cacher, embedding server(s), and trainer(s).
- **training_worker_id** (*int*) - A global identifier for distributed service within trainer(s).
- **trainer_world_size** (*int*) - The number of trainer(s).
- **emb_optim** (*torch.optim*) - Class of torch.optim (eg: torch.optim.SGD).
- **emb_optim_params** (*dict*) - Optimization options the optimizer passed into BagCache (eg: {“lr”: 0.01}). Do not pass in model parameters.

#### Methods
- **CLASSMETHOD** ```next_batch()```
  
  Retrieve the next batch (A training example) from BagCache. 
  
  Returns: A dictionary with the following keys
    - dense_x: Continuous features processed with a bottom multilayer perceptron
    - sparse_vector: Categorical features processed using embeddings
    - target: Target of training example
    
- **CLASSMETHOD** ```get_emb(ls_i)```
  
  Retrieve the embeddings from the instance of BagCache.
  
  Parameters:
  - **ls_i** (*List[[int]]*) - List of list of embedding ids
  
  Returns: List of list of embeddings

#### Basic Use Case
- trainer_simple.py
```ruby
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
```

#### Example Usage On trainer_simple.py

>About Data kaggle_criteo_weekly.txt
>
>The training set consists of a portion of Criteo's traffic over a period of 7 days. Each row corresponds to a display ad served by Criteo. Positive (clicked) and 
>negatives (non-clicked) examples have both been subsampled at different rates in order to reduce the dataset size. The examples are chronologically ordered.

Create two trainers
```
python trainer_simple.py
```
Create the embedding server
```
python embedding_server.py --worker-id 3 --world-size 4 --master-ip 127.0.0.1
```
Create the oracle cacher
```
python oracle_cacher.py --prefetch --lookahead-value 200 --mini-batch-size=4096 --dataset-multiprocessing --worker-id 0 --world-size 4 --cache --master-ip 127.0.0.1 --world-size-trainers 2  --processed-csv  ../../kaggle_criteo_weekly.txt --cache-size 5000000
```

#### Example Usage On trainer_main.py (DLRM)
1. node0: 10.10.1.2
2. node1: 10.10.1.3
3. node2: 10.10.1.1

>About Data kaggle_criteo_weekly.txt
>
>The training set consists of a portion of Criteo's traffic over a period of 7 days. Each row corresponds to a display ad served by Criteo. Positive (clicked) and 
>negatives (non-clicked) examples have both been subsampled at different rates in order to reduce the dataset size. The examples are chronologically ordered.

##### On trainer 0
```
TP_SOCKET_IFNAME=enp94s0f0 GLOO_SOCKET_IFNAME=enp94s0f0 python trainer_main.py --arch-mlp-top "1024-512-256-1" --arch-mlp-bot "13-1024-1024-512-256-64-48" --worker-id 1 --lookahead-val 200 --world-size 4 --cache-size 5000000 --master-ip 10.10.1.1 --dist-worker-id 0 --world-size-trainers 2 --dist-master-ip 10.10.1.2 --device cuda:0 --logging-prefix lookahead_200_batch_size_4096_optimized --dist-backend gloo
```

##### On trainer 1
```
TP_SOCKET_IFNAME=enp94s0f0 GLOO_SOCKET_IFNAME=enp94s0f0 python trainer_main.py --arch-mlp-top "1024-512-256-1" --arch-mlp-bot "13-1024-1024-512-256-64-48" --worker-id 2 --lookahead-val 200 --world-size 4 --cache-size 5000000 --master-ip 10.10.1.1 --dist-worker-id 1 --world-size-trainers 2 --dist-master-ip 10.10.1.2 --device cuda:0 --logging-prefix lookahead_200_batch_size_4096_optimized --dist-backend gloo
```

##### On embedding server
```
TP_SOCKET_IFNAME=enp94s0f0 python embedding_server.py --worker-id 3 --world-size 4 --master-ip 10.10.1.1
```

#### On oracle cacher
```
TP_SOCKET_IFNAME=enp94s0f0 python oracle_cacher.py --prefetch --lookahead-value 200 --mini-batch-size=4096 --dataset-multiprocessing --worker-id 0 --world-size 4 --cache --master-ip 10.10.1.1 --world-size-trainers 2  --processed-csv  ../../kaggle_criteo_weekly.txt --cache-size 5000000
```

## Example Usage On trainer_fgcnn_main.py (FGCNN)
1. node0: 10.10.1.2
2. node1: 10.10.1.3
3. node2: 10.10.1.1

>About Data parsed_train.txt
>
>Another data for FGCNN

### On trainer 0
```
TP_SOCKET_IFNAME=enp94s0f0 GLOO_SOCKET_IFNAME=enp94s0f0 python trainer_fgcnn_main.py --worker-id 1 --lookahead-val 200 --world-size 4 --cache-size 8000000 --master-ip 10.10.1.1 --dist-worker-id 0 --world-size-trainers 2 --dist-master-ip 10.10.1.2 --device cuda:0 --filter-range "2-5" --emb-info-file ../emb_table_info.txt --logging-prefix lookahead_200_batch_size_4096_optimized --dist-backend gloo
```

### On trainer 1
```
TP_SOCKET_IFNAME=enp94s0f0 GLOO_SOCKET_IFNAME=enp94s0f0 python trainer_fgcnn_main.py --worker-id 2 --lookahead-val 200 --world-size 4 --cache-size 8000000 --master-ip 10.10.1.1 --dist-worker-id 1 --world-size-trainers 2 --dist-master-ip 10.10.1.2 --device cuda:0 --filter-range "2-5" --emb-info-file ../emb_table_info.txt --logging-prefix lookahead_200_batch_size_4096_optimized --dist-backend gloo
```

### On embedding server
```
TP_SOCKET_IFNAME=enp94s0f0 python embedding_server.py --worker-id 3 --world-size 4 --master-ip 10.10.1.1 --emb-info-file ../emb_table_info.txt
```

### On oracle cacher
```
TP_SOCKET_IFNAME=enp94s0f0 python oracle_cacher.py --prefetch --lookahead-value 200 --mini-batch-size=4096 --dataset-multiprocessing --worker-id 0 --world-size 4 --cache --master-ip 10.10.1.1 --world-size-trainers 2  --emb-info-file ../../emb_table_info.txt  --processed-csv  ../../parsed_train.txt --cache-size 8000000
```

## Example Usage On trainer_caser_main.py (CASER)
1. node0: 10.10.1.2
2. node1: 10.10.1.3
3. node2: 10.10.1.1

>About Data parsed_caser.txt
>
>Another data for CASER

### On trainer 0
```
TP_SOCKET_IFNAME=enp94s0f0 GLOO_SOCKET_IFNAME=enp94s0f0 python trainer_caser_main.py --worker-id 1 --lookahead-val 200 --world-size 4 --cache-size 8000000 --master-ip 10.10.1.1 --dist-worker-id 0 --world-size-trainers 2 --dist-master-ip 10.10.1.2 --device cuda:0 --emb-info-file ../../emb_table_info.txt --dist-backend gloo
```

### On trainer 1
```
TP_SOCKET_IFNAME=enp94s0f0 GLOO_SOCKET_IFNAME=enp94s0f0 python trainer_caser_main.py --worker-id 2 --lookahead-val 200 --world-size 4 --cache-size 8000000 --master-ip 10.10.1.1 --dist-worker-id 1 --world-size-trainers 2 --dist-master-ip 10.10.1.2 --device cuda:0 --emb-info-file ../../emb_table_info.txt --dist-backend gloo
```

### On embedding server
```
TP_SOCKET_IFNAME=enp94s0f0 python embedding_server.py --emb-size 50 --worker-id 3 --world-size 4 --master-ip 10.10.1.1 --emb-info-file ../../emb_table_info.txt
```

### On oracle cacher
```
TP_SOCKET_IFNAME=enp94s0f0 python oracle_cacher.py --prefetch --lookahead-value 200 --mini-batch-size=512 --dataset-multiprocessing --worker-id 0 --world-size 4 --cache --master-ip 10.10.1.1 --world-size-trainers 2  --emb-info-file ../../emb_table_info.txt  --processed-csv  ../../parsed_caser.txt --cache-size 8000000
```
## Replicating for SOSP artifact evaluation.
For performing artifact evaluation and reproducing results of Bagpipe. 
We have provided reviewers with a private key and the ip address of a instance. 
The authors can ssh into the instance and then run the following commands.
```
source /home/ubuntu/ptorch/bin/activate
cd bagpipe_reset
python ec2_launcher_six_models.py
```
This script will automatically launch an ec2 cluster with the right AMI and then launch the right scripts on each machine of the cluster.
By default this script will run the DLRM model. The reviewers can modify the specific models to run by changing the arguments around Line 600 in the ec2_launcher_six_models.py. 
Post running this script will automatically terminate the cluster to save on EC2 costs and copy back the log files from the trainers.
The file names will include the prefix training\_worker, followed by a time stamp and then model name, global batch size etc. 
The reviewers can then parse the log file using parse_worker_file.py.
```
python parse_worker_file.py file_name
```
This script will output the Average per iter time and the respective standard deviation.
This scripts will reproduce the numbers for Bagpipe in Figure 9 on the DLRM model by outputing the Average per iter time and the respective standard deviation.

### Replicating torchrec baseline
To replicate torchrec baseline. Please run - 
```
python ec2_launcher_six_models_torchrec_baseline.py
```
The script launches DLRM model and collects back the right log files. The reviewers can look at the end of log file to view per iteration time.
Please note for Bagpipe the batch size is specified per node, e.g., 2048 on 8 nodes will translate to global batch size 16384.


## Cite
Please cite the following paper if you use Bagpipe. 
```
    @inproceedings{10.1145/3600006.3613142,
    author = {Agarwal, Saurabh and Yan, Chengpo and Zhang, Ziyi and Venkataraman, Shivaram},
    title = {Bagpipe: Accelerating Deep Recommendation Model Training},
    year = {2023},
    isbn = {9798400702297},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3600006.3613142},
    doi = {10.1145/3600006.3613142},
    booktitle = {Proceedings of the 29th Symposium on Operating Systems Principles},
    pages = {348–363},
    numpages = {16},
    keywords = {distributed training, recommendation models},
    location = {Koblenz, Germany},
    series = {SOSP '23}
    }
```
