import sys
import boto3
import time

from pssh.clients import ParallelSSHClient


def kill_process(
    client_trainers, client_emb_server, client_oracle_cacher, private_ip_trainers
):

    run_args_kill_trainers = [
        {"cmd": "pkill -9 python"} for i in range(len(private_ip_trainers))
    ]

    run_args_kill_oracle = [{"cmd": "pkill -9 python"}]
    run_args_kill_emb_server = [{"cmd": "pkill -9 python"}]

    kill_trainers = client_trainers.run_command(
        "%(cmd)s", host_args=run_args_kill_trainers
    )

    kill_emb_server = client_emb_server.run_command(
        "%(cmd)s", host_args=run_args_kill_emb_server
    )

    kill_oracle_cacher = client_oracle_cacher.run_command(
        "%(cmd)s", host_args=run_args_kill_oracle
    )
    print("Launched python kill command")

    time.sleep(60)


def other_data_args_emb_server(
    private_ip_trainers, private_ip_oracle_cacher, emb_info_file
):

    run_args_emb_server = [
        {
            "cmd": "rm -rf bagpipe && git clone git@github.com:iidsample/bagpipe.git && cd bagpipe && bash run_embedding_server_other_datasets.sh {} {} {} {}".format(
                1,
                (len(private_ip_trainers) + 2),
                private_ip_oracle_cacher,
                emb_info_file,
            )
        }
    ]

    return run_args_emb_server


def fcgnn_args_oracle_server(
    private_ip_trainers,
    private_ip_oracle_cacher,
    csv_location,
    emb_info_file,
    batch_size,
):

    run_args_oracle_cacher = [
        {
            "cmd": "rm -rf bagpipe && git clone git@github.com:iidsample/bagpipe.git && cd bagpipe && git checkout add_fgcnn && bash run_oracle_server_other_datasets.sh {} {} {} {} {} {} {}".format(
                0,
                (len(private_ip_trainers) + 2),
                private_ip_oracle_cacher,
                len(private_ip_trainers),
                csv_location,
                emb_info_file,
                batch_size,
            )
        }
    ]

    return run_args_oracle_cacher


def fcgnn_args_oracle_server_no_cache_no_pref(
    private_ip_trainers,
    private_ip_oracle_cacher,
    csv_location,
    emb_info_file,
    batch_size,
):

    run_args_oracle_cacher = [
        {
            "cmd": "rm -rf bagpipe && git clone git@github.com:iidsample/bagpipe.git && cd bagpipe && git checkout add_fgcnn && bash run_oracle_server_no_cache_no_prefetch_other_datasets.sh {} {} {} {} {} {} {}".format(
                0,
                (len(private_ip_trainers) + 2),
                private_ip_oracle_cacher,
                len(private_ip_trainers),
                csv_location,
                emb_info_file,
                batch_size,
            )
        }
    ]

    return run_args_oracle_cacher


def other_data_args_oracle_server(
    private_ip_trainers, private_ip_oracle_cacher, csv_location, emb_info_file
):

    run_args_oracle_cacher = [
        {
            "cmd": "rm -rf bagpipe && git clone git@github.com:iidsample/bagpipe.git && cd bagpipe && bash run_oracle_server_other_datasets.sh {} {} {} {} {} {}".format(
                0,
                (len(private_ip_trainers) + 2),
                private_ip_oracle_cacher,
                len(private_ip_trainers),
                csv_location,
                emb_info_file,
            )
        }
    ]

    return run_args_oracle_cacher


def other_data_args_oracle_server_no_cache_no_prefetch(
    private_ip_trainers, private_ip_oracle_cacher, csv_location, emb_info_file
):

    run_args_oracle_cacher = [
        {
            "cmd": "rm -rf bagpipe && git clone git@github.com:iidsample/bagpipe.git && cd bagpipe && bash run_oracle_server_no_cache_no_prefetch_other_datasets.sh {} {} {} {} {} {}".format(
                0,
                (len(private_ip_trainers) + 2),
                private_ip_oracle_cacher,
                len(private_ip_trainers),
                csv_location,
                emb_info_file,
            )
        }
    ]

    return run_args_oracle_cacher


def data_download_avazu_single_machine_fae():
    bucket_name = "recommendation-data-bagpipe"
    run_download = [
        {
            "cmd": f"aws s3 cp s3://{bucket_name}/avazu_data ./ && aws s3 cp s3://{bucket_name}/avazu_emb_info ./ && aws s3 cp s3://{bucket_name}/avazu_normal.npz ./ && aws s3 cp s3://{bucket_name}/avazu_hot.npz ./ && aws s3 cp s3://{bucket_name}/avazu_hot_emb_dict.npz ./ "
        }
    ]
    return run_download


def data_download_avazu_trainer_machine_fae(private_ip_trainers):
    bucket_name = "recommendation-data-bagpipe"
    run_download = [
        {
            "cmd": f"aws s3 cp s3://{bucket_name}/avazu_data ./ && aws s3 cp s3://{bucket_name}/avazu_emb_info ./ && aws s3 cp s3://{bucket_name}/avazu_normal.npz ./ && aws s3 cp s3://{bucket_name}/avazu_hot.npz ./ && aws s3 cp s3://{bucket_name}/avazu_hot_emb_dict.npz ./ "
        }
        for i in range(len(private_ip_trainers))
    ]
    return run_download


def data_download_avazu_single_machine():
    run_download = [
        {
            "cmd": "aws s3 cp s3://recommendation-data-bagpipe/avazu_data ./ && aws s3 cp s3://recommendation-data-bagpipe/avazu_emb_info ./"
        }
    ]
    return run_download


def data_download_avazu_trainer_machine(private_ip_trainers):
    run_download = [
        {
            "cmd": "aws s3 cp s3://recommendation-data-bagpipe/avazu_data ./ && aws s3 cp s3://recommendation-data-bagpipe/avazu_emb_info ./"
        }
        for i in range(len(private_ip_trainers))
    ]

    return run_download


def data_download_movie_lens_single_machine():
    run_download = [
        {
            "cmd": "aws s3 cp s3://recommendation-data-bagpipe/movielen_data ./ && aws s3 cp s3://recommendation-data-bagpipe/movielen_emb_info ./"
        }
    ]

    return run_download


def data_download_movie_lens_trainer_machine(private_ip_trainers):
    run_download = [
        {
            "cmd": "aws s3 cp s3://recommendation-data-bagpipe/movielen_data ./ && aws s3 cp s3://recommendation-data-bagpipe/movielen_emb_info ./"
        }
        for i in range(len(private_ip_trainers))
    ]

    return run_download


def data_download_fcgnn(private_ip_trainers):

    run_download = [
        {
            "cmd": "aws s3 cp s3://recommendation-data-bagpipe/parsed_train.txt ./ && aws s3 cp s3://recommendation-data-bagpipe/emb_table_info.txt ./"
        }
        for i in range(len(private_ip_trainers))
    ]

    return run_download


def data_download_fcgnn_single():

    run_download = [
        {
            "cmd": "aws s3 cp s3://recommendation-data-bagpipe/parsed_train.txt ./ && aws s3 cp s3://recommendation-data-bagpipe/emb_table_info.txt ./"
        }
    ]

    return run_download


def other_data_args_trainer(
    private_ip_trainers,
    private_ip_oracle_cacher,
    log_file_name,
    num_iters,
    emb_info_file,
):

    run_args_trainers = [
        {
            "cmd": "rm -rf bagpipe && git clone git@github.com:iidsample/bagpipe.git && cd bagpipe && git checkout hide_cache_sync && bash run_trainers_other_datasets.sh {} {} {} {} {} {} {} {} {}".format(
                i + 2,
                (len(private_ip_trainers) + 2),
                private_ip_oracle_cacher,
                i,
                len(private_ip_trainers),
                private_ip_trainers[0],
                log_file_name,
                num_iters,
                emb_info_file,
            )
        }
        for i in range(len(private_ip_trainers))
    ]

    return run_args_trainers


def fcgnn_other_data_args_trainer_no_hide(
    private_ip_trainers,
    private_ip_oracle_cacher,
    log_file_name,
    num_iters,
    emb_info_file,
):

    run_args_trainers = [
        {
            "cmd": "rm -rf bagpipe && git clone git@github.com:iidsample/bagpipe.git && cd bagpipe && git checkout add_fgcnn && bash run_trainers_fgcnn_no_hide.sh {} {} {} {} {} {} {} {} {} {}".format(
                i + 2,
                (len(private_ip_trainers) + 2),
                private_ip_oracle_cacher,
                i,
                len(private_ip_trainers),
                private_ip_trainers[0],
                log_file_name,
                num_iters,
                emb_info_file,
                "2-5",
            )
        }
        for i in range(len(private_ip_trainers))
    ]

    return run_args_trainers


def fcgnn_other_data_args_trainer_no_cache_no_pref(
    private_ip_trainers,
    private_ip_oracle_cacher,
    log_file_name,
    num_iters,
    emb_info_file,
):

    run_args_trainers = [
        {
            "cmd": "rm -rf bagpipe && git clone git@github.com:iidsample/bagpipe.git && cd bagpipe && git checkout add_fgcnn && bash run_trainers_fgcnn_no_cache_no_prefetch.sh {} {} {} {} {} {} {} {} {} {}".format(
                i + 2,
                (len(private_ip_trainers) + 2),
                private_ip_oracle_cacher,
                i,
                len(private_ip_trainers),
                private_ip_trainers[0],
                log_file_name,
                num_iters,
                emb_info_file,
                "2-5",
            )
        }
        for i in range(len(private_ip_trainers))
    ]

    return run_args_trainers


def fcgnn_other_data_args_trainer(
    private_ip_trainers,
    private_ip_oracle_cacher,
    log_file_name,
    num_iters,
    emb_info_file,
):

    run_args_trainers = [
        {
            "cmd": "rm -rf bagpipe && git clone git@github.com:iidsample/bagpipe.git && cd bagpipe && git checkout add_fgcnn && bash run_trainers_fgcnn.sh {} {} {} {} {} {} {} {} {} {}".format(
                i + 2,
                (len(private_ip_trainers) + 2),
                private_ip_oracle_cacher,
                i,
                len(private_ip_trainers),
                private_ip_trainers[0],
                log_file_name,
                num_iters,
                emb_info_file,
                "2-5",
            )
        }
        for i in range(len(private_ip_trainers))
    ]

    return run_args_trainers


def other_data_args_trainer_no_hide_cache_sync(
    private_ip_trainers,
    private_ip_oracle_cacher,
    log_file_name,
    num_iters,
    emb_info_file,
):

    run_args_trainers = [
        {
            "cmd": "rm -rf bagpipe && git clone git@github.com:iidsample/bagpipe.git && cd bagpipe && bash run_trainers_other_datasets.sh {} {} {} {} {} {} {} {} {}".format(
                i + 2,
                (len(private_ip_trainers) + 2),
                private_ip_oracle_cacher,
                i,
                len(private_ip_trainers),
                private_ip_trainers[0],
                log_file_name,
                num_iters,
                emb_info_file,
            )
        }
        for i in range(len(private_ip_trainers))
    ]

    return run_args_trainers


def other_data_args_trainer_no_cache_no_prefetch(
    private_ip_trainers,
    private_ip_oracle_cacher,
    log_file_name,
    num_iters,
    emb_info_file,
):

    run_args_trainers = [
        {
            "cmd": "rm -rf bagpipe && git clone git@github.com:iidsample/bagpipe.git && cd bagpipe && bash run_trainers_no_cache_no_prefetch_other_datasets.sh {} {} {} {} {} {} {} {} {}".format(
                i + 2,
                (len(private_ip_trainers) + 2),
                private_ip_oracle_cacher,
                i,
                len(private_ip_trainers),
                private_ip_trainers[0],
                log_file_name,
                num_iters,
                emb_info_file,
            )
        }
        for i in range(len(private_ip_trainers))
    ]

    return run_args_trainers


def return_args_trainers_bagpipe(
    private_ip_trainers, private_ip_oracle_cacher, log_file_name, num_iters
):
    """
    Arguments for trainers
    """

    run_args_trainers = [
        {
            "cmd": "rm -rf bagpipe && git clone git@github.com:iidsample/bagpipe.git && cd bagpipe && bash run_trainers.sh {} {} {} {} {} {} {} {}".format(
                i + 2,
                (len(private_ip_trainers) + 2),
                private_ip_oracle_cacher,
                i,
                len(private_ip_trainers),
                private_ip_trainers[0],
                log_file_name,
                num_iters,
            )
        }
        for i in range(len(private_ip_trainers))
    ]

    return run_args_trainers


def return_args_trainers_bagpipe_no_cache_no_prefetch(
    private_ip_trainers, private_ip_oracle_cacher, log_file_name, num_iters
):
    """
    Arguments for trainers
    """

    run_args_trainers = [
        {
            "cmd": "rm -rf bagpipe && git clone git@github.com:iidsample/bagpipe.git && cd bagpipe && bash run_trainers_no_cache_no_prefetch.sh {} {} {} {} {} {} {} {}".format(
                i + 2,
                (len(private_ip_trainers) + 2),
                private_ip_oracle_cacher,
                i,
                len(private_ip_trainers),
                private_ip_trainers[0],
                log_file_name,
                num_iters,
            )
        }
        for i in range(len(private_ip_trainers))
    ]

    return run_args_trainers


def return_args_trainers_bagpipe_fae(
    private_ip_trainers, private_ip_oracle_cacher, log_file_name, num_iters
):
    """
    Arguments for trainers
    """

    run_args_trainers = [
        {
            "cmd": "rm -rf bagpipe && git clone git@github.com:iidsample/bagpipe.git && cd bagpipe && git checkout fae && bash run_fae_trainer.sh {} {} {} {} {} {} {} {}".format(
                i + 2,
                (len(private_ip_trainers) + 2),
                private_ip_oracle_cacher,
                i,
                len(private_ip_trainers),
                private_ip_trainers[0],
                log_file_name,
                num_iters,
            )
        }
        for i in range(len(private_ip_trainers))
    ]

    return run_args_trainers


def fcgnn_return_args_emb_server(
    private_ip_trainers, private_ip_oracle_cacher, emb_info_file
):
    """
    Return arguments for embedding server
    """
    run_args_emb_server = [
        {
            "cmd": "rm -rf bagpipe && git clone git@github.com:iidsample/bagpipe.git && cd bagpipe && git checkout add_fgcnn && bash run_embedding_server_no_hide.sh {} {} {} {}".format(
                1,
                (len(private_ip_trainers) + 2),
                private_ip_oracle_cacher,
                emb_info_file,
            )
        }
    ]

    return run_args_emb_server


def return_args_emb_server(
    private_ip_trainers, private_ip_oracle_cacher, emb_info_file
):
    """
    Return arguments for embedding server
    """
    run_args_emb_server = [
        {
            "cmd": "rm -rf bagpipe && git clone git@github.com:iidsample/bagpipe.git && cd bagpipe && bash run_embedding_server.sh {} {} {} {}".format(
                1,
                (len(private_ip_trainers) + 2),
                private_ip_oracle_cacher,
                emb_info_file,
            )
        }
    ]

    return run_args_emb_server


def return_args_emb_server_fae(private_ip_trainers, private_ip_oracle_cacher):
    """
    Return arguments for embedding server
    """
    run_args_emb_server = [
        {
            "cmd": "rm -rf bagpipe && git clone git@github.com:iidsample/bagpipe.git && cd bagpipe && git checkout fae && bash run_embedding_server.sh {} {} {}".format(
                1, (len(private_ip_trainers) + 2), private_ip_oracle_cacher
            )
        }
    ]

    return run_args_emb_server


def return_args_oracle_server(
    private_ip_trainers,
    private_ip_oracle_cacher,
):
    """
    Return arguments for oracle server
    """

    run_args_oracle_cacher = [
        {
            "cmd": "rm -rf bagpipe && git clone git@github.com:iidsample/bagpipe.git && cd bagpipe && bash run_oracle_server.sh {} {} {} {}".format(
                0,
                (len(private_ip_trainers) + 2),
                private_ip_oracle_cacher,
                len(private_ip_trainers),
            )
        }
    ]

    return run_args_oracle_cacher


def return_args_oracle_server_no_cache_no_prefetch(
    private_ip_trainers,
    private_ip_oracle_cacher,
):
    """
    Return arguments for oracle server
    """

    run_args_oracle_cacher = [
        {
            "cmd": "rm -rf bagpipe && git clone git@github.com:iidsample/bagpipe.git && cd bagpipe && bash run_oracle_server_no_cache_no_prefetch.sh {} {} {} {}".format(
                0,
                (len(private_ip_trainers) + 2),
                private_ip_oracle_cacher,
                len(private_ip_trainers),
            )
        }
    ]

    return run_args_oracle_cacher


def return_args_oracle_server_fae(
    private_ip_trainers,
    private_ip_oracle_cacher,
):
    """
    Return arguments for oracle server
    """

    run_args_oracle_cacher = [
        {
            "cmd": "rm -rf bagpipe && git clone git@github.com:iidsample/bagpipe.git && cd bagpipe && git checkout fae && bash run_fae_oracle_cacher.sh {} {} {} {}".format(
                0,
                (len(private_ip_trainers) + 2),
                private_ip_oracle_cacher,
                len(private_ip_trainers),
            )
        }
    ]

    return run_args_oracle_cacher


def return_args_original_dlrm_training(private_ip_trainers, log_file_name, stop_iter):
    """
    Returns arguments for original DLRM training
    """

    run_args_distributed = [
        {
            "cmd": "rm -rf dlrm_original && git clone git@github.com:iidsample/dlrm_original.git && cd dlrm_original && bash run_dlrm.sh {} {} {} {} {}".format(
                i,
                len(private_ip_trainers),
                private_ip_trainers[0],
                log_file_name,
                stop_iter,
            )
        }
        for i in range(len(private_ip_trainers))
    ]
    return run_args_distributed


def return_data_move_args_original(private_ip_trainers):
    run_args_move_files = [
        {
            "cmd": "aws s3 cp s3://recommendation-data-bagpipe/kaggle_criteo_info ./ && aws s3 cp s3://recommendation-data-bagpipe/kaggle_criteo_weekly.txt ./ && time wc -l  /home/ubuntu/kaggle_criteo_weekly.txt"
        }
        for i in range(len(private_ip_trainers))
    ]

    return run_args_move_files


def return_args_donwload_fae_kaggle_trainers(private_ip_trainers):
    run_args_download_files = [
        {
            "cmd": "aws s3 cp s3://recommendation-data-bagpipe/fae_hot_cold_bsize16384/hot_emb_dict.npz ./ && aws s3 cp  s3://recommendation-data-bagpipe/fae_hot_cold_bsize16384/train_hot.npz ./ &&  aws s3 cp s3://recommendation-data-bagpipe/fae_hot_cold_bsize16384/train_normal.npz ./"
        }
        for i in range(len(private_ip_trainers))
    ]
    return run_args_download_files


def return_args_donwload_fae_kaggle_oracle():
    run_args_download_files = [
        {
            "cmd": "aws s3 cp s3://recommendation-data-bagpipe/fae_hot_cold_bsize16384/hot_emb_dict.npz ./ && aws s3 cp  s3://recommendation-data-bagpipe/fae_hot_cold_bsize16384/train_hot.npz ./ && aws s3 cp s3://recommendation-data-bagpipe/fae_hot_cold_bsize16384/train_normal.npz ./"
        }
    ]

    return run_args_download_files


def return_args_download_avazu(private_ip_trainers):
    run_args_download_movielen = [
        {
            "cmd": "aws s3 cp s3://recommendation-data-bagpipe/avazu_subsampled.npz ./ && aws s3 cp s3://recommendation-data-bagpipe/avazu_emb_info ./"
        }
        for i in range(len(private_ip_trainers))
    ]

    return run_args_download_movielen


def return_args_download_movielen(private_ip_trainers):
    run_args_download_movielen = [
        {
            "cmd": "aws s3 cp s3://recommendation-data-bagpipe/movielen_train.npz ./ && aws s3 cp s3://recommendation-data-bagpipe/movielen_emb_info ./"
        }
        for i in range(len(private_ip_trainers))
    ]

    return run_args_download_movielen


def return_args_original_dlrm_training_movielens(
    private_ip_trainers, log_file_name, stop_iter
):
    """
    Returns arguments for original DLRM training
    """

    run_args_distributed = [
        {
            "cmd": "rm -rf dlrm_original && git clone git@github.com:iidsample/dlrm_original.git && cd dlrm_original && bash run_dlrm_other_datasets.sh {} {} {} {} {}".format(
                i,
                len(private_ip_trainers),
                private_ip_trainers[0],
                log_file_name,
                stop_iter,
            )
        }
        for i in range(len(private_ip_trainers))
    ]
    return run_args_distributed


def launch_instances_on_demand(launch_cfg):
    client = boto3.client("ec2", region_name=launch_cfg["region"])
    ec2 = boto3.resource("ec2", region_name=launch_cfg["region"])

    instance_lifecycle = launch_cfg["method"]
    instance_count = launch_cfg["instance_count"]

    if instance_lifecycle == "onDemand":
        print("in")
        response = client.run_instances(
            MaxCount=launch_cfg["instance_count"],
            MinCount=launch_cfg["instance_count"],
            ImageId=launch_cfg["ami_id"],
            InstanceType=launch_cfg["instance_type"],
            KeyName=launch_cfg["key_name"],
            EbsOptimized=True,
            IamInstanceProfile={"Name": launch_cfg["iam_role"]},
            # Placement={"AvailabilityZone": launch_cfg["az"]},
            # Placement={"GroupName": launch_cfg["GroupName"]},
            SecurityGroups=launch_cfg["security_group"],
        )
    else:
        print("Not a valid launch method")
        sys.exit()

    instance_ids = list()

    for request in response["Instances"]:
        instance_ids.append(request["InstanceId"])
    time.sleep(5)
    loop = True
    while loop:
        loop = False
        print("Instance ids {}".format(instance_ids))
        response = client.describe_instance_status(
            InstanceIds=instance_ids, IncludeAllInstances=True
        )
        # print("Response {}".format(response))
        for status in response["InstanceStatuses"]:
            print("Status {}".format(status["InstanceState"]["Name"]))
            if status["InstanceState"]["Name"] != "running":
                loop = True
                time.sleep(5)
    print("All instances are running ...")

    instance_collection = ec2.instances.filter(
        Filters=[{"Name": "instance-id", "Values": instance_ids}]
    )
    print("Instance collection {}".format(instance_collection))
    private_ip = []
    public_ip = []
    for instance in instance_collection:
        print(instance.private_ip_address)
        private_ip.append(instance.private_ip_address)
        print(instance.public_ip_address)
        public_ip.append(instance.public_ip_address)
    return (private_ip, public_ip, instance_ids)


def launch_instances_spot(launch_cfg):
    client = boto3.client("ec2", region_name=launch_cfg["region"])
    ec2 = boto3.resource("ec2", region_name=launch_cfg["region"])

    instance_lifecycle = launch_cfg["method"]
    instance_count = launch_cfg["instance_count"]
    launch_dict = {
        "KeyName": launch_cfg["key_name"],
        "ImageId": launch_cfg["ami_id"],
        "InstanceType": launch_cfg["instance_type"],
        "Placement": {"AvailabilityZone": launch_cfg["az"]},
        # "Placement": {"GroupName": launch_cfg["GroupName"]},
        "SecurityGroups": ["pytorch-distributed"],
        "IamInstanceProfile": {"Name": launch_cfg["iam_role"]},
    }

    if instance_lifecycle == "spot":
        response = client.request_spot_instances(
            InstanceCount=launch_cfg["instance_count"],
            LaunchSpecification=launch_dict,
            SpotPrice=launch_cfg["spot_price"],
        )
        print(response)
    else:
        print("Spot is not being used")
        sys.exit()

    request_ids = list()
    for request in response["SpotInstanceRequests"]:
        request_ids.append(request["SpotInstanceRequestId"])

    fulfilled_instances = list()
    loop = True

    print("Waiting for requests to fulfill")
    time.sleep(5)
    while loop:
        request = client.describe_spot_instance_requests(
            SpotInstanceRequestIds=request_ids
        )
        for req in request["SpotInstanceRequests"]:
            print(req)
            if req["State"] in ["closed", "cancelled", "failed"]:
                print("{}:{}".format(req["SpotInstanceRequestId"], req["State"]))
                loop = False
                break
            if "InstanceId" in req and req["InstanceId"]:
                fulfilled_instances.append(req["InstanceId"])
                print(req["InstanceId"] + "running...")
        if len(fulfilled_instances) == launch_cfg["instance_count"]:
            print("All requested instances are fulfilled")
            break
        time.sleep(5)
    if loop == False:
        print("Unable to fulfill all requested instance ..")
        sys.exit()

    while loop:
        loop = False
        response = client.describe_instance_status(InstanceIds=fulfilled_instances)
        for status in response["InstanceStatuses"]:
            if status["InstanceType"]["Name"] != "running":
                loop = True
    print("All instances are running ..")

    # getting host keys

    instance_collection = ec2.instances.filter(
        Filters=[{"Name": "instance-id", "Values": fulfilled_instances}]
    )
    private_ip = []
    public_ip = []
    for instance in instance_collection:
        print(instance.private_ip_address)
        private_ip.append(instance.private_ip_address)
        print(instance.public_ip_address)
        public_ip.append(instance.public_ip_address)
    return (private_ip, public_ip, fulfilled_instances)


def terminate_instances(instance_id, launch_cfg):
    print("Terminating instances ....")
    client = boto3.client("ec2", region_name=launch_cfg["region"])
    ec2 = boto3.resource("ec2", region_name=launch_cfg["region"])
    instance_collection = ec2.instances.filter(
        Filters=[{"Name": "instance-id", "Values": instance_id}]
    )
    for instance in instance_collection:
        instance.terminate()
    print("Bye Bye instances ...")


def get_az(instance_id, launch_cfg):

    client = boto3.client("ec2", region_name=launch_cfg["region"])
    ec2 = boto3.resource("ec2", region_name=launch_cfg["region"])
    response = client.describe_instance_status(
        InstanceIds=[instance_id], IncludeAllInstances=True
    )

    for status in response["InstanceStatuses"]:
        az_val = status["AvailabilityZone"]
        return az_val


def other_dataset_args_trainers_bagpipe_fae(
    private_ip_trainers,
    private_ip_oracle_cacher,
    log_file_name,
    num_iters,
    emb_info_file,
    hot_emb_dict,
):
    """
    Arguments for trainers
    """

    run_args_trainers = [
        {
            "cmd": "rm -rf bagpipe && git clone git@github.com:iidsample/bagpipe.git && cd bagpipe && git checkout fae && bash run_fae_trainer_other_datasets.sh {} {} {} {} {} {} {} {} {} {}".format(
                i + 2,
                (len(private_ip_trainers) + 2),
                private_ip_oracle_cacher,
                i,
                len(private_ip_trainers),
                private_ip_trainers[0],
                log_file_name,
                num_iters,
                emb_info_file,
                hot_emb_dict,
            )
        }
        for i in range(len(private_ip_trainers))
    ]

    return run_args_trainers


def other_dataset_args_emb_server_fae(
    private_ip_trainers, private_ip_oracle_cacher, emb_info_file
):
    """
    Return arguments for embedding server
    """
    run_args_emb_server = [
        {
            "cmd": "rm -rf bagpipe && git clone git@github.com:iidsample/bagpipe.git && cd bagpipe && git checkout fae && bash run_embedding_server.sh {} {} {}".format(
                1, (len(private_ip_trainers) + 2), private_ip_oracle_cacher
            )
        }
    ]

    return run_args_emb_server


def other_data_args_oracle_server_fae(
    private_ip_trainers,
    private_ip_oracle_cacher,
    processed_csv,
    train_normal_file,
    train_hot_file,
    train_hot_dict,
    emb_info_file,
):
    """
    Return arguments for oracle server
    """

    run_args_oracle_cacher = [
        {
            "cmd": "rm -rf bagpipe && git clone git@github.com:iidsample/bagpipe.git && cd bagpipe && git checkout fae && bash run_fae_oracle_cacher_other_datasets.sh {} {} {} {} {} {} {} {} {}".format(
                0,
                (len(private_ip_trainers) + 2),
                private_ip_oracle_cacher,
                len(private_ip_trainers),
                processed_csv,
                train_normal_file,
                train_hot_file,
                train_hot_dict,
                emb_info_file,
            )
        }
    ]

    return run_args_oracle_cacher


def other_data_args_emb_server_fae(
    private_ip_trainers, private_ip_oracle_cacher, emb_info_file
):
    """
    Return arguments for embedding server
    """
    run_args_emb_server = [
        {
            "cmd": "rm -rf bagpipe && git clone git@github.com:iidsample/bagpipe.git && cd bagpipe && git checkout fae && bash run_embedding_server_other_datasets.sh {} {} {} {}".format(
                1,
                (len(private_ip_trainers) + 2),
                private_ip_oracle_cacher,
                emb_info_file,
            )
        }
    ]

    return run_args_emb_server


run_args_ebs_warmnup = [
    {
        "cmd": "aws s3 cp s3://recommendation-data-bagpipe/kaggle_criteo_info ./ && aws s3 cp s3://recommendation-data-bagpipe/kaggle_criteo_weekly.txt ./ && time wc -l  /home/ubuntu/kaggle_criteo_weekly.txt"
    }
]


def run_large_scale():

    launch_cfg = {
        "name": "recommendation-setup",
        "key_name": "saurabh_oregon_pc",
        "key_path": "/home/saurabh/credentials/cs-shivaram/saurabh_oregon_pc.pem",
        "region": "us-west-2",
        "method": "onDemand",  # onDemand
        "az": "us-west-2c",
        "GroupName": "distributed-training",
        # "ami_id": "ami-0f07487e2b2761b0a", # nv old
        # "ami_id": "ami-04e4121bc8f056792", # oregon old
        "ami_id": "ami-00cfdc3a2d9df3424",
        "ssh_username": "ubuntu",
        "iam_role": "ec2-s3-final",
        "instance_type": "p3.2xlarge",
        # "instance_type": "t2.medium",
        "instance_count": 2,
        "spot_price": "4.5",
        "security_group": ["pytorch-distributed"],
    }

    # launching trainers
    launch_cfg["instance_type"] = "p3.2xlarge"
    launch_cfg["method"] = "onDemand"
    launch_cfg["instance_count"] = 8
    start_time = time.time()
    (
        private_ip_trainers,
        public_ip_trainers,
        instance_ids_trainers,
    ) = launch_instances_on_demand(launch_cfg)
    print("Time to launch {}".format(time.time() - start_time))
    # launching  oracle cacher
    p3_az = get_az(instance_ids_trainers[0], launch_cfg)
    launch_cfg["instance_type"] = "c5.18xlarge"
    launch_cfg["spot_price"] = "2.5"
    launch_cfg["method"] = "onDemand"
    launch_cfg["instance_count"] = 1
    launch_cfg["az"] = p3_az

    private_ips, public_ips, instance_ids = launch_instances_on_demand(launch_cfg)

    private_ip_oracle_cacher = private_ips[0]
    public_ip_oracle_cacher = public_ips[0]
    instance_id_oracle_cacher = instance_ids[0]

    # launch emb server
    launch_cfg["instance_type"] = "c5.18xlarge"
    launch_cfg["spot_price"] = "2.5"
    launch_cfg["method"] = "onDemand"
    launch_cfg["instance_count"] = 1
    launch_cfg["az"] = p3_az

    # launched emb server
    private_ips, public_ips, instance_ids = launch_instances_on_demand(launch_cfg)

    private_ip_emb_server = private_ips[0]
    public_ip_emb_server = public_ips[0]
    instance_id_emb_server = instance_ids[0]

    # client oracle cache
    client_oracle_cacher = ParallelSSHClient(
        [public_ip_oracle_cacher], user="ubuntu", pkey=launch_cfg["key_path"]
    )
    # trainer client
    client_trainers = ParallelSSHClient(
        public_ip_trainers, user="ubuntu", pkey=launch_cfg["key_path"]
    )
    # client for emb server
    client_emb_server = ParallelSSHClient(
        [public_ip_emb_server], user="ubuntu", pkey=launch_cfg["key_path"]
    )

    if False:
        # =================Run bagpipe fcgnn =========================
        batch_size = 8192
        log_file_name = "fcgnn_criteo_kaggle_{}_{}_machines_2000_iter_run1".format(
            batch_size, len(private_ip_trainers)
        )

        emb_info_file = "/home/ubuntu/emb_table_info.txt"
        parsed_data_file = "/home/ubuntu/parsed_train.txt"

        download_single = data_download_fcgnn_single()
        download_trainers = data_download_fcgnn(private_ip_trainers)

        run_args_oracle_cacher = fcgnn_args_oracle_server(
            private_ip_trainers,
            private_ip_oracle_cacher,
            parsed_data_file,
            emb_info_file,
            batch_size,
        )

        run_args_trainers = fcgnn_other_data_args_trainer(
            private_ip_trainers,
            private_ip_oracle_cacher,
            log_file_name,
            2000,
            emb_info_file,
        )

        run_args_emb_server = fcgnn_return_args_emb_server(
            private_ip_trainers, private_ip_oracle_cacher, emb_info_file
        )

        output_download_tr = client_trainers.run_command(
            "%(cmd)s", host_args=download_trainers
        )

        output_download_single_mach = client_emb_server.run_command(
            "%(cmd)s", host_args=download_single
        )

        output_download_single_mach = client_oracle_cacher.run_command(
            "%(cmd)s", host_args=download_single
        )

        for hosts_out in output_download_tr:
            for line in hosts_out.stdout:
                print(line)
        print("Run args trainers {}".format(run_args_trainers))
        print("Run args emb server {}".format(run_args_emb_server))
        print("Runs args oracle {}".format(run_args_oracle_cacher))

        trainer_output = client_trainers.run_command(
            "%(cmd)s", host_args=run_args_trainers
        )
        emb_server_out = client_emb_server.run_command(
            "%(cmd)s", host_args=run_args_emb_server
        )
        oracle_server_out = client_oracle_cacher.run_command(
            "%(cmd)s", host_args=run_args_oracle_cacher
        )

        for hosts_out in trainer_output:
            for line in hosts_out.stdout:
                print(line)

        run_args_kill_trainers = [
            {"cmd": "pkill -9 python"} for i in range(len(private_ip_trainers))
        ]
        run_args_kill_oracle = [{"cmd": "pkill -9 python"}]
        run_args_kill_emb_server = [{"cmd": "pkill -9 python"}]

        kill_trainers = client_trainers.run_command(
            "%(cmd)s", host_args=run_args_kill_trainers
        )

        kill_emb_server = client_emb_server.run_command(
            "%(cmd)s", host_args=run_args_kill_emb_server
        )

        kill_oracle_cacher = client_oracle_cacher.run_command(
            "%(cmd)s", host_args=run_args_kill_oracle
        )
        print("Launched python kill command")
        time.sleep(60)

    if False:
        # =================Run bagpipe fcgnn 2nd time =========================
        batch_size = 8192
        log_file_name = "fcgnn_criteo_kaggle_{}_{}_machines_2000_iter_run2".format(
            batch_size, len(private_ip_trainers)
        )

        emb_info_file = "/home/ubuntu/emb_table_info.txt"
        parsed_data_file = "/home/ubuntu/parsed_train.txt"

        download_single = data_download_fcgnn_single()
        download_trainers = data_download_fcgnn(private_ip_trainers)

        run_args_oracle_cacher = fcgnn_args_oracle_server(
            private_ip_trainers,
            private_ip_oracle_cacher,
            parsed_data_file,
            emb_info_file,
            batch_size,
        )

        run_args_trainers = fcgnn_other_data_args_trainer(
            private_ip_trainers,
            private_ip_oracle_cacher,
            log_file_name,
            2000,
            emb_info_file,
        )

        run_args_emb_server = fcgnn_return_args_emb_server(
            private_ip_trainers, private_ip_oracle_cacher, emb_info_file
        )

        output_download_tr = client_trainers.run_command(
            "%(cmd)s", host_args=download_trainers
        )

        output_download_single_mach = client_emb_server.run_command(
            "%(cmd)s", host_args=download_single
        )

        output_download_single_mach = client_oracle_cacher.run_command(
            "%(cmd)s", host_args=download_single
        )

        for hosts_out in output_download_tr:
            for line in hosts_out.stdout:
                print(line)
        print("Run args trainers {}".format(run_args_trainers))
        print("Run args emb server {}".format(run_args_emb_server))
        print("Runs args oracle {}".format(run_args_oracle_cacher))

        trainer_output = client_trainers.run_command(
            "%(cmd)s", host_args=run_args_trainers
        )
        emb_server_out = client_emb_server.run_command(
            "%(cmd)s", host_args=run_args_emb_server
        )
        oracle_server_out = client_oracle_cacher.run_command(
            "%(cmd)s", host_args=run_args_oracle_cacher
        )

        for hosts_out in trainer_output:
            for line in hosts_out.stdout:
                print(line)

        run_args_kill_trainers = [
            {"cmd": "pkill -9 python"} for i in range(len(private_ip_trainers))
        ]
        run_args_kill_oracle = [{"cmd": "pkill -9 python"}]
        run_args_kill_emb_server = [{"cmd": "pkill -9 python"}]

        kill_trainers = client_trainers.run_command(
            "%(cmd)s", host_args=run_args_kill_trainers
        )

        kill_emb_server = client_emb_server.run_command(
            "%(cmd)s", host_args=run_args_kill_emb_server
        )

        kill_oracle_cacher = client_oracle_cacher.run_command(
            "%(cmd)s", host_args=run_args_kill_oracle
        )
        print("Launched python kill command")
        time.sleep(60)
    if True:
        # =================Run bagpipe fcgnn no cache no prefetch =========================

        print("Running bagpipe")
        batch_size = 8192
        log_file_name = (
            "fcgnn_criteo_kaggle_{}_{}_machines_2000_iter_no_cache_no_pref".format(
                batch_size, len(private_ip_trainers)
            )
        )

        emb_info_file = "/home/ubuntu/emb_table_info.txt"
        parsed_data_file = "/home/ubuntu/parsed_train.txt"

        download_single = data_download_fcgnn_single()
        download_trainers = data_download_fcgnn(private_ip_trainers)

        run_args_oracle_cacher = fcgnn_args_oracle_server_no_cache_no_pref(
            private_ip_trainers,
            private_ip_oracle_cacher,
            parsed_data_file,
            emb_info_file,
            batch_size,
        )

        run_args_trainers = fcgnn_other_data_args_trainer_no_cache_no_pref(
            private_ip_trainers,
            private_ip_oracle_cacher,
            log_file_name,
            2000,
            emb_info_file,
        )

        run_args_emb_server = fcgnn_return_args_emb_server(
            private_ip_trainers, private_ip_oracle_cacher, emb_info_file
        )

        output_download_tr = client_trainers.run_command(
            "%(cmd)s", host_args=download_trainers
        )

        output_download_single_mach = client_emb_server.run_command(
            "%(cmd)s", host_args=download_single
        )

        output_download_single_mach = client_oracle_cacher.run_command(
            "%(cmd)s", host_args=download_single
        )

        for hosts_out in output_download_tr:
            for line in hosts_out.stdout:
                print(line)
        print("Run args trainers {}".format(run_args_trainers))
        print("Run args emb server {}".format(run_args_emb_server))
        print("Runs args oracle {}".format(run_args_oracle_cacher))

        trainer_output = client_trainers.run_command(
            "%(cmd)s", host_args=run_args_trainers
        )
        emb_server_out = client_emb_server.run_command(
            "%(cmd)s", host_args=run_args_emb_server
        )
        oracle_server_out = client_oracle_cacher.run_command(
            "%(cmd)s", host_args=run_args_oracle_cacher
        )

        for hosts_out in oracle_server_out:
            for line in hosts_out.stdout:
                print(line)

        run_args_kill_trainers = [
            {"cmd": "pkill -9 python"} for i in range(len(private_ip_trainers))
        ]
        run_args_kill_oracle = [{"cmd": "pkill -9 python"}]
        run_args_kill_emb_server = [{"cmd": "pkill -9 python"}]

        kill_trainers = client_trainers.run_command(
            "%(cmd)s", host_args=run_args_kill_trainers
        )

        kill_emb_server = client_emb_server.run_command(
            "%(cmd)s", host_args=run_args_kill_emb_server
        )

        kill_oracle_cacher = client_oracle_cacher.run_command(
            "%(cmd)s", host_args=run_args_kill_oracle
        )
        print("Launched python kill command")
        time.sleep(60)
    if True:

        # ======= Run bagpipe with no embedding overlap ===================
        batch_size = 8192
        log_file_name = (
            "fcgnn_criteo_kaggle_{}_{}_machines_2000_iter_emb_no_overlap".format(
                batch_size, len(private_ip_trainers)
            )
        )

        emb_info_file = "/home/ubuntu/emb_table_info.txt"
        parsed_data_file = "/home/ubuntu/parsed_train.txt"

        download_single = data_download_fcgnn_single()
        download_trainers = data_download_fcgnn(private_ip_trainers)

        run_args_oracle_cacher = fcgnn_args_oracle_server(
            private_ip_trainers,
            private_ip_oracle_cacher,
            parsed_data_file,
            emb_info_file,
            batch_size,
        )

        run_args_trainers = fcgnn_other_data_args_trainer_no_hide(
            private_ip_trainers,
            private_ip_oracle_cacher,
            log_file_name,
            2000,
            emb_info_file,
        )

        run_args_emb_server = fcgnn_return_args_emb_server(
            private_ip_trainers, private_ip_oracle_cacher, emb_info_file
        )

        output_download_tr = client_trainers.run_command(
            "%(cmd)s", host_args=download_trainers
        )

        output_download_single_mach = client_emb_server.run_command(
            "%(cmd)s", host_args=download_single
        )

        output_download_single_mach = client_oracle_cacher.run_command(
            "%(cmd)s", host_args=download_single
        )

        for hosts_out in output_download_tr:
            for line in hosts_out.stdout:
                print(line)
        print("Run args trainers {}".format(run_args_trainers))
        print("Run args emb server {}".format(run_args_emb_server))
        print("Runs args oracle {}".format(run_args_oracle_cacher))

        trainer_output = client_trainers.run_command(
            "%(cmd)s", host_args=run_args_trainers
        )
        emb_server_out = client_emb_server.run_command(
            "%(cmd)s", host_args=run_args_emb_server
        )
        oracle_server_out = client_oracle_cacher.run_command(
            "%(cmd)s", host_args=run_args_oracle_cacher
        )

        for hosts_out in oracle_server_out:
            for line in hosts_out.stdout:
                print(line)

        run_args_kill_trainers = [
            {"cmd": "pkill -9 python"} for i in range(len(private_ip_trainers))
        ]
        run_args_kill_oracle = [{"cmd": "pkill -9 python"}]
        run_args_kill_emb_server = [{"cmd": "pkill -9 python"}]

        kill_trainers = client_trainers.run_command(
            "%(cmd)s", host_args=run_args_kill_trainers
        )

        kill_emb_server = client_emb_server.run_command(
            "%(cmd)s", host_args=run_args_kill_emb_server
        )

        kill_oracle_cacher = client_oracle_cacher.run_command(
            "%(cmd)s", host_args=run_args_kill_oracle
        )
        print("Launched python kill command")
        time.sleep(60)

    if False:
        # =================Run bagpipe avazu fae =====================
        log_file_name = "avazu_bagpip_fae_prefetch_16384_{}_machines_1500_iter".format(
            len(private_ip_trainers)
        )

        emb_info_file = "/home/ubuntu/avazu_emb_info"

        avazu_processed_csv = "/home/ubuntu/avazu_data"
        avazu_normal_csv = "/home/ubuntu/avazu_normal.npz"
        avazu_hot_csv = "/home/ubuntu/avazu_hot.npz"
        avazu_hot_dict = "/home/ubuntu/avazu_hot_emb_dict.npz"

        run_args_trainers = other_dataset_args_trainers_bagpipe_fae(
            private_ip_trainers,
            private_ip_oracle_cacher,
            log_file_name,
            1500,
            emb_info_file,
            avazu_hot_dict,
        )
        run_args_oracle_cacher = other_data_args_oracle_server_fae(
            private_ip_trainers,
            private_ip_oracle_cacher,
            avazu_processed_csv,
            avazu_normal_csv,
            avazu_hot_csv,
            avazu_hot_dict,
            emb_info_file,
        )

        run_args_emb_server = other_data_args_emb_server_fae(
            private_ip_trainers, private_ip_oracle_cacher, emb_info_file
        )

        download_single = data_download_avazu_single_machine_fae()
        download_trainers = data_download_avazu_trainer_machine_fae(private_ip_trainers)

        print("Download single {}".format(download_single))
        print("Download trainers {}".format(download_trainers))

        output_download_tr = client_trainers.run_command(
            "%(cmd)s", host_args=download_trainers
        )

        output_download_single_mach = client_emb_server.run_command(
            "%(cmd)s", host_args=download_single
        )

        output_download_single_mach = client_oracle_cacher.run_command(
            "%(cmd)s", host_args=download_single
        )

        for hosts_out in output_download_tr:
            for line in hosts_out.stdout:
                print(line)
        print("Run args trainers {}".format(run_args_trainers))
        print("Run args emb server {}".format(run_args_emb_server))
        print("Runs args oracle {}".format(run_args_oracle_cacher))

        trainer_output = client_trainers.run_command(
            "%(cmd)s", host_args=run_args_trainers
        )
        emb_server_out = client_emb_server.run_command(
            "%(cmd)s", host_args=run_args_emb_server
        )
        oracle_server_out = client_oracle_cacher.run_command(
            "%(cmd)s", host_args=run_args_oracle_cacher
        )

        for hosts_out in oracle_server_out:
            for line in hosts_out.stdout:
                print(line)

    if False:
        # =========Run bagpipe avazu cache and prefetch ============

        log_file_name = "rerun_1_avazu_bagpipe_prefetch_hide_cache_sync_16384_{}_machines_2000_iter".format(
            len(private_ip_trainers)
        )
        emb_info_file = "/home/ubuntu/avazu_emb_info"
        run_args_emb_server = other_data_args_emb_server(
            private_ip_trainers, private_ip_oracle_cacher, emb_info_file
        )
        csv_location = "/home/ubuntu/avazu_data"
        run_args_oracle = other_data_args_oracle_server(
            private_ip_trainers, private_ip_oracle_cacher, csv_location, emb_info_file
        )

        run_args_trainers = other_data_args_trainer(
            private_ip_trainers,
            private_ip_oracle_cacher,
            log_file_name,
            2000,
            emb_info_file,
        )
        download_trainers = data_download_avazu_trainer_machine(private_ip_trainers)
        download_single_machine = data_download_avazu_single_machine()

        output_download_tr = client_trainers.run_command(
            "%(cmd)s", host_args=download_trainers
        )

        output_download_single_mach = client_emb_server.run_command(
            "%(cmd)s", host_args=download_single_machine
        )

        output_download_single_mach = client_oracle_cacher.run_command(
            "%(cmd)s", host_args=download_single_machine
        )

        for hosts_out in output_download_tr:
            for line in hosts_out.stdout:
                print(line)
        print("Run args trainers {}".format(run_args_trainers))
        print("Run args emb server {}".format(run_args_emb_server))
        print("Runs args oracle {}".format(run_args_oracle))
        trainer_output = client_trainers.run_command(
            "%(cmd)s", host_args=run_args_trainers
        )
        emb_server_out = client_emb_server.run_command(
            "%(cmd)s", host_args=run_args_emb_server
        )
        oracle_server_out = client_oracle_cacher.run_command(
            "%(cmd)s", host_args=run_args_oracle
        )

        for hosts_out in trainer_output:
            for line in hosts_out.stdout:
                print(line)
        kill_process(
            client_trainers,
            client_emb_server,
            client_oracle_cacher,
            private_ip_trainers,
        )

    if False:
        # =========Run bagpipe avazu cache and prefetch no hiding cache sync ============

        log_file_name = "rerun_1_avazu_bagpipe_prefetch_not_hide_cache_sync_16384_{}_machines_2000_iter".format(
            len(private_ip_trainers)
        )
        emb_info_file = "/home/ubuntu/avazu_emb_info"
        run_args_emb_server = other_data_args_emb_server(
            private_ip_trainers, private_ip_oracle_cacher, emb_info_file
        )
        csv_location = "/home/ubuntu/avazu_data"
        run_args_oracle = other_data_args_oracle_server(
            private_ip_trainers, private_ip_oracle_cacher, csv_location, emb_info_file
        )

        run_args_trainers = other_data_args_trainer_no_hide_cache_sync(
            private_ip_trainers,
            private_ip_oracle_cacher,
            log_file_name,
            2000,
            emb_info_file,
        )
        download_trainers = data_download_avazu_trainer_machine(private_ip_trainers)
        download_single_machine = data_download_avazu_single_machine()

        output_download_tr = client_trainers.run_command(
            "%(cmd)s", host_args=download_trainers
        )

        output_download_single_mach = client_emb_server.run_command(
            "%(cmd)s", host_args=download_single_machine
        )

        output_download_single_mach = client_oracle_cacher.run_command(
            "%(cmd)s", host_args=download_single_machine
        )

        for hosts_out in output_download_tr:
            for line in hosts_out.stdout:
                print(line)
        print("Run args trainers {}".format(run_args_trainers))
        print("Run args emb server {}".format(run_args_emb_server))
        print("Runs args oracle {}".format(run_args_oracle))
        trainer_output = client_trainers.run_command(
            "%(cmd)s", host_args=run_args_trainers
        )
        emb_server_out = client_emb_server.run_command(
            "%(cmd)s", host_args=run_args_emb_server
        )
        oracle_server_out = client_oracle_cacher.run_command(
            "%(cmd)s", host_args=run_args_oracle
        )

        for hosts_out in trainer_output:
            for line in hosts_out.stdout:
                print(line)
        kill_process(
            client_trainers,
            client_emb_server,
            client_oracle_cacher,
            private_ip_trainers,
        )

    if False:
        # ========Run bagpipe avazu no cache no prefetch ===========
        log_file_name = (
            "avazu_bagpipe_no_cache_no_prefetch_16384_{}_machines_2000_iter".format(
                len(private_ip_trainers)
            )
        )

        emb_info_file = "/home/ubuntu/avazu_emb_info"
        run_args_emb_server = other_data_args_emb_server(
            private_ip_trainers, private_ip_oracle_cacher, emb_info_file
        )

        csv_location = "/home/ubuntu/avazu_data"
        run_args_oracle = other_data_args_oracle_server_no_cache_no_prefetch(
            private_ip_trainers, private_ip_oracle_cacher, csv_location, emb_info_file
        )

        run_args_trainers = other_data_args_trainer_no_cache_no_prefetch(
            private_ip_trainers,
            private_ip_oracle_cacher,
            log_file_name,
            2000,
            emb_info_file,
        )

        download_trainers = data_download_avazu_trainer_machine(private_ip_trainers)
        download_single_machine = data_download_avazu_single_machine()

        output_download_tr = client_trainers.run_command(
            "%(cmd)s", host_args=download_trainers
        )

        output_download_single_mach = client_emb_server.run_command(
            "%(cmd)s", host_args=download_single_machine
        )

        output_download_single_mach = client_oracle_cacher.run_command(
            "%(cmd)s", host_args=download_single_machine
        )

        for hosts_out in output_download_tr:
            for line in hosts_out.stdout:
                print(line)
        trainer_output = client_trainers.run_command(
            "%(cmd)s", host_args=run_args_trainers
        )
        emb_server_out = client_emb_server.run_command(
            "%(cmd)s", host_args=run_args_emb_server
        )
        oracle_server_out = client_oracle_cacher.run_command(
            "%(cmd)s", host_args=run_args_oracle
        )

        for hosts_out in trainer_output:
            for line in hosts_out.stdout:
                print(line)

        kill_process(
            client_trainers,
            client_emb_server,
            client_oracle_cacher,
            private_ip_trainers,
        )

    if False:
        # ============================Run Bagpipe movie lens no cache no prefetch ===============
        log_file_name = (
            "movielens_bagpipe_no_cache_no_prefetch_16384_{}_machines_1000_iter".format(
                len(private_ip_trainers)
            )
        )

        emb_info_file = "/home/ubuntu/movielen_emb_info"
        run_args_emb_server = other_data_args_emb_server(
            private_ip_trainers, private_ip_oracle_cacher, emb_info_file
        )

        csv_location = "/home/ubuntu/movielen_data"
        run_args_oracle = other_data_args_oracle_server_no_cache_no_prefetch(
            private_ip_trainers, private_ip_oracle_cacher, csv_location, emb_info_file
        )

        run_args_trainers = other_data_args_trainer_no_cache_no_prefetch(
            private_ip_trainers,
            private_ip_oracle_cacher,
            log_file_name,
            1000,
            emb_info_file,
        )

        download_trainers = data_download_movie_lens_trainer_machine(
            private_ip_trainers
        )
        download_single_machine = data_download_movie_lens_single_machine(
            private_ip_emb_server
        )

        output_download_tr = client_trainers.run_command(
            "%(cmd)s", host_args=download_trainers
        )

        output_download_single_mach = client_emb_server.run_command(
            "%(cmd)s", host_args=download_single_machine
        )

        output_download_single_mach = client_oracle_cacher.run_command(
            "%(cmd)s", host_args=download_single_machine
        )

        for hosts_out in output_download_tr:
            for line in hosts_out.stdout:
                print(line)
        trainer_output = client_trainers.run_command(
            "%(cmd)s", host_args=run_args_trainers
        )
        emb_server_out = client_emb_server.run_command(
            "%(cmd)s", host_args=run_args_emb_server
        )
        oracle_server_out = client_oracle_cacher.run_command(
            "%(cmd)s", host_args=run_args_oracle_
        )

        for hosts_out in trainer_output:
            for line in hosts_out.stdout:
                print(line)
    if False:
        # ======================Run avazu CL ==================
        log_file_name = "avazu_hybrid_cpu_gpu_final_run_batch_size_16384_{}_machine_original_dlrm_2000_iters.log".format(
            len(private_ip_trainers)
        )
        download_data = return_args_download_avazu(private_ip_trainers)
        print("Download data {}".format(download_data))
        output_trainers = client_trainers.run_command(
            "%(cmd)s", host_args=download_data
        )

        for hosts_out in output_trainers:
            for line in hosts_out.stdout:
                print(line)

        run_args_trainers = return_args_original_dlrm_training_movielens(
            private_ip_trainers, log_file_name, 2000
        )
        print("Train args {}".format(run_args_trainers))
        output_trainers = client_trainers.run_command(
            "%(cmd)s", host_args=run_args_trainers
        )

        client_trainers.join(consume_output=True)
        for hosts_out in output_trainers:
            for line in hosts_out.stdout:
                print(line)
        print("Done avazu")
    if False:
        # ======================Run movie lens==================
        log_file_name = "hybrid_cpu_gpu_final_run_batch_size_16384_{}_machine_original_dlrm_1000_iters.log".format(
            len(private_ip_trainers)
        )
        download_data = return_args_download_movielen(private_ip_trainers)
        print("Download data {}".format(download_data))
        output_trainers = client_trainers.run_command(
            "%(cmd)s", host_args=download_data
        )

        for hosts_out in output_trainers:
            for line in hosts_out.stdout:
                print(line)

        run_args_trainers = return_args_original_dlrm_training_movielens(
            private_ip_trainers, log_file_name, 1000
        )
        print("Train args {}".format(run_args_trainers))
        output_trainers = client_trainers.run_command(
            "%(cmd)s", host_args=run_args_trainers
        )

        client_trainers.join(consume_output=True)
        for hosts_out in output_trainers:
            for line in hosts_out.stdout:
                print(line)
        print("Done movielens")


if __name__ == "__main__":
    run_large_scale()
