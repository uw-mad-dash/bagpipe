import sys
import boto3
import time

from pssh.clients import ParallelSSHClient


def return_args_trainers_dlrm(
    private_ip_trainers, private_ip_oracle_cacher, log_file_name, num_iters, cache_size
):
    """
    Arguments for trainers
    """

    run_args_trainers = [
        {
            "cmd": "rm -rf bagpipe && git clone git@github.com:iidsample/bagpipe.git && cd bagpipe && bash run_trainers_dlrm.sh {} {} {} {} {} {} {} {} {}".format(
                i + 2,
                (len(private_ip_trainers) + 2),
                private_ip_oracle_cacher,
                i,
                len(private_ip_trainers),
                private_ip_trainers[0],
                log_file_name,
                num_iters,
                cache_size,
            )
        }
        for i in range(len(private_ip_trainers))
    ]

    return run_args_trainers


def return_args_trainers_wdn(
    private_ip_trainers, private_ip_oracle_cacher, log_file_name, num_iters, cache_size
):
    """
    Arguments for trainers
    """

    run_args_trainers = [
        {
            "cmd": "rm -rf bagpipe && git clone git@github.com:iidsample/bagpipe.git && cd bagpipe && bash run_trainers_wdn.sh {} {} {} {} {} {} {} {} {}".format(
                i + 2,
                (len(private_ip_trainers) + 2),
                private_ip_oracle_cacher,
                i,
                len(private_ip_trainers),
                private_ip_trainers[0],
                log_file_name,
                num_iters,
                cache_size,
            )
        }
        for i in range(len(private_ip_trainers))
    ]

    return run_args_trainers


def return_args_trainers_dcn(
    private_ip_trainers, private_ip_oracle_cacher, log_file_name, num_iters, cache_size
):
    """
    Arguments for trainers
    """

    run_args_trainers = [
        {
            "cmd": "rm -rf bagpipe && git clone git@github.com:iidsample/bagpipe.git && cd bagpipe && bash run_trainers_dcn.sh {} {} {} {} {} {} {} {} {}".format(
                i + 2,
                (len(private_ip_trainers) + 2),
                private_ip_oracle_cacher,
                i,
                len(private_ip_trainers),
                private_ip_trainers[0],
                log_file_name,
                num_iters,
                cache_size,
            )
        }
        for i in range(len(private_ip_trainers))
    ]

    return run_args_trainers


def return_args_trainers_dfm(
    private_ip_trainers, private_ip_oracle_cacher, log_file_name, num_iters, cache_size
):
    """
    Arguments for trainers
    """

    run_args_trainers = [
        {
            "cmd": "rm -rf bagpipe && git clone git@github.com:iidsample/bagpipe.git && cd bagpipe && bash run_trainers_dfm.sh {} {} {} {} {} {} {} {} {}".format(
                i + 2,
                (len(private_ip_trainers) + 2),
                private_ip_oracle_cacher,
                i,
                len(private_ip_trainers),
                private_ip_trainers[0],
                log_file_name,
                num_iters,
                cache_size,
            )
        }
        for i in range(len(private_ip_trainers))
    ]

    return run_args_trainers


def return_args_emb_server_dlrm(private_ip_trainers, private_ip_oracle_cacher):
    """
    Return arguments for embedding server
    """
    run_args_emb_server = [
        {
            "cmd": "rm -rf bagpipe && git clone git@github.com:iidsample/bagpipe.git && cd bagpipe && bash run_embedding_server_dlrm.sh {} {} {}".format(
                1, (len(private_ip_trainers) + 2), private_ip_oracle_cacher
            )
        }
    ]

    return run_args_emb_server


def return_args_emb_server_wdn(private_ip_trainers, private_ip_oracle_cacher):
    """
    Return arguments for embedding server
    """
    run_args_emb_server = [
        {
            "cmd": "rm -rf bagpipe && git clone git@github.com:iidsample/bagpipe.git && cd bagpipe && bash run_embedding_server_wdn.sh {} {} {}".format(
                1, (len(private_ip_trainers) + 2), private_ip_oracle_cacher
            )
        }
    ]

    return run_args_emb_server


def return_args_oracle_server_dlrm(
    private_ip_trainers, private_ip_oracle_cacher, batch_size, cache_size
):
    """
    Return arguments for oracle server
    """

    run_args_oracle_cacher = [
        {
            "cmd": "rm -rf bagpipe && git clone git@github.com:iidsample/bagpipe.git && cd bagpipe && bash run_oracle_server_dlrm.sh {} {} {} {} {} {}".format(
                0,
                (len(private_ip_trainers) + 2),
                private_ip_oracle_cacher,
                len(private_ip_trainers),
                batch_size,
                cache_size,
            )
        }
    ]

    return run_args_oracle_cacher


def return_args_trainers_fgcnn(
    private_ip_trainers, private_ip_oracle_cacher, log_file_name, num_iters, cache_size
):
    """
    Arguments for trainers
    """

    run_args_trainers = [
        {
            "cmd": "rm -rf bagpipe && git clone git@github.com:iidsample/bagpipe.git && cd bagpipe && bash run_trainers_fgcnn.sh {} {} {} {} {} {} {} {} {}".format(
                i + 2,
                (len(private_ip_trainers) + 2),
                private_ip_oracle_cacher,
                i,
                len(private_ip_trainers),
                private_ip_trainers[0],
                log_file_name,
                num_iters,
                cache_size,
            )
        }
        for i in range(len(private_ip_trainers))
    ]

    return run_args_trainers


def return_args_emb_server_fgcnn(private_ip_trainers, private_ip_oracle_cacher):
    """
    Return arguments for embedding server
    """
    run_args_emb_server = [
        {
            "cmd": "rm -rf bagpipe && git clone git@github.com:iidsample/bagpipe.git && cd bagpipe && bash run_embedding_server_fgcnn.sh {} {} {}".format(
                1, (len(private_ip_trainers) + 2), private_ip_oracle_cacher
            )
        }
    ]

    return run_args_emb_server


def return_args_oracle_server_fgcnn(
    private_ip_trainers, private_ip_oracle_cacher, batch_size, cache_size
):
    """
    Return arguments for oracle server
    """

    run_args_oracle_cacher = [
        {
            "cmd": "rm -rf bagpipe && git clone git@github.com:iidsample/bagpipe.git && cd bagpipe && bash run_oracle_server_fgcnn.sh {} {} {} {} {} {}".format(
                0,
                (len(private_ip_trainers) + 2),
                private_ip_oracle_cacher,
                len(private_ip_trainers),
                batch_size,
                cache_size,
            )
        }
    ]

    return run_args_oracle_cacher


def return_args_trainers_caser(
    private_ip_trainers, private_ip_oracle_cacher, log_file_name, num_iters, cache_size
):
    """
    Arguments for trainers
    """

    run_args_trainers = [
        {
            "cmd": "rm -rf bagpipe && git clone git@github.com:iidsample/bagpipe.git && cd bagpipe && bash run_trainers_caser.sh {} {} {} {} {} {} {} {} {}".format(
                i + 2,
                (len(private_ip_trainers) + 2),
                private_ip_oracle_cacher,
                i,
                len(private_ip_trainers),
                private_ip_trainers[0],
                log_file_name,
                num_iters,
                cache_size,
            )
        }
        for i in range(len(private_ip_trainers))
    ]

    return run_args_trainers


def return_args_emb_server_caser(private_ip_trainers, private_ip_oracle_cacher):
    """
    Return arguments for embedding server
    """
    run_args_emb_server = [
        {
            "cmd": "rm -rf bagpipe && git clone git@github.com:iidsample/bagpipe.git && cd bagpipe && bash run_embedding_server_caser.sh {} {} {}".format(
                1, (len(private_ip_trainers) + 2), private_ip_oracle_cacher
            )
        }
    ]

    return run_args_emb_server


def return_args_oracle_server_caser(
    private_ip_trainers, private_ip_oracle_cacher, batch_size, cache_size
):
    """
    Return arguments for oracle server
    """

    run_args_oracle_cacher = [
        {
            "cmd": "rm -rf bagpipe && git clone git@github.com:iidsample/bagpipe.git && cd bagpipe && bash run_oracle_server_caser.sh {} {} {} {} {} {}".format(
                0,
                (len(private_ip_trainers) + 2),
                private_ip_oracle_cacher,
                len(private_ip_trainers),
                batch_size,
                cache_size,
            )
        }
    ]

    return run_args_oracle_cacher


def return_data_move_args_original(private_ip_trainers):
    run_args_move_files = [
        {
            "cmd": "aws s3 cp s3://recommendation-data-bagpipe/kaggle_criteo_info ./ && aws s3 cp s3://recommendation-data-bagpipe/emb_table_info.txt ./ && aws s3 cp s3://recommendation-data-bagpipe/emb_table_info_h.txt ./"
        }
        for i in range(len(private_ip_trainers))
    ]

    return run_args_move_files


run_args_oc_warmnup = [
    {
        "cmd": "aws s3 cp s3://recommendation-data-bagpipe/kaggle_criteo_info ./ && aws s3 cp s3://recommendation-data-bagpipe/kaggle_16 ./kaggle_16 --recursive && aws s3 cp s3://recommendation-data-bagpipe/emb_table_info.txt ./ && aws s3 cp s3://recommendation-data-bagpipe/fgcnn_16 ./fgcnn_16 --recursive && aws s3 cp s3://recommendation-data-bagpipe/emb_table_info_h.txt ./"
    }
]

run_args_ebs_warmnup = [
    {
        "cmd": "aws s3 cp s3://recommendation-data-bagpipe/kaggle_criteo_info ./ && aws s3 cp s3://recommendation-data-bagpipe/emb_table_info.txt ./ && aws s3 cp s3://recommendation-data-bagpipe/emb_table_info_h.txt ./"
    }
]


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
        # "ami_id": "ami-00cfdc3a2d9df3424",
        "ami_id": "ami-00893001ccd9f85b9",
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
    (
        private_ip_trainers,
        public_ip_trainers,
        instance_ids_trainers,
    ) = launch_instances_on_demand(launch_cfg)

    p3_az = get_az(instance_ids_trainers[0], launch_cfg)

    # launching oracle cacher
    launch_cfg["instance_type"] = "c5.18xlarge"
    launch_cfg["spot_price"] = "2.5"
    launch_cfg["method"] = "onDemand"
    launch_cfg["instance_count"] = 1
    # launch_cfg["ami_id"] = "ami-0a6479ed242da39cd"

    private_ips, public_ips, instance_ids = launch_instances_on_demand(launch_cfg)

    private_ip_oracle_cacher = private_ips[0]
    public_ip_oracle_cacher = public_ips[0]
    instance_id_oracle_cacher = instance_ids[0]

    # launch emb server
    launch_cfg["instance_type"] = "c5.18xlarge"
    launch_cfg["spot_price"] = "2.5"
    launch_cfg["method"] = "onDemand"
    launch_cfg["instance_count"] = 1
    # launch_cfg["ami_id"] = "ami-07526246b6e8e6e4c"

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

    # trainer client warmup ebs

    run_args_get_data = return_data_move_args_original(private_ip_trainers)

    time.sleep(60)

    output_trainers = client_trainers.run_command(
        "%(cmd)s", host_args=run_args_get_data
    )

    output_line_count_oc = client_oracle_cacher.run_command(
        "%(cmd)s", host_args=run_args_oc_warmnup
    )

    output_line_count_ebs = client_emb_server.run_command(
        "%(cmd)s", host_args=run_args_ebs_warmnup
    )

    for hosts_out in output_trainers:
        for line in hosts_out.stdout:
            print(line)

    for hosts_out in output_line_count_oc:
        for line in hosts_out.stdout:
            print(line)

    for hosts_out in output_line_count_ebs:
        for line in hosts_out.stdout:
            print(line)

    batch_size = 16384 // 8
    kaggle_size = [4000000, 7000000, 15000000]
    fgcnn_size = [6000000, 8000000]
    cache_size_kaggle = kaggle_size[0]
    cache_size_fgcnn = fgcnn_size[0]
    iterations = 2000
    with_dlrm = True
    with_wdn = False
    with_dcn = False
    with_dfm = False
    with_fgcnn = False
    with_caser = False
    if with_dlrm:
        # ========Launching Bagpipe run 1========================================
        cache_size = cache_size_kaggle
        log_file_name = "run_dlrm_{}_num_machines_{}_run_1".format(
            len(private_ip_trainers), batch_size
        )
        run_args_trainers = return_args_trainers_dlrm(
            private_ip_trainers,
            private_ip_oracle_cacher,
            log_file_name,
            iterations,
            cache_size,
        )

        run_args_emb_server = return_args_emb_server_dlrm(
            private_ip_trainers, private_ip_oracle_cacher
        )

        run_args_oracle_cacher = return_args_oracle_server_dlrm(
            private_ip_trainers, private_ip_oracle_cacher, batch_size, cache_size
        )

        print("Run args trainer {}".format(run_args_trainers))
        print("Run args emb server {}".format(run_args_emb_server))
        print("Run args oracle cacher {}".format(run_args_oracle_cacher))

        output_trainers = client_trainers.run_command(
            "%(cmd)s", host_args=run_args_trainers
        )

        output_emb_server = client_emb_server.run_command(
            "%(cmd)s", host_args=run_args_emb_server
        )

        output_oracle_cacher = client_oracle_cacher.run_command(
            "%(cmd)s", host_args=run_args_oracle_cacher
        )

        for hosts_out in output_trainers:
            for line in hosts_out.stdout:
                print(line)

        time.sleep(60)

        run_args_kill_oracle = [{"cmd": "pkill -9 python"}]
        run_args_kill_emb_server = [{"cmd": "pkill -9 python"}]

        kill_emb_server = client_emb_server.run_command(
            "%(cmd)s", host_args=run_args_kill_emb_server
        )

        kill_oracle_cacher = client_oracle_cacher.run_command(
            "%(cmd)s", host_args=run_args_kill_oracle
        )
        print("Launched python kill command")
        time.sleep(30)

    if with_wdn:
        # ========Launching Bagpipe run 1========================================
        cache_size = cache_size_kaggle
        log_file_name = "run_wdn_{}_num_machines_{}_run_1".format(
            len(private_ip_trainers), batch_size
        )
        run_args_trainers = return_args_trainers_wdn(
            private_ip_trainers,
            private_ip_oracle_cacher,
            log_file_name,
            iterations,
            cache_size,
        )

        run_args_emb_server = return_args_emb_server_wdn(
            private_ip_trainers, private_ip_oracle_cacher
        )

        run_args_oracle_cacher = return_args_oracle_server_dlrm(
            private_ip_trainers, private_ip_oracle_cacher, batch_size, cache_size
        )

        print("Run args trainer {}".format(run_args_trainers))
        print("Run args emb server {}".format(run_args_emb_server))
        print("Run args oracle cacher {}".format(run_args_oracle_cacher))

        output_trainers = client_trainers.run_command(
            "%(cmd)s", host_args=run_args_trainers
        )

        output_emb_server = client_emb_server.run_command(
            "%(cmd)s", host_args=run_args_emb_server
        )

        output_oracle_cacher = client_oracle_cacher.run_command(
            "%(cmd)s", host_args=run_args_oracle_cacher
        )

        for hosts_out in output_trainers:
            for line in hosts_out.stdout:
                print(line)

        time.sleep(60)

        run_args_kill_oracle = [{"cmd": "pkill -9 python"}]
        run_args_kill_emb_server = [{"cmd": "pkill -9 python"}]

        kill_emb_server = client_emb_server.run_command(
            "%(cmd)s", host_args=run_args_kill_emb_server
        )

        kill_oracle_cacher = client_oracle_cacher.run_command(
            "%(cmd)s", host_args=run_args_kill_oracle
        )
        print("Launched python kill command")
        time.sleep(30)

    if with_dcn:
        # ========Launching Bagpipe run 1========================================
        cache_size = cache_size_kaggle
        log_file_name = "run_dcn_{}_num_machines_{}_run_1".format(
            len(private_ip_trainers), batch_size
        )
        run_args_trainers = return_args_trainers_dcn(
            private_ip_trainers,
            private_ip_oracle_cacher,
            log_file_name,
            iterations,
            cache_size,
        )

        run_args_emb_server = return_args_emb_server_dlrm(
            private_ip_trainers, private_ip_oracle_cacher
        )

        run_args_oracle_cacher = return_args_oracle_server_dlrm(
            private_ip_trainers, private_ip_oracle_cacher, batch_size, cache_size
        )

        print("Run args trainer {}".format(run_args_trainers))
        print("Run args emb server {}".format(run_args_emb_server))
        print("Run args oracle cacher {}".format(run_args_oracle_cacher))

        output_trainers = client_trainers.run_command(
            "%(cmd)s", host_args=run_args_trainers
        )

        output_emb_server = client_emb_server.run_command(
            "%(cmd)s", host_args=run_args_emb_server
        )

        output_oracle_cacher = client_oracle_cacher.run_command(
            "%(cmd)s", host_args=run_args_oracle_cacher
        )

        for hosts_out in output_trainers:
            for line in hosts_out.stdout:
                print(line)

        time.sleep(60)

        run_args_kill_oracle = [{"cmd": "pkill -9 python"}]
        run_args_kill_emb_server = [{"cmd": "pkill -9 python"}]

        kill_emb_server = client_emb_server.run_command(
            "%(cmd)s", host_args=run_args_kill_emb_server
        )

        kill_oracle_cacher = client_oracle_cacher.run_command(
            "%(cmd)s", host_args=run_args_kill_oracle
        )
        print("Launched python kill command")
        time.sleep(30)

    if with_dfm:
        # ========Launching Bagpipe run 1========================================
        cache_size = cache_size_kaggle
        log_file_name = "run_dfm_{}_num_machines_{}_run_1".format(
            len(private_ip_trainers), batch_size
        )
        run_args_trainers = return_args_trainers_dfm(
            private_ip_trainers,
            private_ip_oracle_cacher,
            log_file_name,
            iterations,
            cache_size,
        )

        run_args_emb_server = return_args_emb_server_dlrm(
            private_ip_trainers, private_ip_oracle_cacher
        )

        run_args_oracle_cacher = return_args_oracle_server_dlrm(
            private_ip_trainers, private_ip_oracle_cacher, batch_size, cache_size
        )

        print("Run args trainer {}".format(run_args_trainers))
        print("Run args emb server {}".format(run_args_emb_server))
        print("Run args oracle cacher {}".format(run_args_oracle_cacher))

        output_trainers = client_trainers.run_command(
            "%(cmd)s", host_args=run_args_trainers
        )

        output_emb_server = client_emb_server.run_command(
            "%(cmd)s", host_args=run_args_emb_server
        )

        output_oracle_cacher = client_oracle_cacher.run_command(
            "%(cmd)s", host_args=run_args_oracle_cacher
        )

        for hosts_out in output_trainers:
            for line in hosts_out.stdout:
                print(line)

        time.sleep(60)

        run_args_kill_oracle = [{"cmd": "pkill -9 python"}]
        run_args_kill_emb_server = [{"cmd": "pkill -9 python"}]

        kill_emb_server = client_emb_server.run_command(
            "%(cmd)s", host_args=run_args_kill_emb_server
        )

        kill_oracle_cacher = client_oracle_cacher.run_command(
            "%(cmd)s", host_args=run_args_kill_oracle
        )
        print("Launched python kill command")
        time.sleep(30)

    if with_fgcnn:
        cache_size = cache_size_fgcnn
        # ========Launching Bagpipe run 1========================================
        log_file_name = "run_fgcnn_{}_num_machines_{}_run_1".format(
            len(private_ip_trainers), batch_size
        )
        run_args_trainers = return_args_trainers_fgcnn(
            private_ip_trainers,
            private_ip_oracle_cacher,
            log_file_name,
            iterations,
            cache_size,
        )

        run_args_emb_server = return_args_emb_server_fgcnn(
            private_ip_trainers, private_ip_oracle_cacher
        )

        run_args_oracle_cacher = return_args_oracle_server_fgcnn(
            private_ip_trainers, private_ip_oracle_cacher, batch_size, cache_size
        )

        print("Run args trainer {}".format(run_args_trainers))
        print("Run args emb server {}".format(run_args_emb_server))
        print("Run args oracle cacher {}".format(run_args_oracle_cacher))

        output_trainers = client_trainers.run_command(
            "%(cmd)s", host_args=run_args_trainers
        )

        output_emb_server = client_emb_server.run_command(
            "%(cmd)s", host_args=run_args_emb_server
        )

        output_oracle_cacher = client_oracle_cacher.run_command(
            "%(cmd)s", host_args=run_args_oracle_cacher
        )

        for hosts_out in output_trainers:
            for line in hosts_out.stdout:
                print(line)

        time.sleep(60)

        run_args_kill_oracle = [{"cmd": "pkill -9 python"}]
        run_args_kill_emb_server = [{"cmd": "pkill -9 python"}]

        kill_emb_server = client_emb_server.run_command(
            "%(cmd)s", host_args=run_args_kill_emb_server
        )

        kill_oracle_cacher = client_oracle_cacher.run_command(
            "%(cmd)s", host_args=run_args_kill_oracle
        )
        print("Launched python kill command")
        time.sleep(30)

    if with_caser:
        cache_size = 500000
        # ========Launching Bagpipe run 1========================================
        log_file_name = "run_caser_{}_num_machines_{}_run_1".format(
            len(private_ip_trainers), batch_size
        )
        run_args_trainers = return_args_trainers_caser(
            private_ip_trainers,
            private_ip_oracle_cacher,
            log_file_name,
            iterations,
            cache_size,
        )

        run_args_emb_server = return_args_emb_server_caser(
            private_ip_trainers, private_ip_oracle_cacher
        )

        run_args_oracle_cacher = return_args_oracle_server_caser(
            private_ip_trainers, private_ip_oracle_cacher, batch_size, cache_size
        )

        print("Run args trainer {}".format(run_args_trainers))
        print("Run args emb server {}".format(run_args_emb_server))
        print("Run args oracle cacher {}".format(run_args_oracle_cacher))

        output_trainers = client_trainers.run_command(
            "%(cmd)s", host_args=run_args_trainers
        )

        output_emb_server = client_emb_server.run_command(
            "%(cmd)s", host_args=run_args_emb_server
        )

        output_oracle_cacher = client_oracle_cacher.run_command(
            "%(cmd)s", host_args=run_args_oracle_cacher
        )

        for hosts_out in output_trainers:
            for line in hosts_out.stdout:
                print(line)

        time.sleep(60)

        run_args_kill_oracle = [{"cmd": "pkill -9 python"}]
        run_args_kill_emb_server = [{"cmd": "pkill -9 python"}]

        kill_emb_server = client_emb_server.run_command(
            "%(cmd)s", host_args=run_args_kill_emb_server
        )

        kill_oracle_cacher = client_oracle_cacher.run_command(
            "%(cmd)s", host_args=run_args_kill_oracle
        )
        print("Launched python kill command")
        time.sleep(30)

    terminate_instances(instance_ids_trainers, launch_cfg)
    terminate_instances([instance_id_emb_server, instance_id_oracle_cacher], launch_cfg)


if __name__ == "__main__":
    run_large_scale()
