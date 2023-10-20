import argparse
import sys

sys.path.append("..")
sys.path.append(".")

from BagPipe import init_oracle_cacher


def parse_args(parser):
    parser.add_argument(
        "--prefetch",
        action="store_true",
        default=False,
        help="Enable Prefetch for training",
    )

    parser.add_argument(
        "--cache",
        action="store_true",
        default=False,
        help="Enable Logic for caching",
    )
    parser.add_argument(
        "--lookahead-value",
        type=int,
        default=200,
        help="The number of batches further to look ahead for getting cache",
    )

    parser.add_argument(
        "--worker-addresses", type=str, help="Worker IP addresses to perform training"
    )

    parser.add_argument(
        "--ln-emb",
        type=list,
        # type=utils.dash_separated_ints,
        help="embedding table sizes in the right order",
        default=[
            1460,
            583,
            10131227,
            2202608,
            305,
            24,
            12517,
            633,
            3,
            93145,
            5683,
            8351593,
            3194,
            27,
            14992,
            5461306,
            10,
            5652,
            2173,
            4,
            7046547,
            18,
            15,
            286181,
            105,
            142572,
        ],
    )

    # parser.add_argument("")
    parser.add_argument(
        "--arch-mlp-bot", type=dash_separated_ints, default="4-3-2")
    parser.add_argument("--data-size", type=int, default=1)
    parser.add_argument("--num-batches", type=int, default=0)
    parser.add_argument("--data-generation", type=str, default="dataset")
    parser.add_argument("--data-set", type=str, default="kaggle")
    parser.add_argument("--raw-data-file", type=str, default="")
    parser.add_argument("--processed-data-file", type=str, default="")
    parser.add_argument("--data-randomize", type=str, default="total")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--data-trace-enable-padding",
                        type=bool, default=False)
    parser.add_argument("--max-ind-range", type=int, default=-1)
    parser.add_argument("--data-sub-sample-rate",
                        type=float, default=0.0)  # in [0, 1]
    parser.add_argument("--num-indices-per-lookup", type=int, default=5)
    parser.add_argument("--num-indices-per-lookup-fixed",
                        type=bool, default=False)
    parser.add_argument("--memory-map", action="store_true", default=False)
    parser.add_argument("--stop-iter", type=int, default=2000)
    parser.add_argument("--mini-batch-size", type=int, default=1)
    parser.add_argument("--test-mini-batch-size", type=int, default=10)
    parser.add_argument(
        "--dataset-multiprocessing",
        action="store_true",
        default=False,
        help="The Kaggle dataset can be multiprocessed in an environment \
                        with more than 7 CPU cores and more than 20 GB of memory. \n \
                        The Terabyte dataset can be multiprocessed in an environment \
                        with more than 24 CPU cores and at least 1 TB of memory.",
    )

    parser.add_argument("--dataset-multi-num", type=int, required=True)

    parser.add_argument(
        "--worker-id",
        type=int,
        required=True,
        help="Global Worker ID used for RPC init",
    )

    parser.add_argument("--cache-size", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--master-ip", type=str, default="localhost")
    parser.add_argument("--master-port", type=str, default="18000")

    parser.add_argument(
        "--oracle-prefix",
        type=str,
        default="oracle",
        help="Prefix to name oracle cacher",
    )
    parser.add_argument(
        # "--num-trainers",
        "--world-size-trainers",
        type=int,
        required=True,
        help="Number of trainers",
    )
    parser.add_argument(
        "--trainer-prefix",
        type=str,
        default="worker",
        help="prefix to call the trainer",
    )

    parser.add_argument(
        "--processed-csv", type=str, required=True, help="CSV file name"
    )
    parser.add_argument("--ragged", action="store_true", default=False, help="Ragged Sparse Vector?")
    parser.add_argument(
        "--multiplicative-decrease-param",
        type=int,
        default=2,
        help="Multiplicative decrease param when no cache space is available",
    )

    parser.add_argument(
        "--additive-increase-param",
        default=1,
        help="Additive increase param to increase training",
    )

    parser.add_argument(
        "--cleanup-proportion",
        type=float,
        default=0.25,
        help="Cleanup proportion to use",
    )
    
    parser.add_argument("--emb-info-file", type=str, default=None)
    parser.add_argument("-v", "--verbose", help="Be verbose", action="store_true", default=False)
    parser.add_argument("--log-file-path", type=str, default="oracle_cacher.log")

    args = parser.parse_args()

    if args.emb_info_file is not None:
        args.ln_emb = get_emb_length(args.emb_info_file)
    return args


def dash_separated_ints(value):
    vals = value.split("-")
    for val in vals:
        try:
            int(val)
        except ValueError:
            raise argparse.ArgumentTypeError(
                "%s is not a valid dash separated list of ints" % value
            )

    return value


def get_emb_length(in_file):
    with open(in_file, "r") as fin:
        data = fin.readlines()

    data = [int(d) for d in data]
    return data


if __name__ == "__main__":
    args = parse_args(argparse.ArgumentParser(description="Oracle cacher arguments"))
    init_oracle_cacher(args)
