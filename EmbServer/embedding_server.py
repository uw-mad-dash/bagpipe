import argparse
import sys

sys.path.append("..")
sys.path.append(".")

from BagPipe import init_emb_server


def parse_args(parser):
    parser.add_argument("--emb-size", type=int, default=48)
    parser.add_argument(
        "--ln-emb",
        type=dash_separated_ints,
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
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--master-ip", type=str, default="localhost")
    parser.add_argument("--master-port", type=str, default="18000")
    parser.add_argument(
        "--emb-prefix",
        type=str,
        default="emb_worker",
        help="Name of embedding worker Currently I am assuming there is currently only one worker ID",
    )
    parser.add_argument("--emb-info-file", type=str, default=None)
    args = parser.parse_args()
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


if __name__ == "__main__":
    args = parse_args(argparse.ArgumentParser(
        description="Embedding server startup"))
    init_emb_server(args)
