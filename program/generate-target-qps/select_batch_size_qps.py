#!/usr/bin/env python3

import argparse

def main(args):
    max_qps = 0
    max_b_size = 1
    with open(args.input_file, "r") as f:
        for line in f.readlines():
            line = line.split(',')
            if args.batch_size == "":
                batch_size = 0
            else:
                batch_size= int(args.batch_size)
            if ( args.sut == line[0] and args.inference == line[1] and args.model == line[2]):
                b_size = int(line[3].split('=')[1])
                target_qps = int(line[4])
                if batch_size != b_size:
                    if max_qps < target_qps:
                        max_qps = target_qps
                        if batch_size == 0:
                            max_b_size = b_size
                        else:
                            max_b_size = batch_size
                else:
                    max_qps = target_qps
                    max_b_size = b_size
                    break;
                    break;
    print(max_b_size, max_qps)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sut",
        metavar="SUT",
        type=str,
    )
    parser.add_argument(
        "--inference",
        metavar="INFERENCE",
        type=str,
        help="set as <inference_engine>-<inference_engine_version>-<inference_engine_backend>"
    )
    parser.add_argument(
        "--batch_size",
        metavar="BATCH_SIZE",
        type=str,
    )
    parser.add_argument(
        "--model",
        metavar="MODEL",
        type=str,
    )
    parser.add_argument(
        "--input_file",
        metavar="FILE",
        type=str,
    )
    main(parser.parse_args())

