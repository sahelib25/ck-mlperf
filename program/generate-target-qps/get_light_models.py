#!/usr/bin/env python3

import argparse

def parse_tags(tags):
    result = {}
    tags = tags.split(',')
    for tag in tags:
        if "." in tag:
            key, value = tag.split(".", 1)
            result[key] = value
        else:
            result[tag] = None
    return result

def main(args):
    list_tags = parse_tags(args.tags)
    inference_str = list_tags["inference_engine"] + '-' + list_tags["inference_engine_version"] + '-' + list_tags["inference_engine_backend"]
    light_models_str = ""
    with open(args.input_file, "r") as f:
        for line in f.readlines():
            line = line.split()
            target_latency = line[1]
            inference_model_list = line[0].split(',')
            if (inference_model_list[0] == list_tags["sut"] and inference_model_list[1] == inference_str):
                if ( int(target_latency) <= args.latency_threshold and inference_model_list[2] != "ssd_mobilenet_v1_quantized_coco"):
                    if light_models_str != "":
                        light_models_str= light_models_str + args.separator + inference_model_list[2]
                    else:
                        light_models_str = inference_model_list[2]
    f.close
    print(light_models_str)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tags",
        metavar="TAGS",
        type=str,
    )
    parser.add_argument(
        "--latency_threshold",
        metavar="LATENCY_THRESHOLD",
        type=int,
        default=80,
    )
    parser.add_argument(
        "--separator",
        metavar="SEPARATOR",
        type=str,
        default=":",
    )
    parser.add_argument(
        "--input_file",
        metavar="FILE",
        type=str,
    )
    main(parser.parse_args())

