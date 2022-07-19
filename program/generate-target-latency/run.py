#!/usr/bin/env python3

import argparse
import ck.kernel as ck
import json
import math
import os.path
import sys


def ck_access(**kwargs):
    r = ck.access(kwargs)
    if r["return"] > 0:
        print("Error: %s" % r["error"])
        exit(1)
    return r


def parse_ck_tags(tags):
    result = {}
    for tag in tags:
        if "." in tag:
            key, value = tag.split(".", 1)
            result[key] = value
        else:
            result[tag] = None
    return result


def parse_mlperf_log_detail(lines):
    PREFIX = ":::MLLOG "
    result = {}
    for line in lines:
        if not line.startswith(PREFIX):
            continue
        line = json.loads(line[len(PREFIX) :])
        if "key" in line and "value" in line:
            result[line["key"]] = line["value"]
    return result


def main(args):
    if(args.mode == "accuracy"):
        mode_tags="mode.accuracy"
    elif(args.mode == "performance"):
        mode_tags="mode.performance,scenario.range_singlestream"
    else:
        print("Please provide valid argument for mode.")
        exit(1)
    tags="mlperf," + mode_tags + ("," + args.tags if args.tags else "")

    experiments = ck_access(
            repo_uoa=args.repo_uoa,
            action="search",
            module_uoa="experiment",
            tags=tags,
        )["lst"]

    for experiment in experiments:
        pipeline = ck_access(
            action="load_pipeline",
            repo_uoa=experiment["repo_uoa"],
            module_uoa=experiment["module_uoa"],
            data_uoa=experiment["data_uoa"],
        )["pipeline"]

        points = ck_access(
            action="list_points",
            repo_uoa=experiment["repo_uoa"],
            module_uoa=experiment["module_uoa"],
            data_uoa=experiment["data_uoa"],
        )

        tags = parse_ck_tags(points["dict"]["tags"])
        library_backend = (
            tags["inference_engine"]
            + "-"
            + tags["inference_engine_version"]
            + (
                ("-" + tags["inference_engine_backend"])
                if "inference_engine_backend" in tags
                else ""
            )
        )

        for point in points["points"]:
            point_file_path = os.path.join(points["path"], "ckp-%s.0001.json" % point)
            with open(point_file_path) as point_file:
                point_data_raw = json.load(point_file)

            for characteristic in point_data_raw["characteristics_list"]:
                if(args.mode == "accuracy"):
                    metric = characteristic["run"]["accuracy"]
                    comment_value = characteristic["run"]["total"]
                    comment = "total"
                elif(args.mode == "performance"):
                    detail = parse_mlperf_log_detail(
                        characteristic["run"]["mlperf_log"]["detail"]
                    )
                    metric = math.ceil(detail["result_mean_latency_ns"] / 10 ** 6)
                    comment_value = tags.get("max_query_count", tags.get("query_count"))
                    comment = "max query count"
                print(
                    "{:35} {:-4} # {}={}".format(
                        tags["platform"]
                        + ","
                        + library_backend
                        + ","
                        + tags["workload"],
                        metric,
                        comment,
                        comment_value
                    ),
                    file=args.out,
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo_uoa",
        metavar="REPO_UOA",
        type=str,
        default="local",
        help="defaults to 'local'",
    )
    parser.add_argument(
        "--tags",
        metavar="TAGS",
        type=str,
        default="",
    )
    parser.add_argument(
        "--out",
        metavar="FILE",
        type=argparse.FileType("w"),
        default=sys.stdout,
    )
    parser.add_argument(
        "--mode",
        metavar="MODE",
        type=str,
        default="performance",
        help="defaults to 'performance'"
    )
    main(parser.parse_args())
