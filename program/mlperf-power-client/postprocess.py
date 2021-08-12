#!/usr/bin/env python3

import base64
import datetime
import json
import os
import subprocess
import sys


def load_json(fname):
    with open(fname) as f:
        return json.load(f)


def load_lines(fname):
    with open(fname) as f:
        return list(map(lambda x: x.rstrip("\n\r"), f))


def load_raw(fname):
    with open(fname, "rb") as f:
        return base64.encodebytes(f.read()).decode("utf8")


def load_dir(basename, raw=False):
    result = {}
    for path, dirs, files in os.walk(basename, topdown=True):
        for file in files:
            p = os.path.join(path, file)
            if raw:
                r = load_raw(p)
            else:
                r = load_lines(p)
            result[os.path.relpath(p, basename)] = r
    result.pop("tmp-ck-timer.json", None)  # avoid self-containing
    return result


def ck_postprocess(i):

    inference_src_dep       = i['deps']['mlperf-inference-src']
    inference_root          = inference_src_dep['dict']['env']['CK_ENV_MLPERF_INFERENCE']
    submissions_code_dir    = os.path.join(inference_root, 'tools', 'submission')
    testing_path            = "run_1"

    sys.path.append( submissions_code_dir )
    from log_parser import MLPerfLog

    client_struct   = load_json("power/client.json")
    client_timezone = datetime.timedelta(seconds=client_struct["timezone"])

    server_struct   = load_json("power/server.json")
    server_timezone = datetime.timedelta(seconds=server_struct["timezone"])

    detail_log_fname = os.path.join(testing_path, "mlperf_log_detail.txt")
    mlperf_log = MLPerfLog(detail_log_fname)

    datetime_format = '%m-%d-%Y %H:%M:%S.%f'
    power_begin = datetime.datetime.strptime(mlperf_log["power_begin"], datetime_format) + client_timezone
    power_end = datetime.datetime.strptime(mlperf_log["power_end"], datetime_format) + client_timezone

    power_list  = []
    spl_fname   = os.path.join(testing_path, "spl.txt")
    for line in load_lines(spl_fname):
        timestamp = datetime.datetime.strptime(line.split(",")[1], datetime_format) + server_timezone
        if timestamp > power_begin and timestamp < power_end:
            power_list.append(float(line.split(",")[3]))

    power_count = len(power_list)
    avg_power = sum(power_list) / power_count if power_count>0 else 'NO POWER DATA'

    result = {
        "avg_power": avg_power,
        "power": {
            "client.json": client_struct,
            "client.log": load_lines("power/client.log"),
            "ptd_logs.txt": load_lines("power/ptd_logs.txt"),
            "server.json": server_struct,
            "server.log": load_lines("power/server.log"),
        },
        "ranging": load_dir("ranging"),
        "run_1": load_dir("run_1"),
        "raw": load_dir(".", True),
    }

    with open("tmp-ck-timer.json", "w") as f:
        json.dump(result, f, indent=2)

    return {'return': 0}

