#!/usr/bin/env python3

import array
import os
import random
import time

import numpy as np
import mlperf_loadgen as lg


## LoadGen test properties:
#
LOADGEN_SCENARIO        = os.getenv('CK_LOADGEN_SCENARIO', 'SingleStream')
LOADGEN_MODE            = os.getenv('CK_LOADGEN_MODE', 'AccuracyOnly')
LOADGEN_BUFFER_SIZE     = int(os.getenv('CK_LOADGEN_BUFFER_SIZE'))          # set to how many samples are you prepared to keep in memory at once
LOADGEN_DATASET_SIZE    = int(os.getenv('CK_LOADGEN_DATASET_SIZE'))         # set to how many total samples to choose from (0 = full set)
LOADGEN_COUNT_OVERRIDE  = os.getenv('CK_LOADGEN_COUNT_OVERRIDE', '')        # if not set, use value from LoadGen's config file
LOADGEN_MULTISTREAMNESS = os.getenv('CK_LOADGEN_MULTISTREAMNESS', '')       # if not set, use value from LoadGen's config file
#
MLPERF_CONF_PATH        = os.environ['CK_ENV_MLPERF_INFERENCE_MLPERF_CONF']
USER_CONF_PATH          = os.environ['CK_LOADGEN_USER_CONF']
#
MODEL_NAME              = os.environ['ML_MODEL_MODEL_NAME']                 # an obligatory input parameter for now


## Specific for this example:
#
LATENCY_SECONDS         = float(os.environ['CK_EXAMPLE_LATENCY_SEC'])       # fractional seconds


## Global input data and expected labels:
#
dataset                 = [10*i for i in range(LOADGEN_DATASET_SIZE)]
labelset                = [10*i+random.randint(0,1) for i in range(LOADGEN_DATASET_SIZE)]


def predict_label(x_vector):
    time.sleep(LATENCY_SECONDS)
    return int(x_vector/10)+1


def issue_queries(query_samples):

    printable_query = [(qs.index, qs.id) for qs in query_samples]
    print("LG: issue_queries( {} )".format(printable_query))

    predicted_results = {}
    for qs in query_samples:
        query_index, query_id = qs.index, qs.id

        x_vector        = dataset[query_index]
        predicted_label = predict_label(x_vector)

        predicted_results[query_index] = predicted_label
    print("LG: predicted_results = {}".format(predicted_results))

    response = []
    for qs in query_samples:
        query_index, query_id = qs.index, qs.id

        response_array = array.array("B", np.array(predicted_results[query_index], np.float32).tobytes())
        bi = response_array.buffer_info()
        response.append(lg.QuerySampleResponse(query_id, bi[0], bi[1]))
    lg.QuerySamplesComplete(response)


def flush_queries():
    print("LG called flush_queries()")

def process_latencies(latencies_ns):
    print("LG called process_latencies({})".format(latencies_ns))

    latencies_size      = len(latencies_ns)
    latencies_avg       = int(sum(latencies_ns)/latencies_size)
    latencies_sorted    = sorted(latencies_ns)
    latencies_p50       = int(latencies_size * 0.5);
    latencies_p90       = int(latencies_size * 0.9);

    print("--------------------------------------------------------------------")
    print("|                LATENCIES (in nanoseconds and fps)                |")
    print("--------------------------------------------------------------------")
    print("Number of queries run:       {:9d}".format(latencies_size))
    print("Min latency:                 {:9d} ns   ({:.3f} fps)".format(latencies_sorted[0], 1e9/latencies_sorted[0]))
    print("Median latency:              {:9d} ns   ({:.3f} fps)".format(latencies_sorted[latencies_p50], 1e9/latencies_sorted[latencies_p50]))
    print("Average latency:             {:9d} ns   ({:.3f} fps)".format(latencies_avg, 1e9/latencies_avg))
    print("90 percentile latency:       {:9d} ns   ({:.3f} fps)".format(latencies_sorted[latencies_p90], 1e9/latencies_sorted[latencies_p90]))
    print("Max latency:                 {:9d} ns   ({:.3f} fps)".format(latencies_sorted[-1], 1e9/latencies_sorted[-1]))
    print("--------------------------------------------------------------------")


def load_query_samples(sample_indices):
    print("LG called load_query_samples({})".format(sample_indices))

def unload_query_samples(sample_indices):
    print("LG called unload_query_samples({})".format(sample_indices))
    print("")


def benchmark_using_loadgen():
    "Perform the benchmark using python API for the LoadGen library"

    scenario = {
        'SingleStream':     lg.TestScenario.SingleStream,
        'MultiStream':      lg.TestScenario.MultiStream,
        'Server':           lg.TestScenario.Server,
        'Offline':          lg.TestScenario.Offline,
    }[LOADGEN_SCENARIO]

    mode = {
        'AccuracyOnly':     lg.TestMode.AccuracyOnly,
        'PerformanceOnly':  lg.TestMode.PerformanceOnly,
        'SubmissionRun':    lg.TestMode.SubmissionRun,
    }[LOADGEN_MODE]

    ts = lg.TestSettings()
    ts.FromConfig(MLPERF_CONF_PATH, MODEL_NAME, LOADGEN_SCENARIO)
    ts.FromConfig(USER_CONF_PATH, MODEL_NAME, LOADGEN_SCENARIO)
    ts.scenario = scenario
    ts.mode     = mode

    sut = lg.ConstructSUT(issue_queries, flush_queries, process_latencies)
    qsl = lg.ConstructQSL(LOADGEN_DATASET_SIZE, LOADGEN_BUFFER_SIZE, load_query_samples, unload_query_samples)

    log_settings = lg.LogSettings()
    log_settings.enable_trace = False
    lg.StartTestWithLogSettings(sut, qsl, ts, log_settings)

    lg.DestroyQSL(qsl)
    lg.DestroySUT(sut)


try:
    benchmark_using_loadgen()
except Exception as e:
    print(str(e))
