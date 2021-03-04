# Description

This program performs "performance ranging" by extracting the mean latency over a fixed number of queries.
This latency can then be used for proper performance runs. This is especially useful for families of models
such as MobileNet and EfficientNet where the performance can range by an order of magnitude.

# Usage

## Store

Run `cmdgen:benchmark.*` with `--scenario=range_singlestream --max_query_count=<...>` arguments to store performance ranging results to the local CK repo.

### Example

```bash
$ ck run cmdgen:benchmark.image-classification.tflite-loadgen \
--library=armnn-v21.02-neon --model=resnet50 --mode=performance \
--scenario=range_singlestream --max_query_count=100 \
--verbose --sut=xavier
```

## Extract

Use this script to extract the ranging results from the CK repo.
All arguments are optional.

```bash
$ ck run ck-mlperf:program:generate-target-latency \
  --env.CK_MLPERF_SUBMISSION_REPO=local \
  --env.CK_MLPERF_SUBMISSION_TAGS=foo,bar \
  --env.CK_MLPERF_SUBMISSION_OUT="$PWD"/target_latency.txt
```

Alternatively:

```bash
$ cd $(ck find program:generate-target-latency)
$ python3 generate-target-latency/run.py \
  --repo-uoa local \
  --tags foo,bar \
  --out $(pwd)/target_latency.txt
```

### Example

```bash
$ generate-target-latency/run.py | tee target_latency.txt
xavier,armnn-v20.08-neon,resnet50     64 # max_query_count=100
```


### Tags

The tags are related to experiments, i.e. the following query matches the same experiments as `./run.py --tags=$tags`:

```bash
$ ck search experiment --tags=mlperf,scenario.range_singlestream.$tags`
```

You can use the following one-liner to list models with corresponding tags:
```bash
 $ for i in $(ck search experiment --tags=mlperf,scenario.range_singlestream); do echo $i; ck list_tags $i; echo; done
```

## Use

Run `cmdgen:benchmark.*` with `--target_latency_file=<...>` instead of `--target_latency=<...>`.

### Example

```bash
$ ck run cmdgen:benchmark.image-classification.tflite-loadgen \
--library=armnn-v21.02-neon --model=resnet50 --mode=performance \
--scenario=singlestream --target_latency_file=target_latency.txt \
--verbose --sut=xavier
```
