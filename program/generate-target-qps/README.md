# Description

This program performs "performance ranging" by extracting the target qps (samples per second) over a fixed number of queries.
This qps can then be used for proper performance runs.

# Usage

## Store

Run `cmdgen:benchmark.*` with `--scenario=range_offline --query_count=<...>` arguments to store performance ranging results to the local CK repo.

### Example

```bash
export CK_IMAGE_NAME=mlperf-inference-vision SDK_VER=22.01-py3 TF_VER=2.7.1
export CK_IMAGE="krai/${CK_IMAGE_NAME}:${SDK_VER}_tf-${TF_VER}"
export CK_EXPERIMENT_REPO="mlperf_v2.0.object-detection.$(hostname).$(id -un)"
export CK_EXPERIMENT_DIR="${HOME}/CK/${CK_EXPERIMENT_REPO}/experiment"

CONTAINER_ID=`ck run cmdgen:benchmark.mlperf-inference-vision --docker=container_only \
--out=none  --library=tensorflow-v2.7.1-gpu --docker_image=${CK_IMAGE} --experiment_dir`

$ time ck run cmdgen:benchmark.mlperf-inference-vision --verbose --sut=chai \
--model:=$(ck list_variations misc --query_module_uoa=package --tags=model,tf2-zoo --separator=:):$(ck list_variations misc --query_module_uoa=package --tags=model,\
tf1-zoo --separator=:) --library=tensorflow-v2.7.1-gpu --device_ids=1 --scenario=range_offline \
--mode=performance --dataset_size=5000  --batch_size=1 --query_count=1024 --container=$CONTAINER_ID
```

## Extract

Use this script to extract the ranging results from the CK repo.
All arguments are optional.

```bash
$ cd $(ck find program:generate-target-qps) && ./run.py \
  --repo-uoa local \
  --tags foo,bar \
  --out $(pwd)/target_qps.txt
```

### Example

```bash
$ $(ck find ck-mlperf:program:generate-target-qps)/run.py --tags=inference_engine.tensorflow,inference_engine_version.v2.7.1 --repo_uoa=$CK_EXPERIMENT_REPO | \
sort | tee -a $(ck find ck-mlperf:program:mlperf-inference-vision)/target_qps.chai.txt
chai,tensorflow-v2.7.1-default-gpu,rcnn-nas-lowproposals-coco  5.2 # max_query_count=1024

```

### Tags

The tags are related to experiments, i.e. the following query matches the same experiments as `./run.py --tags=$tags`:

```bash
$ ck search experiment --tags=mlperf,scenario.range_offline.$tags`
```

You can use the following one-liner to list tags for particular ranging experiments:
```bash
 $ for i in $(ck search experiment --tags=mlperf,scenario.range_offline); do echo $i; ck list_tags $i; echo; done
```

## Use

Run `cmdgen:benchmark.*` with `--target_qps_file=<...>` instead of `--target_qps=<...>`.

### Example

```bash
$ time ck run cmdgen:benchmark.mlperf-inference-vision --verbose --sut=chai \
--model:=$(ck list_variations misc --query_module_uoa=package --tags=model,tf2-zoo --separator=:):$(ck list_variations misc --query_module_uoa=package --tags=model,\
tf1-zoo --separator=:)--library=tensorflow-v2.7.1-gpu --device_ids=1 --scenario=offline --mode=performance --dataset_size=5000  --batch_size=1 \
---target_qps_file=/home/krai/CK_REPOS/ck-mlperf/program/mlperf-inference-vision/target_qps.chai.txt --container=$CONTAINER_ID \
--power=yes --power_server_port=4951 --power_server_ip=192.168.0.3

```
