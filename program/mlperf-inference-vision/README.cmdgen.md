# MLPerf Inference Vision - extended for Object Detection - CmdGen

CmdGen maps high-level CmdGen commands to low-level CK commands.

## Example

A high-level command:

```
"ck gen cmdgen:benchmark.mlperf-inference-vision --sut=chai \
  --scenario=offline --mode=accuracy --dataset_size=50 \
  --model=yolo-v3-coco --library=tensorflow-v2.7.1-cpu"
```

gets mapped to:

```
time docker run -it --rm ${CK_IMAGE} \
"ck run program:mlperf-inference-vision --cmd_key=direct --skip_print_timers \
    --env.CK_LOADGEN_SCENARIO=Offline \
    --env.CK_LOADGEN_MODE='--accuracy' \
    --env.CK_LOADGEN_EXTRA_PARAMS='--count 50' \
    \
    --dep_add_tags.weights=yolo-v3-coco \
    --env.CK_MODEL_PROFILE=tf_yolo \
    \
    --env.CK_INFERENCE_ENGINE=tensorflow \
    --env.CK_INFERENCE_ENGINE_BACKEND=default-cpu\
    --env.CUDA_VISIBLE_DEVICES=-1"
```

## Save experimental results into a host directory

The user should belong to the group `krai` on the host machine.
If it does not exist:

```
sudo groupadd krai
sudo usermod -aG krai $USER
```
### Set up environment
```
export CK_IMAGE_NAME=mlperf-inference-vision SDK_VER=21.08-py3 TF_VER=2.7.1
export CK_IMAGE="krai/${CK_IMAGE_NAME}:${SDK_VER}_tf-${TF_VER}"
export CK_EXPERIMENT_REPO="mlperf.object-detection.$(hostname).$(id -un)"
export CK_EXPERIMENT_DIR="${HOME}/CK/${CK_EXPERIMENT_REPO}/experiment"
```

### Create a new repository

```
ck add repo:$CK_EXPERIMENT_REPO --quiet
ck add $CK_EXPERIMENT_REPO:experiment:dummy --common_func
ck rm  $CK_EXPERIMENT_REPO:experiment:dummy --force
```

### Make its `experiment` directory writable by group `krai`

```
chgrp -R krai $CK_EXPERIMENT_DIR
chmod -R g+ws $CK_EXPERIMENT_DIR
setfacl -R -d -m group:krai:rwx $CK_EXPERIMENT_DIR
```

### Run

#### Run a CmdGen command from a Docker command

```
export CK_IMAGE_NAME=mlperf-inference-vision SDK_VER=21.08-py3 TF_VER=2.7.1
export CK_IMAGE="krai/${CK_IMAGE_NAME}:${SDK_VER}_tf-${TF_VER}"
export CK_EXPERIMENT_REPO="mlperf.object-detection.$(hostname).$(id -un)"
export CK_EXPERIMENT_DIR="${HOME}/CK/${CK_EXPERIMENT_REPO}/experiment"

docker run --user=krai:kraig --group-add $(cut -d: -f3 < <(getent group krai)) \
--volume ${CK_EXPERIMENT_DIR}:/home/krai/CK_REPOS/local/experiment --rm ${CK_IMAGE} \
"ck run cmdgen:benchmark.mlperf-inference-vision --verbose \
--scenario=offline --mode=accuracy --dataset_size=50 --buffer_size=64 \
--model=yolo-v3-coco --library=tensorflow-v2.7.1-cpu --sut=chai"
```

#### Run a Docker command from a CmdGen command

```
export CK_IMAGE_NAME=mlperf-inference-vision SDK_VER=21.08-py3 TF_VER=2.7.1
export CK_IMAGE="krai/${CK_IMAGE_NAME}:${SDK_VER}_tf-${TF_VER}"
export CK_EXPERIMENT_REPO="mlperf.object-detection.$(hostname).$(id -un)"
export CK_EXPERIMENT_DIR="${HOME}/CK/${CK_EXPERIMENT_REPO}/experiment"

CONTAINER_ID=`ck run cmdgen:benchmark.mlperf-inference-vision --docker=container_only --out=none  --library=tensorflow-v2.7.1-gpu --docker_image=${CK_IMAGE} --experiment_dir`

ck run cmdgen:benchmark.mlperf-inference-vision --verbose \
--scenario=offline --mode=accuracy --dataset_size=50 --buffer_size=64 \
--model=yolo-v3-coco --library=tensorflow-v2.7.1-cpu --sut=chai --container=$CONTAINER_ID
```

---
---

## Mappings

### Mapping for `--scenario`

It will affect the following flags in the CK environment:

```
--env.CK_LOADGEN_SCENARIO=[SCENARIO]
```
|SCENARIO|
|---|
| `SingleStream` |
| `MultiStream` | 
| `Server` |
| `Offline` |

---
---

### Mapping for `--mode`, `--dataset_size`,  `--target_qps`, `--target_latency`, `--buffer_size`, <br>`--query_count` and `--cache_opt`

`mode` for specifying mode,

`dataset_size` for specifying `count`,

`target_qps` for specifying `qps`,

`target_latency` for specifying `target-latency`,

`buffer_size` for specifying `performance-sample-count`,

`query_count` for specifying `query-count`,

`cache_opt` for specifying `cache`.

It will affect the following flags in the CK environment:
```
--env.CK_LOADGEN_MODE
--env.CK_LOADGEN_EXTRA_PARAMS
--env.CK_OPTIMIZE_GRAPH
```

| Accuracy Mode | Performance Mode |
| ---------------------- | --------------------|
|`--env.CK_LOADGEN_MODE='--accuracy'` <br> `--env.CK_LOADGEN_EXTRA_PARAMS=` <br> `'--count 5000 ` <br> `--performance-sample-count 50 --cache 1'` <br> `--env.CK_OPTIMIZE_GRAPH='False'`| `--env.CK_LOADGEN_EXTRA_PARAMS='----count 50 --qps 200` <br> `--target-latency 35 --performance-sample-count 64 --query-count 2048 --cache 1'` <br> `--env.CK_OPTIMIZE_GRAPH='True'`|


For accuracy:
`--performance-sample-count` 50 is optional

For performance:
`--qps 200`, `--target-latency 35`,  `--query-count 2048` are optional.

---
---
### Mapping for `--model`

It will affect the following flags in the ck environment:
```
--dep_add_tags.weights=[TF_ZOO],[MODEL_NAME]
--env.CK_MODEL_PROFILE=[MODEL_PROFILE]
--env.CK_INFERENCE_ENGINE=[INFERENCE_ENGINE] (as shown in README.md)
--env.CK_INFERENCE_ENGINE_BACKEND=[INFERENCE_ENGINE_BACKEND] (as shown in README.md)
--env.CUDA_VISIBLE_DEVICES=[DEVICE_IDS] (will be discussed in the next section)
```

See README.md for the table.

---
---

## Mapping for `--library`
It will affect the following flags in the ck environment:
```
--env.CK_INFERENCE_ENGINE=[INFERENCE_ENGINE]
--env.CK_INFERENCE_ENGINE_BACKEND=[INFERENCE_ENGINE_BACKEND]
--env.CUDA_VISIBLE_DEVICES=[DEVICE_IDS]
```

|INFERENCE_ENGINE|INFERENCE_ENGINE_BACKEND|DEVICE_IDS|
|---|---|---|
|`tensorflow` |`default-cpu` |`-1`|
|`tensorflow` |`default-gpu` |`<device_id>`|
|`tensorflow` |`tensorrt-dynamic` |`<device_id>`|
|`tensorflow` |`openvino-cpu`|`-1`|
|`tensorflow` |`openvino-gpu` |`-1` for an Intel chip with an integrated GPU; `0` for an Intel GPU|
