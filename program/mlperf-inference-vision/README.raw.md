# MLPerf Inference Vision - extended for Object Detection

This Collective Knowledge workflow is based on the [official MLPerf Inference Vision application](https://github.com/mlcommons/inference/tree/master/vision/classification_and_detection) extended for diverse Object Detection models, as found e.g. in the [TF1 Object Detection Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md) and the [TF2 Object Detection Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md).

The table below shows currently supported models, frameworks ("inference engines") and library/device combinations ("inference engine backends").

| `MODEL_NAME`                                 | `INFERENCE_ENGINE`  | `INFERENCE_ENGINE_BACKEND`                 |
| -------------------------------------------- | ------------------- | ------------------------------------------ |
| `rcnn-nas-lowproposals-coco`                 | `tensorflow`        | `default.cpu`,`default.gpu`,`openvino.cpu` |
| `rcnn-resnet50-lowproposals-coco`            | `tensorflow`        | `default.cpu`,`default.gpu`,`openvino.cpu` |
| `rcnn-resnet101-lowproposals-coco`           | `tensorflow`        | `default.cpu`,`default.gpu`,`openvino.cpu` |
| `rcnn-inception-resnet-v2-lowproposals-coco` | `tensorflow`        | `default.cpu`,`default.gpu`,`openvino.cpu` |
| `rcnn-inception-v2-coco`                     | `tensorflow`        | `default.cpu`,`default.gpu`,`openvino.cpu` |
| `ssd-inception-v2-coco`                      | `tensorflow`        | `default.cpu`,`default.gpu`                |
| `ssd_mobilenet_v1_coco`                      | `tensorflow`        | `default.cpu`,`default.gpu`                |
| `ssd_mobilenet_v1_quantized_coco`            | `tensorflow`        | `default.cpu`,`default.gpu`                |
| `ssd-mobilenet-v1-fpn-sbp-coco`              | `tensorflow`        | `default.cpu`,`default.gpu`                |
| `ssd-resnet50-v1-fpn-sbp-coco`               | `tensorflow`        | `default.cpu`,`default.gpu`                |
| `ssdlite-mobilenet-v2-coco`                  | `tensorflow`        | `default.cpu`,`default.gpu`                |
| `yolo-v3-coco`                               | `tensorflow`        | `default.cpu`,`default.gpu`,`openvino.cpu` |
| `ssd_resnet50_v1_fpn_640x640`                | `tensorflow`        | `default.cpu`,`default.gpu`                |
| `ssd_resnet50_v1_fpn_1024x1024`              | `tensorflow`        | `default.cpu`,`default.gpu`                |
|`ssd_resnet101_v1_fpn_640x640`                | `tensorflow`        | `default.cpu`,`default.gpu`                |
|`ssd_resnet101_v1_fpn_1024x1024`              | `tensorflow`        | `default.cpu`,`default.gpu`                |
|`ssd_resnet152_v1_fpn_640x640`                | `tensorflow`        | `default.cpu`,`default.gpu`                |
|`ssd_resnet152_v1_fpn_1024x1024`              | `tensorflow`        | `default.cpu`,`default.gpu`                |
|`ssd_mobilenet_v2_320x320`                    | `tensorflow`        | `default.cpu`,`default.gpu`                |
|`ssd_mobilenet_v1_fpn_640x640`                | `tensorflow`        | `default.cpu`,`default.gpu`                |
|`ssd_mobilenet_v2_fpnlite_320x320`            | `tensorflow`        | `default.cpu`,`default.gpu`                |
|`ssd_mobilenet_v2_fpnlite_640x640`            | `tensorflow`        | `default.cpu`,`default.gpu`                |


# A) Building the environment with Docker

## 1. Building the Docker image

In the following examples, TensorFlow 2.7.1 and NVIDIA container image for TensorRT 21.08 were used.
**NB:** The
[TensorRT 21.06](https://docs.nvidia.com/deeplearning/tensorrt/container-release-notes/rel_21-06.html#rel_21-06)
release is the last one to support TensorRT 7.2, needed by TensorFlow 2.7.

```
export CK_IMAGE_NAME=mlperf-inference-vision SDK_VER=21.08-py3 TF_VER=2.7.1
cd $(ck find program:$CK_IMAGE_NAME) && ./build.sh
```

<details>
<summary>Click to expand</summary>

```
Successfully built 9c39ebef9ad2
Successfully tagged krai/mlperf-inference-vision:21.08-py3_tf-2.7.1

real    14m29.990s
user    0m10.826s
sys     0m11.604s

Done.
```
</details>

Set an environment variable for the built image and validate:

```
export CK_IMAGE="krai/${CK_IMAGE_NAME}:${SDK_VER}_tf-${TF_VER}"
docker image ls ${CK_IMAGE}
```

```
REPOSITORY                     TAG                  IMAGE ID       CREATED         SIZE
krai/mlperf-inference-vision   21.08-py3_tf-2.7.1   362d3cd6ddd5   8 minutes ago   16.6GB
```

## 2. Using the Docker image

### a) Run the Docker image AND execute a CK command

Following the format below:
```
time docker run -it --rm ${CK_IMAGE} \
"ck run program:mlperf-inference-vision ... "
```

Example:
```
time docker run -it --rm ${CK_IMAGE} \
"ck run program:mlperf-inference-vision --cmd_key=direct --skip_print_timers \
  --env.CK_LOADGEN_SCENARIO=SingleStream \
  --env.CK_LOADGEN_MODE='--accuracy' \
  --env.CK_LOADGEN_EXTRA_PARAMS='--count 50' \
  \
  --env.CK_MODEL_PROFILE=tf_yolo \
  --dep_add_tags.weights=tf1-zoo,yolo-v3-coco \
  \
  --env.CK_INFERENCE_ENGINE=tensorflow \
  --env.CK_INFERENCE_ENGINE_BACKEND=default.cpu \
  --env.CUDA_VISIBLE_DEVICES=-1"
```

### b) Edit the environment and run a CK command

#### Create a container called `ck`

##### Without GPU support

```
docker run -td --entrypoint /bin/bash --name ck ${CK_IMAGE}
```

##### With GPU support
```
docker run -td --gpus all --entrypoint /bin/bash --name ck ${CK_IMAGE}
```

#### Start the container

```
docker exec -it ck /bin/bash
```

#### Stop the container
```
docker stop ck
```

#### Remove the container
```
docker rm ck
```

<!-- ## 2) Locally

### Repositories

```bash
$ ck pull repo:ck-object-detection --url=https://github.com/krai/ck-object-detection
$ ck pull repo:ck-tensorflow --url=https://github.com/krai/ck-tensorflow
```

### TensorFlow

Install from source:
```bash
$ ck install package:lib-tensorflow-1.10.1-src-{cpu,cuda}
```
or from a binary `x86_64` package:
```bash
$ ck install package:lib-tensorflow-1.10.1-{cpu,cuda}
```

Or you can choose from different available version of TensorFlow packages:
```bash
$ ck install package --tags=lib,tensorflow
```

### TensorFlow models
```bash
$ ck install ck-tensorflow:package:tensorflowmodel-api
```

Install one or more object detection model package:
```bash
$ ck install package --tags=object-detection,model,tf,tensorflow,tf1-zoo
```

### Datasets
```bash
$ ck install package --tags=dataset,object-detection
```

**NB:** If you have previously installed the `coco` dataset, you should probably renew them:
```bash
$ ck refresh env:{dataset-env-uoa}
```
where `dataset-env-uoa` is one of the env identifiers returned by:
```bash
$ ck show env --tags=dataset,coco
``` -->

---

# B) Running

General Form:
```
time docker run -it --rm ${CK_IMAGE} \
"ck run program:mlperf-inference-vision --cmd_key=direct --skip_print_timers \
  # Model_Specifications
  --dep_add_tags.weights=[TF_ZOO],[MODEL_NAME] \
  --env.CK_MODEL_PROFILE=[MODEL_PROFILE] \
  --env.CK_METRIC_TYPE=[DATA_TYPE] \

  # Backend_Specifications
  --env.CK_INFERENCE_ENGINE=tensorflow \
  --env.CK_INFERENCE_ENGINE_BACKEND=default.cpu \
  # Pass in relevant devices: CPU/GPU
  --env.CUDA_VISIBLE_DEVICES=-1 \

  # Scenario_Specifications
  --env.CK_LOADGEN_SCENARIO=Offline \

  # Mode_Specifications
  --env.CK_LOADGEN_MODE='--accuracy' \
  --env.CK_LOADGEN_EXTRA_PARAMS='--count 50'"
```


## 1. Specify a Model

Specify `--dep_add_tags.weights=[TF_ZOO],[MODEL_NAME]` and `--env.CK_MODEL_PROFILE=[MODEL_PROFILE]`.


### Supported `MODEL_NAME`/`MODEL_PROFILE` combinations

| `MODEL_NAME`                               |`TF_ZOO`   |`MODEL_PROFILE`      |
| ------------------------------------------ | ----------| --------------------|
|`rcnn-nas-lowproposals-coco`                | `tf1-zoo` |`tf1_object_det_zoo` |
|`rcnn-resnet50-lowproposals-coco`           | `tf1-zoo` |`tf1_object_det_zoo` |
|`rcnn-resnet101-lowproposals-coco`          | `tf1-zoo` |`tf1_object_det_zoo` |
|`rcnn-inception-resnet-v2-lowproposals-coco`| `tf1-zoo` |`tf1_object_det_zoo` |
|`rcnn-inception-v2-coco`                    | `tf1-zoo` |`tf1_object_det_zoo` |
|`ssd-inception-v2-coco`                     | `tf1-zoo` |`tf1_object_det_zoo` |
|`ssd_mobilenet_v1_coco`                     | `tf1-zoo` |`tf1_object_det_zoo` |
|`ssd_mobilenet_v1_quantized_coco`           | `tf1-zoo` |`tf1_object_det_zoo` |
|`ssd-mobilenet-v1-fpn-sbp-coco`             | `tf1-zoo` |`tf1_object_det_zoo` |
|`ssd-resnet50-v1-fpn-sbp-coco`              | `tf1-zoo` |`tf1_object_det_zoo` |
|`ssdlite-mobilenet-v2-coco`                 | `tf1-zoo` |`tf1_object_det_zoo` |
|`yolo-v3-coco`                              | `tf1-zoo` |`tf_yolo`            |
|`ssd_resnet50_v1_fpn_640x640`               | `tf2-zoo` |`tf2_object_det_zoo` |
|`ssd_resnet50_v1_fpn_1024x1024`             | `tf2-zoo` |`tf2_object_det_zoo` |
|`ssd_resnet101_v1_fpn_640x640`              | `tf2-zoo` |`tf2_object_det_zoo` |
|`ssd_resnet101_v1_fpn_1024x1024`            | `tf2-zoo` |`tf2_object_det_zoo` |
|`ssd_resnet152_v1_fpn_640x640`              | `tf2-zoo` |`tf2_object_det_zoo` |
|`ssd_resnet152_v1_fpn_1024x1024`            | `tf2-zoo` |`tf2_object_det_zoo` |
|`ssd_mobilenet_v2_320x320`                  | `tf2-zoo` |`tf2_object_det_zoo` |
|`ssd_mobilenet_v1_fpn_640x640`              | `tf2-zoo` |`tf2_object_det_zoo` |
|`ssd_mobilenet_v2_fpnlite_320x320`          | `tf2-zoo` |`tf2_object_det_zoo` |
|`ssd_mobilenet_v2_fpnlite_640x640`          | `tf2-zoo` |`tf2_object_det_zoo` |

### Examples
<details>
<summary>Click to expand</summary>

`tf1-zoo` model
```
time docker run -it --rm ${CK_IMAGE} \
"ck run program:mlperf-inference-vision --cmd_key=direct --skip_print_timers \
  --dep_add_tags.weights=tf1-zoo,ssdlite-mobilenet-v2-coco \
  --env.CK_MODEL_PROFILE=tf1_object_det_zoo \
  \
  --env.CK_LOADGEN_SCENARIO=SingleStream \
  --env.CK_LOADGEN_MODE='--accuracy' \
  --env.CK_LOADGEN_EXTRA_PARAMS='--count 50' \
  --env.CK_INFERENCE_ENGINE=tensorflow \
  --env.CK_INFERENCE_ENGINE_BACKEND=default.cpu \
  --env.CUDA_VISIBLE_DEVICES=-1"
```

`tf2-zoo` model
```
time docker run -it --rm ${CK_IMAGE} \
"ck run program:mlperf-inference-vision --cmd_key=direct --skip_print_timers \
  --dep_add_tags.weights=tf2-zoo,ssd_resnet50_v1_fpn_640x640 \
  --env.CK_MODEL_PROFILE=tf2_object_det_zoo \
  \
  --env.CK_LOADGEN_SCENARIO=SingleStream \
  --env.CK_LOADGEN_MODE='--accuracy' \
  --env.CK_LOADGEN_EXTRA_PARAMS='--count 50' \
  --env.CK_INFERENCE_ENGINE=tensorflow \
  --env.CK_INFERENCE_ENGINE_BACKEND=default.cpu \
  --env.CUDA_VISIBLE_DEVICES=-1"
```
</details>
<br>

## 2. Specify a Mode

The LoadGen mode can be selected by the environment variable `--env.CK_LOADGEN_MODE`. (When the mode is specified, it is `AccuracyOnly`; otherwise, it is `PerformanceOnly`.)

## Accuracy

For the Accuracy mode, you should specify the dataset size and the number of queries e.g. `--env.CK_LOADGEN_EXTRA_PARAMS='--count 200 --query-count 200'`.

```
time docker run -it --rm ${CK_IMAGE}
"ck run program:mlperf-inference-vision --cmd_key=direct --skip_print_timers \
  --env.CK_LOADGEN_MODE='--accuracy' \
  --env.CK_LOADGEN_EXTRA_PARAMS='--count 5000 --performance-sample-count 500' \
  \
  --dep_add_tags.weights=tf1-zoo,ssd_mobilenet_v1_coco \
  --env.CK_MODEL_PROFILE=tf1_object_det_zoo \
  --env.CK_INFERENCE_ENGINE=tensorflow \
  --env.CK_INFERENCE_ENGINE_BACKEND=default.cpu \
  --env.CUDA_VISIBLE_DEVICES=-1 \
  --env.CK_LOADGEN_SCENARIO=SingleStream"
```

## Performance

For the Performance mode, we recommended to specify `--env.CK_OPTIMIZE_GRAPH='True'`. You should also specify the dataset size, buffer size and the target QPS/ the target latency 

For `Offline` scenario, use target QPS to control how many queries Loadgen would generate:
```
time docker run -it --rm ${CK_IMAGE} \
"ck run program:mlperf-inference-vision --cmd_key=direct --skip_print_timers \
  --env.CK_LOADGEN_EXTRA_PARAMS='--count 50 --performance-sample-count 64 --query-count 2048 --target-latency 35' \
  --env.CK_LOADGEN_SCENARIO=Offline \
  --env.CK_OPTIMIZE_GRAPH='True' \
  \
  --dep_add_tags.weights=tf1-zoo,ssd_mobilenet_v1_coco \
  --env.CK_MODEL_PROFILE=tf1_object_det_zoo \
  --env.CK_INFERENCE_ENGINE=tensorflow \
  --env.CK_INFERENCE_ENGINE_BACKEND=default.cpu \
  --env.CUDA_VISIBLE_DEVICES=-1"
```

For `SingleStream` scenario, use target latency to control how many queries Loadgen would generate (unit of latency is in milliseconds):
```
time docker run -it --rm ${CK_IMAGE} \
"ck run program:mlperf-inference-vision --cmd_key=direct --skip_print_timers \
  --env.CK_LOADGEN_EXTRA_PARAMS='--count 50 --performance-sample-count 64 --query-count 2048 --target-latency 35' \
  --env.CK_LOADGEN_SCENARIO=SingleStream \
  --env.CK_OPTIMIZE_GRAPH='True' \
  \
  --dep_add_tags.weights=tf1-zoo,ssd_mobilenet_v1_coco \
  --env.CK_MODEL_PROFILE=tf1_object_det_zoo \
  --env.CK_INFERENCE_ENGINE=tensorflow \
  --env.CK_INFERENCE_ENGINE_BACKEND=default.cpu \
  --env.CUDA_VISIBLE_DEVICES=-1"
```
### Use a uniform target latency

Set up environment:

```
export CK_IMAGE_NAME=mlperf-inference-vision SDK_VER=21.08-py3 TF_VER=2.7.1
export CK_IMAGE="krai/${CK_IMAGE_NAME}:${SDK_VER}_tf-${TF_VER}"
export CK_EXPERIMENT_REPO="mlperf_v2.0.object-detection.$(hostname).$(id -un)"
export CK_EXPERIMENT_DIR="${HOME}/CK/${CK_EXPERIMENT_REPO}/experiment"
```

Create container:

```
CONTAINER_ID=`ck run cmdgen:benchmark.mlperf-inference-vision --docker=container_only \
--out=none  --library=tensorflow-v2.7.1-gpu --docker_image=${CK_IMAGE} --experiment_dir`
```

For `tf1-zoo` models:
```
time ck run cmdgen:benchmark.mlperf-inference-vision --verbose --sut=chai \
--model:=$(ck list_variations misc --query_module_uoa=package --tags=model,tf1-zoo --separator=:) \
--library=tensorflow-v2.7.1-gpu --device_ids=1 --scenario=singlestream --mode=performance \
--dataset_size=5000  --batch_size=1 \
--target_latency=200 \
--container=$CONTAINER_ID
```

### Estimate target latencies

```
time ck run cmdgen:benchmark.mlperf-inference-vision --verbose \
--sut=chai --model:=$(ck list_variations misc --query_module_uoa=package \
--tags=model,tf1-zoo --separator=:) --library=tensorflow-v2.7.1-gpu \
--device_ids=1 --scenario=range_singlestream --mode=performance --dataset_size=5000 \
--batch_size=1 --query_count=1024 --container=$CONTAINER_ID
```
Create target_latency.chai.txt

```
$(ck find program:generate-target-latency)/run.py \
--tags=inference_engine.tensorflow,inference_engine_version.v2.7.1 \
--repo_uoa=$CK_EXPERIMENT_REPO | sort \
| tee -a $(ck find program:mlperf-inference-vision)/target_latency.chai.txt
```

For `tf1-zoo` models:
```
time ck run cmdgen:benchmark.mlperf-inference-vision --verbose --sut=chai \
--model:=$(ck list_variations misc --query_module_uoa=package --tags=model,tf1-zoo --separator=:) \
--library=tensorflow-v2.7.1-gpu --device_ids=1 --scenario=singlestream --mode=performance \
--dataset_size=5000  --batch_size=1 \
--target_latency_file=/home/krai/CK_REPOS/ck-mlperf/program/mlperf-inference-vision/target_latency.chai.txt \
--container=$CONTAINER_ID
```
### Use a uniform target qps

Set up environment:

```
export CK_IMAGE_NAME=mlperf-inference-vision SDK_VER=22.01-py3 TF_VER=2.7.1
export CK_IMAGE="krai/${CK_IMAGE_NAME}:${SDK_VER}_tf-${TF_VER}"
export CK_EXPERIMENT_REPO="mlperf_v2.0.object-detection.$(hostname).$(id -un)"
export CK_EXPERIMENT_DIR="${HOME}/CK/${CK_EXPERIMENT_REPO}/experiment"
```

Create container:

```
CONTAINER_ID=`ck run cmdgen:benchmark.mlperf-inference-vision --docker=container_only \
--out=none  --library=tensorflow-v2.7.1-gpu --docker_image=${CK_IMAGE} --experiment_dir`
```

For "light" models (`target_latency` <= 80 in the `target_latency.chai.txt`):
```
time ck run cmdgen:benchmark.mlperf-inference-vision --verbose --sut=chai \
--model:=$($(ck find program:generate-target-qps)/get_light_models.py \
--tags=sut.chai,inference_engine.tensorflow,inference_engine_version.v2.7.1,inference_engine_backend.default.gpu \
--input_file=$(ck find program:mlperf-inference-vision)/target_latency.chai.txt) \
--library=tensorflow-v2.7.1-gpu --device_ids=1 --scenario=offline --mode=performance \
--dataset_size=5000 --batch_size=1 \
--target_qps=20 \
--container=$CONTAINER_ID
```
### Estimate target qps and batch size

Run `cmdgen:benchmark.mlperf-inference-visio` with 

`--query_count=4096` for `default.gpu` or `--query_count=1024` for `default.cpu` and `openvino.cpu`

`--scenario=range_offline`, 

`--batch_size,=1,4,8,16` for `default.gpu` or `--batch_size,=1,2,4,8` for `default.cpu`, `openvino.cpu`.

```
time ck run cmdgen:benchmark.mlperf-inference-vision --verbose \
--sut=chai --model:=$($(ck find program:generate-target-qps)/get_light_models.py \
--tags=sut.chai,inference_engine.tensorflow,inference_engine_version.v2.7.1,inference_engine_backend.default.gpu \
--input_file=$(ck find program:mlperf-inference-vision)/target_latency.chai.txt) \
--library=tensorflow-v2.7.1-gpu \
--device_ids=1 --scenario=range_offline --mode=performance --dataset_size=5000 \
--batch_size,=1,4,8,16 --query_count=4096 --container=$CONTAINER_ID
```

Create `target_qps.chai.txt`

```
$(ck find program:generate-target-qps)/run.py \
--tags=inference_engine.tensorflow,inference_engine_version.v2.7.1 \
--repo_uoa=$CK_EXPERIMENT_REPO | sort \
| tee -a $(ck find program:mlperf-inference-vision)/target_qps.chai.txt
```

#### Examples
<details>
<summary>Click to expand</summary>

```
maria@chai:~$ cat $(ck find program:mlperf-inference-vision)/target_qps.chai.txt
chai,tensorflow-v2.7.1-default.gpu,rcnn-inception-v2-coco,batch_size=1,20
chai,tensorflow-v2.7.1-default.gpu,rcnn-inception-v2-coco,batch_size=16,25
chai,tensorflow-v2.7.1-default.gpu,rcnn-inception-v2-coco,batch_size=4,24
chai,tensorflow-v2.7.1-default.gpu,rcnn-inception-v2-coco,batch_size=8,24
chai,tensorflow-v2.7.1-default.gpu,rcnn-resnet50-lowproposals-coco,batch_size=1,23
chai,tensorflow-v2.7.1-default.gpu,rcnn-resnet50-lowproposals-coco,batch_size=16,32
chai,tensorflow-v2.7.1-default.gpu,rcnn-resnet50-lowproposals-coco,batch_size=4,27
chai,tensorflow-v2.7.1-default.gpu,rcnn-resnet50-lowproposals-coco,batch_size=8,28
chai,tensorflow-v2.7.1-default.gpu,ssd-inception-v2-coco,batch_size=1,21
chai,tensorflow-v2.7.1-default.gpu,ssd-inception-v2-coco,batch_size=16,35
chai,tensorflow-v2.7.1-default.gpu,ssd-inception-v2-coco,batch_size=4,32
chai,tensorflow-v2.7.1-default.gpu,ssd-inception-v2-coco,batch_size=8,34
chai,tensorflow-v2.7.1-default.gpu,ssdlite-mobilenet-v2-coco,batch_size=1,25
chai,tensorflow-v2.7.1-default.gpu,ssdlite-mobilenet-v2-coco,batch_size=16,39
chai,tensorflow-v2.7.1-default.gpu,ssdlite-mobilenet-v2-coco,batch_size=4,36
chai,tensorflow-v2.7.1-default.gpu,ssdlite-mobilenet-v2-coco,batch_size=8,39
chai,tensorflow-v2.7.1-default.gpu,ssd_mobilenet_v1_coco,batch_size=1,28
chai,tensorflow-v2.7.1-default.gpu,ssd_mobilenet_v1_coco,batch_size=16,40
chai,tensorflow-v2.7.1-default.gpu,ssd_mobilenet_v1_coco,batch_size=4,36
chai,tensorflow-v2.7.1-default.gpu,ssd_mobilenet_v1_coco,batch_size=8,39
chai,tensorflow-v2.7.1-default.gpu,ssd-mobilenet-v1-fpn-sbp-coco,batch_size=1,18
chai,tensorflow-v2.7.1-default.gpu,ssd-mobilenet-v1-fpn-sbp-coco,batch_size=16,33
chai,tensorflow-v2.7.1-default.gpu,ssd-mobilenet-v1-fpn-sbp-coco,batch_size=4,32
chai,tensorflow-v2.7.1-default.gpu,ssd-mobilenet-v1-fpn-sbp-coco,batch_size=8,33
chai,tensorflow-v2.7.1-default.gpu,ssd-resnet50-v1-fpn-sbp-coco,batch_size=1,14
chai,tensorflow-v2.7.1-default.gpu,ssd-resnet50-v1-fpn-sbp-coco,batch_size=16,27
chai,tensorflow-v2.7.1-default.gpu,ssd-resnet50-v1-fpn-sbp-coco,batch_size=4,25
chai,tensorflow-v2.7.1-default.gpu,ssd-resnet50-v1-fpn-sbp-coco,batch_size=8,27
chai,tensorflow-v2.7.1-default.gpu,yolo-v3-coco,batch_size=1,31
chai,tensorflow-v2.7.1-default.gpu,yolo-v3-coco,batch_size=16,64
chai,tensorflow-v2.7.1-default.gpu,yolo-v3-coco,batch_size=4,71
chai,tensorflow-v2.7.1-default.gpu,yolo-v3-coco,batch_size=8,68
```
  
</details>
<br>
  
For "light" models:

1. `batch_size` is presented in `$(ck find program:mlperf-inference-vision)/target_qps.chai.txt`. 

`target_qps` is set to a value  that corresponds to a given `batch_size` for each model.

```
time ck run cmdgen:benchmark.mlperf-inference-vision --verbose \
--sut=chai --model:=$($(ck find program:generate-target-qps)/get_light_models.py \
--tags=sut.chai,inference_engine.tensorflow,inference_engine_version.v2.7.1,inference_engine_backend.default.gpu \
--input_file=$(ck find program:mlperf-inference-vision)/target_latency.chai.txt) \
--library=tensorflow-v2.7.1-gpu \
--device_ids=1 --scenario=offline --mode=performance --dataset_size=5000 \
--batch_size=4 \
--target_qps_file=/home/krai/CK_REPOS/ck-mlperf/program/mlperf-inference-vision/target_qps.chai.txt \
--container=$CONTAINER_ID
```

#### Examples
<details>
<summary>Click to expand</summary>
  
```
maria@chai:~/CK/ck-mlperf/program/mlperf-inference-vision$ time ck run cmdgen:benchmark.mlperf-inference-vision --verbose \
--sut=chai --model=ssd_mobilenet_v1_coco --library=tensorflow-v2.7.1-gpu --device_ids=1 --scenario=offline --mode=performance \
--dataset_size=5000 --batch_size=4 \
--target_qps_file=/home/krai/CK_REPOS/ck-mlperf/program/mlperf-inference-vision/target_qps.chai.txt \
--container=$CONTAINER_ID
...
export CK_BATCH_SIZE=4
export CK_CPU_FREQUENCY=max
export CK_GPU_FREQUENCY=max
export CK_INFERENCE_ENGINE=tensorflow
export CK_INFERENCE_ENGINE_BACKEND=default.gpu
export CK_LOADGEN_BUFFER_SIZE=256
export CK_LOADGEN_DATASET_SIZE=5000
export CK_LOADGEN_EXTRA_PARAMS="--count 5000 --qps $CK_LOADGEN_TARGET_QPS --performance-sample-count 256 --cache 0"                                                     export CK_LOADGEN_MODE=
export CK_LOADGEN_SCENARIO=Offline
export CK_LOADGEN_TARGET_QPS=36
...
================================================                                                                                                                        MLPerf Results Summary
================================================
SUT name : PySUT
Scenario : Offline
Mode     : PerformanceOnly
Samples per second: 35.0436
Result is : VALID
  Min duration satisfied : Yes
  Min queries satisfied : Yes
  Early stopping satisfied: Yes

================================================
Additional Stats
================================================
Min latency (ns)                : 16063930859
Max latency (ns)                : 701298703761
Mean latency (ns)               : 360012787583
50.00 percentile latency (ns)   : 360008370165
90.00 percentile latency (ns)   : 636558727329
95.00 percentile latency (ns)   : 671297709534
97.00 percentile latency (ns)   : 684586117475
99.00 percentile latency (ns)   : 699137120608
99.90 percentile latency (ns)   : 701224657563

================================================
Test Parameters Used
================================================
samples_per_query : 24576
target_qps : 36
target_latency (ns): 0
max_async_queries : 1
min_duration (ms): 600000
max_duration (ms): 0
min_query_count : 1
max_query_count : 0
qsl_rng_seed : 6655344265603136530
sample_index_rng_seed : 15863379492028895792
schedule_rng_seed : 12662793979680847247
accuracy_log_rng_seed : 0
accuracy_log_probability : 0
accuracy_log_sampling_target : 0
print_timestamps : 0
performance_issue_unique : 0
performance_issue_same : 0
performance_issue_same_index : 0
performance_sample_count : 256

No warnings encountered during test.

No errors encountered during test.
0.11user 0.09system 12:05.67elapsed 0%CPU (0avgtext+0avgdata 59236maxresident)k
  0inputs+0outputs (0major+7649minor)pagefaults 0swaps
==========================================================================================

real    12m5.782s
user    0m0.206s
sys     0m0.109s
```
  
</details>
<br>

2. `batch_size` is not presented in `$(ck find program:mlperf-inference-vision)/target_qps.chai.txt`. 

`target_qps` is set to a maximum value for each model.

```
time ck run cmdgen:benchmark.mlperf-inference-vision --verbose \
--sut=chai --model:=$($(ck find program:generate-target-qps)/get_light_models.py \
--tags=sut.chai,inference_engine.tensorflow,inference_engine_version.v2.7.1,inference_engine_backend.default.gpu \
--input_file=$(ck find program:mlperf-inference-vision)/target_latency.chai.txt) \
--library=tensorflow-v2.7.1-gpu \
--device_ids=1 --scenario=offline --mode=performance --dataset_size=5000 \
--batch_size=3 \
--target_qps_file=/home/krai/CK_REPOS/ck-mlperf/program/mlperf-inference-vision/target_qps.chai.txt \
--container=$CONTAINER_ID
```
#### Examples
<details>
<summary>Click to expand</summary>
  
```
maria@chai:~/CK/ck-mlperf/program/mlperf-inference-vision$ time ck run cmdgen:benchmark.mlperf-inference-vision --verbose \
--sut=chai --model=ssd_mobilenet_v1_coco --library=tensorflow-v2.7.1-gpu --device_ids=1 --scenario=offline --mode=performance \
--dataset_size=5000 --batch_size=3 \
--target_qps_file=/home/krai/CK_REPOS/ck-mlperf/program/mlperf-inference-vision/target_qps.chai.txt \
--container=$CONTAINER_ID
...
export CK_BATCH_SIZE=3                                                                                                                                                  export CK_CPU_FREQUENCY=max
export CK_GPU_FREQUENCY=max
export CK_INFERENCE_ENGINE=tensorflow
export CK_INFERENCE_ENGINE_BACKEND=default.gpu                                                                                                                          export CK_LOADGEN_BUFFER_SIZE=256
export CK_LOADGEN_DATASET_SIZE=5000
export CK_LOADGEN_EXTRA_PARAMS="--count 5000 --qps $CK_LOADGEN_TARGET_QPS --performance-sample-count 256 --cache 0"                                                     export CK_LOADGEN_MODE=
export CK_LOADGEN_SCENARIO=Offline
export CK_LOADGEN_TARGET_QPS=40
...
MLPerf Results Summary
================================================
SUT name : PySUT
Scenario : Offline
Mode     : PerformanceOnly
Samples per second: 33.4703
Result is : VALID
  Min duration satisfied : Yes
  Min queries satisfied : Yes
  Early stopping satisfied: Yes

================================================
Additional Stats
================================================
Min latency (ns)                : 14834928607
Max latency (ns)                : 788759960558
Mean latency (ns)               : 403001855289
50.00 percentile latency (ns)   : 402672669788
90.00 percentile latency (ns)   : 714740129370
95.00 percentile latency (ns)   : 753728258386
97.00 percentile latency (ns)   : 768572763858
99.00 percentile latency (ns)   : 784403814732
99.90 percentile latency (ns)   : 788626111655

================================================
Test Parameters Used
================================================
samples_per_query : 26400
target_qps : 40
target_latency (ns): 0
max_async_queries : 1
min_duration (ms): 600000
max_duration (ms): 0
min_query_count : 1
max_query_count : 0
qsl_rng_seed : 6655344265603136530
sample_index_rng_seed : 15863379492028895792
schedule_rng_seed : 12662793979680847247
accuracy_log_rng_seed : 0
accuracy_log_probability : 0
accuracy_log_sampling_target : 0
print_timestamps : 0
performance_issue_unique : 0
performance_issue_same : 0
performance_issue_same_index : 0
performance_sample_count : 256

No warnings encountered during test.

No errors encountered during test.
0.12user 0.09system 13:33.06elapsed 0%CPU (0avgtext+0avgdata 59212maxresident)k
0inputs+0outputs (0major+7665minor)pagefaults 0swaps
==========================================================================================

real    13m33.166s
user    0m0.211s
sys     0m0.117s 
```
  
</details>
<br>

3. `batch_size` is not set.

`target_qps` is set to a maximum value  and `batch_size` is set to a value that corresponds to a maximum `target_qps` in `$(ck find program:mlperf-inference-vision)/target_qps.chai.txt` for each model. 

```
time ck run cmdgen:benchmark.mlperf-inference-vision --verbose \
--sut=chai --model:=$($(ck find program:generate-target-qps)/get_light_models.py \
--tags=sut.chai,inference_engine.tensorflow,inference_engine_version.v2.7.1,inference_engine_backend.default.gpu \
--input_file=$(ck find program:mlperf-inference-vision)/target_latency.chai.txt) \
--library=tensorflow-v2.7.1-gpu \
--device_ids=1 --scenario=offline --mode=performance --dataset_size=5000 \
--target_qps_file=/home/krai/CK_REPOS/ck-mlperf/program/mlperf-inference-vision/target_qps.chai.txt \
--container=$CONTAINER_ID
```

#### Examples
<details>
<summary>Click to expand</summary>
  
```
maria@chai:~/CK/ck-mlperf/program/mlperf-inference-vision$ time ck run cmdgen:benchmark.mlperf-inference-vision --verbose \
--sut=chai --model=ssd_mobilenet_v1_coco --library=tensorflow-v2.7.1-gpu --device_ids=1 --scenario=offline --mode=performance \
--dataset_size=5000 --target_qps_file=/home/krai/CK_REPOS/ck-mlperf/program/mlperf-inference-vision/target_qps.chai.txt \
--container=$CONTAINER_ID
...
  
export CK_BATCH_SIZE=16
export CK_CPU_FREQUENCY=max
export CK_GPU_FREQUENCY=max
export CK_INFERENCE_ENGINE=tensorflow
export CK_INFERENCE_ENGINE_BACKEND=default.gpu
export CK_LOADGEN_BUFFER_SIZE=256
export CK_LOADGEN_DATASET_SIZE=5000
export CK_LOADGEN_EXTRA_PARAMS="--count 5000 --qps $CK_LOADGEN_TARGET_QPS --performance-sample-count 256 --cache 0"
export CK_LOADGEN_MODE=
export CK_LOADGEN_SCENARIO=Offline
export CK_LOADGEN_TARGET_QPS=40   
...
================================================  
MLPerf Results Summary
================================================
SUT name : PySUT
Scenario : Offline
Mode     : PerformanceOnly
Samples per second: 38.9646
Result is : VALID
  Min duration satisfied : Yes
  Min queries satisfied : Yes
  Early stopping satisfied: Yes

================================================
Additional Stats
================================================
Min latency (ns)                : 28058236730
Max latency (ns)                : 677537944945
Mean latency (ns)               : 358745212845
50.00 percentile latency (ns)   : 362242670288
90.00 percentile latency (ns)   : 626526519749
95.00 percentile latency (ns)   : 660320060012
97.00 percentile latency (ns)   : 671160987618
99.00 percentile latency (ns)   : 676811114462
99.90 percentile latency (ns)   : 677535089893

================================================
Test Parameters Used
================================================
samples_per_query : 26400
target_qps : 40
target_latency (ns): 0
max_async_queries : 1
min_duration (ms): 600000
max_duration (ms): 0
min_query_count : 1
max_query_count : 0
qsl_rng_seed : 6655344265603136530
sample_index_rng_seed : 15863379492028895792
schedule_rng_seed : 12662793979680847247
accuracy_log_rng_seed : 0
accuracy_log_probability : 0
accuracy_log_sampling_target : 0
print_timestamps : 0
performance_issue_unique : 0
performance_issue_same : 0
performance_issue_same_index : 0
performance_sample_count : 256

No warnings encountered during test.

No errors encountered during test.
0.12user 0.06system 11:42.49elapsed 0%CPU (0avgtext+0avgdata 59128maxresident)k
0inputs+0outputs (0major+7689minor)pagefaults 0swaps
==========================================================================================

real    11m42.598s
user    0m0.223s
sys     0m0.078s  
```
  
</details>
<br>

## 3. Specify a Scenario

You can specify the scenario with the `--env.CK_LOADGEN_SCENARIO` environment variable.

|SCENARIO|
|---|
| `SingleStream`, `MultiStream`, `Server`, `Offline` |

```
time docker run -it --rm ${CK_IMAGE} \
"ck run program:mlperf-inference-vision --cmd_key=direct --skip_print_timers \
  --env.CK_LOADGEN_SCENARIO=[SCENARIO] \
  \
  --env.CK_LOADGEN_MODE='--accuracy' \
  --env.CK_LOADGEN_EXTRA_PARAMS='--count 5000 --performance-sample-count 500' \
  --dep_add_tags.weights=tf1-zoo,ssd_mobilenet_v1_coco \
  --env.CK_MODEL_PROFILE=tf1_object_det_zoo \
  --env.CK_INFERENCE_ENGINE=tensorflow \
  --env.CK_INFERENCE_ENGINE_BACKEND=default.cpu \
  --env.CUDA_VISIBLE_DEVICES=-1"
```

## Batch Size in Offline Mode

The batch size is 1 by default. You can experiment with `CK_BATCH_SIZE` in the `Offline` scenario:

Using the batch size of 32 under the `Accuracy` mode and `Offline` scenario:
```
time docker run -it --rm ${CK_IMAGE} \
"ck run program:mlperf-inference-vision --cmd_key=direct --skip_print_timers \
  --env.CK_BATCH_SIZE=32 \
  --env.CK_LOADGEN_MODE='--accuracy' \
  --env.CK_LOADGEN_EXTRA_PARAMS='--count 5000 --performance-sample-count 500' \
  --env.CK_LOADGEN_SCENARIO=Offline \
  \
  --dep_add_tags.weights=tf1-zoo,ssd_mobilenet_v1_coco \
  --env.CK_MODEL_PROFILE=tf1_object_det_zoo \
  --env.CK_INFERENCE_ENGINE=tensorflow \
  --env.CK_INFERENCE_ENGINE_BACKEND=default.cpu \
  --env.CUDA_VISIBLE_DEVICES=-1"
```

Using the batch size of 32 under the `Performance` mode and `Offline` scenario:
```
time docker run -it --rm ${CK_IMAGE} \
"ck run program:mlperf-inference-vision --cmd_key=direct --skip_print_timers \
--env.CK_BATCH_SIZE=32 \
--env.CK_LOADGEN_EXTRA_PARAMS='--count 50 --performance-sample-count 64 --query-count 2048 --qps 3' \
--env.CK_OPTIMIZE_GRAPH='True' \
--env.CK_LOADGEN_SCENARIO=Offline \
\
--dep_add_tags.weights=tf1-zoo,ssd_mobilenet_v1_coco \
--env.CK_MODEL_PROFILE=tf1_object_det_zoo \
--env.CK_INFERENCE_ENGINE=tensorflow \
--env.CK_INFERENCE_ENGINE_BACKEND=default.cpu \
--env.CUDA_VISIBLE_DEVICES=-1"
```

## 4. Summary of Scenario and Mode combinations

|SCENARIO| MODE | CK_LOADGEN_EXTRA_PARAMS | CK_BATCH_SIZE |
|---|--- |--- |--- |
| `SingleStream`| Accuracy | `--count 5000 --performance-sample-count 500` | / |
| `SingleStream`| Performance | `--count 50 --performance-sample-count 64 --query-count 2048 --target-latency 35` | / |
| `Offline` | Accuracy | `--count 5000 --performance-sample-count 500` | avaliable |
| `Offline` | Performance | `--count 50 --performance-sample-count 64 --query-count 2048 --qps 3` | avaliable |

### Examples
<details>
<summary>Click to expand</summary>

#### Offline - Accuracy - Batch Size 1
```
time docker run -it --rm ${CK_IMAGE} \
"ck run program:mlperf-inference-vision --cmd_key=direct --skip_print_timers \
  --env.CK_LOADGEN_MODE='--accuracy' \
  --env.CK_LOADGEN_EXTRA_PARAMS='--count 5000 --performance-sample-count 500' \
  --env.CK_LOADGEN_SCENARIO=Offline \
  \
  --dep_add_tags.weights=tf1-zoo,ssd_mobilenet_v1_coco \
  --env.CK_MODEL_PROFILE=tf1_object_det_zoo \
  --env.CK_INFERENCE_ENGINE=tensorflow \
  --env.CK_INFERENCE_ENGINE_BACKEND=default.cpu \
  --env.CUDA_VISIBLE_DEVICES=-1"
```
#### Offline - Performance - Batch Size 1
```
time docker run -it --rm ${CK_IMAGE} \
"ck run program:mlperf-inference-vision --cmd_key=direct --skip_print_timers \
  --env.CK_LOADGEN_EXTRA_PARAMS='--count 50 --performance-sample-count 64 --query-count 2048 --qps 3' \
  --env.CK_OPTIMIZE_GRAPH='True' \
  --env.CK_LOADGEN_SCENARIO=Offline \
  \
  --dep_add_tags.weights=tf1-zoo,ssd_mobilenet_v1_coco \
  --env.CK_MODEL_PROFILE=tf1_object_det_zoo \
  --env.CK_INFERENCE_ENGINE=tensorflow \
  --env.CK_INFERENCE_ENGINE_BACKEND=default.cpu \
  --env.CUDA_VISIBLE_DEVICES=-1"
```
#### Single Stream - Accuracy - Batch Size 1
```
time docker run -it --rm ${CK_IMAGE} \
"ck run program:mlperf-inference-vision --cmd_key=direct --skip_print_timers \
  --env.CK_LOADGEN_MODE='--accuracy' \
  --env.CK_LOADGEN_EXTRA_PARAMS='--count 5000 --performance-sample-count 500' \
  --env.CK_LOADGEN_SCENARIO=SingleStream \
  \
  --dep_add_tags.weights=tf1-zoo,ssd_mobilenet_v1_coco \
  --env.CK_MODEL_PROFILE=tf1_object_det_zoo \
  --env.CK_INFERENCE_ENGINE=tensorflow \
  --env.CK_INFERENCE_ENGINE_BACKEND=default.cpu \
  --env.CUDA_VISIBLE_DEVICES=-1"
```
#### Single Stream - Performance - Batch Size 1
```
time docker run -it --rm ${CK_IMAGE} \
"ck run program:mlperf-inference-vision --cmd_key=direct --skip_print_timers \
  --env.CK_LOADGEN_EXTRA_PARAMS='--count 50 --performance-sample-count 64 --query-count 2048 --target-latency 35' \
  --env.CK_OPTIMIZE_GRAPH='True' \
  --env.CK_LOADGEN_SCENARIO=SingleStream \
  \
  --dep_add_tags.weights=tf1-zoo,ssd_mobilenet_v1_coco \
  --env.CK_MODEL_PROFILE=tf1_object_det_zoo \
  --env.CK_INFERENCE_ENGINE=tensorflow \
  --env.CK_INFERENCE_ENGINE_BACKEND=default.cpu \
  --env.CUDA_VISIBLE_DEVICES=-1"
```
#### Offline - Accuracy - Batch Size 32
```
time docker run -it --rm ${CK_IMAGE} \
"ck run program:mlperf-inference-vision --cmd_key=direct --skip_print_timers \
  --env.CK_BATCH_SIZE=32 \
  --env.CK_LOADGEN_MODE='--accuracy' \
  --env.CK_LOADGEN_EXTRA_PARAMS='--count 5000 --performance-sample-count 500' \
  --env.CK_LOADGEN_SCENARIO=Offline \
  \
  --dep_add_tags.weights=tf1-zoo,ssd_mobilenet_v1_coco \
  --env.CK_MODEL_PROFILE=tf1_object_det_zoo \
  --env.CK_INFERENCE_ENGINE=tensorflow \
  --env.CK_INFERENCE_ENGINE_BACKEND=default.cpu \
  --env.CUDA_VISIBLE_DEVICES=-1"
```
#### Offline - Performance - Batch Size 32
```
time docker run -it --rm ${CK_IMAGE} \
"ck run program:mlperf-inference-vision --cmd_key=direct --skip_print_timers \
  --env.CK_BATCH_SIZE=32 \
  --env.CK_LOADGEN_EXTRA_PARAMS='--count 50 --performance-sample-count 64 --query-count 2048 --qps 3' \
  --env.CK_OPTIMIZE_GRAPH='True' \
  --env.CK_LOADGEN_SCENARIO=Offline \
  \
  --dep_add_tags.weights=tf1-zoo,ssd_mobilenet_v1_coco \
  --env.CK_MODEL_PROFILE=tf1_object_det_zoo \
  --env.CK_INFERENCE_ENGINE=tensorflow \
  --env.CK_INFERENCE_ENGINE_BACKEND=default.cpu \
  --env.CUDA_VISIBLE_DEVICES=-1"
```
</details>
<br>

## 5. Graph Optimization

Use the environment variable `--env.CK_OPTIMIZE_GRAPH` to configure whether to optimize the model graph for execution (default: `False`).

We recommended it to be set to `False` when running in the Accuracy mode, and to `True` when running in the Performance mode.

```
time docker run -it --rm ${CK_IMAGE} \
"ck run program:mlperf-inference-vision --cmd_key=direct --skip_print_timers \
  --env.CK_OPTIMIZE_GRAPH='True' \
  \
  --env.CK_LOADGEN_MODE='--accuracy' \
  --env.CK_LOADGEN_EXTRA_PARAMS='--count 5000 --performance-sample-count 500' \
  --dep_add_tags.weights=tf1-zoo,yolo-v3-coco \
  --env.CK_MODEL_PROFILE=tf_yolo \
  --env.CK_INFERENCE_ENGINE=tensorflow \
  --dep_add_tags.inference-engine-backend=default.cpu \
  --env.CUDA_VISIBLE_DEVICES=-1 \
  --env.CK_LOADGEN_SCENARIO=SingleStream"
```


## 6. Select an Engine/Backend/Device

### Supported `INFERENCE_ENGINE`/`INFERENCE_ENGINE_BACKEND`/`CUDA_VISIBLE_DEVICES` combinations

| `INFERENCE_ENGINE` | `INFERENCE_ENGINE_BACKEND`  | `CUDA_VISIBLE_DEVICES`       |
| ------------------ | --------------------------- | ---------------------------- |
| `tensorflow`       | `default.cpu`               | `-1`                         |
| `tensorflow`       | `default.gpu`               | `<device_id>`                |
| `tensorflow`       | `tensorrt-dynamic`          | `<device_id>`                |
| `tensorflow`       | `openvino.cpu`              | `-1`                         |
| `tensorflow`       | `openvino.gpu` (not tested) | `-1` (integrated Intel GPU)  |
| `tensorflow`       | `openvino.gpu` (not tested) | `0` (discreet Intel GPU)     |

### Examples
<details>
<summary>Click to expand</summary>

#### `tensorflow/default-cpu/-1`
```
time docker run -it --rm ${CK_IMAGE} \
"ck run program:mlperf-inference-vision --cmd_key=direct --skip_print_timers \
  --env.CK_INFERENCE_ENGINE=tensorflow \
  --env.CK_INFERENCE_ENGINE_BACKEND=default.cpu \
  --env.CUDA_VISIBLE_DEVICES=-1 \
  \
  --env.CK_LOADGEN_SCENARIO=SingleStream \
  --env.CK_LOADGEN_MODE='--accuracy' \
  --env.CK_LOADGEN_EXTRA_PARAMS='--count 5000 --performance-sample-count 500' \
  \
  --env.CK_MODEL_PROFILE=tf1_object_det_zoo \
  --dep_add_tags.weights=tf1-zoo,rcnn-inception-v2-coco"
```

#### `tensorflow/default-gpu/0`
```
time docker run --gpus all -it --rm ${CK_IMAGE} \
"ck run program:mlperf-inference-vision --cmd_key=direct --skip_print_timers \
  --env.CK_INFERENCE_ENGINE=tensorflow \
  --env.CK_INFERENCE_ENGINE_BACKEND=default.gpu \
  --env.CUDA_VISIBLE_DEVICES=0 \
  \
  --env.CK_LOADGEN_SCENARIO=SingleStream \
  --env.CK_LOADGEN_MODE='--accuracy' \
  --env.CK_LOADGEN_EXTRA_PARAMS='--count 5000 --performance-sample-count 500' \
  \
  --env.CK_MODEL_PROFILE=tf1_object_det_zoo \
  --dep_add_tags.weights=tf1-zoo,rcnn-inception-v2-coco"
```

#### `tensorflow/tensorrt-dynamic/0`
```
time docker run --gpus all -it --rm ${CK_IMAGE} \
"ck run program:mlperf-inference-vision --cmd_key=direct --skip_print_timers \
  --env.CK_INFERENCE_ENGINE=tensorflow \
  --env.CK_INFERENCE_ENGINE_BACKEND=tensorrt-dynamic\
  --env.CUDA_VISIBLE_DEVICES=0 \
  \
  --env.CK_LOADGEN_SCENARIO=SingleStream \
  --env.CK_LOADGEN_MODE='--accuracy' \
  --env.CK_LOADGEN_EXTRA_PARAMS='--count 5000 --performance-sample-count 500' \
  \
  --env.CK_MODEL_PROFILE=tf2_object_det_zoo \
  --dep_add_tags.weights=tf2-zoo,ssd_resnet50_v1_fpn_640x640"
```

#### `tensorflow/openvino-cpu/-1`

```
time docker run -it --rm ${CK_IMAGE} \
"ck run program:mlperf-inference-vision --cmd_key=direct --skip_print_timers \
  --env.CK_INFERENCE_ENGINE=tensorflow \
  --env.CK_INFERENCE_ENGINE_BACKEND=openvino.cpu \
  --env.CUDA_VISIBLE_DEVICES=-1 \
  \
  --env.CK_LOADGEN_SCENARIO=SingleStream \
  --env.CK_LOADGEN_MODE='--accuracy' \
  --env.CK_LOADGEN_EXTRA_PARAMS='--count 5000 --performance-sample-count 500' \
  \
  --env.CK_MODEL_PROFILE=tf1_object_det_zoo \
  --dep_add_tags.weights=tf1-zoo,rcnn-inception-v2-coco"
```

#### `tensorflow/openvino-gpu/-1` (not tested)

If the machine has an Intel chip with an integrated GPU, set `--env.CUDA_VISIBLE_DEVICES=-1`:
```
time docker run -it --rm ${CK_IMAGE} \
"ck run program:mlperf-inference-vision --cmd_key=direct --skip_print_timers \
  --env.CK_INFERENCE_ENGINE=tensorflow \
  --env.CK_INFERENCE_ENGINE_BACKEND=openvino.gpu \
  --env.CUDA_VISIBLE_DEVICES=-1 \
  \
  --env.CK_LOADGEN_SCENARIO=SingleStream \
  --env.CK_LOADGEN_MODE='--accuracy' \
  --env.CK_LOADGEN_EXTRA_PARAMS='--count 5000 --performance-sample-count 500' \
  \
  --env.CK_MODEL_PROFILE=tf1_object_det_zoo \
  --dep_add_tags.weights=tf1-zoo,rcnn-inception-v2-coco"
```

#### `tensorflow/openvino-gpu/0` (not tested)

If the machine has a discreet Intel GPU, set `--env.CUDA_VISIBLE_DEVICES=0`:
```
time docker run -it --rm ${CK_IMAGE} \
"ck run program:mlperf-inference-vision --cmd_key=direct --skip_print_timers \
  --env.CK_INFERENCE_ENGINE=tensorflow \
  --env.CK_INFERENCE_ENGINE_BACKEND=openvino.gpu \
  --env.CUDA_VISIBLE_DEVICES=0 \
  \
  --env.CK_LOADGEN_SCENARIO=SingleStream \
  --env.CK_LOADGEN_MODE='--accuracy' \
  --env.CK_LOADGEN_EXTRA_PARAMS='--count 5000 --performance-sample-count 500' \
  \
  --env.CK_MODEL_PROFILE=tf1_object_det_zoo \
  --dep_add_tags.weights=tf1-zoo,rcnn-inception-v2-coco"
```
</details>
<br>

# C) TO DO
- Consider to support SSD models to use `openvino` inference engine backend ([link](https://github.com/openvinotoolkit/openvino_tensorflow/issues/201))
- Models from TF2 object detection zoo have issues running in `Offline` scenario. 
- Extend supports for `openvino-gpu` inference engine backend
