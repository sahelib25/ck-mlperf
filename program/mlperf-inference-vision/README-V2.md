# MLPerf Inference Vision - extended for Object Detection

This Collective Knowledge workflow is based on the [official MLPerf Inference Vision application](https://github.com/mlcommons/inference/tree/master/vision/classification_and_detection) extended for diverse Object Detection models, as found e.g. in the [TF1 Object Detection Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md) and the [TF2 Object Detection Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md).

The table below shows currently supported models, frameworks ("inference engines") and library/device combinations ("inference engine backends").



| `MODEL_NAME`                                 | `INFERENCE_ENGINE`  | `INFERENCE_ENGINE_BACKEND`                 |
| -------------------------------------------- | ------------------- | ------------------------------------------ |
| `rcnn-nas-lowproposals-coco`                 | `tensorflow`        | `default-cpu`,`default-gpu`,`openvino-cpu` |
| `rcnn-resnet50-lowproposals-coco`            | `tensorflow`        | `default-cpu`,`default-gpu`,`openvino-cpu` |
| `rcnn-resnet101-lowproposals-coco`           | `tensorflow`        | `default-cpu`,`default-gpu`,`openvino-cpu` |
| `rcnn-inception-resnet-v2-lowproposals-coco` | `tensorflow`        | `default-cpu`,`default-gpu`,`openvino-cpu` |
| `rcnn-inception-v2-coco`                     | `tensorflow`        | `default-cpu`,`default-gpu`,`openvino-cpu` |
| `ssd-inception-v2-coco`                      | `tensorflow`        | `default-cpu`,`default-gpu`                |
| `ssd_mobilenet_v1_coco`                      | `tensorflow`        | `default-cpu`,`default-gpu`                |
| `ssd_mobilenet_v1_quantized_coco`            | `tensorflow`        | `default-cpu`,`default-gpu`                |
| `ssd-mobilenet-v1-fpn-sbp-coco`              | `tensorflow`        | `default-cpu`,`default-gpu`                |
| `ssd-resnet50-v1-fpn-sbp-coco`               | `tensorflow`        | `default-cpu`,`default-gpu`                |
| `ssdlite-mobilenet-v2-coco`                  | `tensorflow`        | `default-cpu`,`default-gpu`                |
| `yolo-v3-coco`                               | `tensorflow`        | `default-cpu`,`default-gpu`,`openvino-cpu` |
| `ssd_resnet50_v1_fpn_640x640`                | `tensorflow`        | `default-cpu`,`default-gpu`                |
| `ssd_resnet50_v1_fpn_1024x1024`              | `tensorflow`        | `default-cpu`,`default-gpu`                |
|`ssd_resnet101_v1_fpn_640x640`                | `tensorflow`        | `default-cpu`,`default-gpu`                |
|`ssd_resnet101_v1_fpn_1024x1024`              | `tensorflow`        | `default-cpu`,`default-gpu`                |
|`ssd_resnet152_v1_fpn_640x640`                | `tensorflow`        | `default-cpu`,`default-gpu`                |
|`ssd_resnet152_v1_fpn_1024x1024`              | `tensorflow`        | `default-cpu`,`default-gpu`                |
|`ssd_mobilenet_v2_320x320`                    | `tensorflow`        | `default-cpu`,`default-gpu`                |
|`ssd_mobilenet_v1_fpn_640x640`                | `tensorflow`        | `default-cpu`,`default-gpu`                |
|`ssd_mobilenet_v2_fpnlite_320x320`            | `tensorflow`        | `default-cpu`,`default-gpu`                |
|`ssd_mobilenet_v2_fpnlite_640x640`            | `tensorflow`        | `default-cpu`,`default-gpu`                |


### Supported Combinations of Backend-Scenario-BatchSize
<details>
<summary>Click to expand</summary>
(to be updated)

|  | tensorflow-cpu |  |  |  |  |  |  | tensorflow-gpu |  |  |  |  |  |  | tensorrt-dynamic |  |  |  |  |  |  | openvino-cpu |  |  |  |  |  |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  | Single Stream | Single Stream | Offline | Offline | Offline | Offline |  | Single Stream | Single Stream | Offline | Offline | Offline | Offline |  | Single Stream | Single Stream | Offline | Offline | Offline | Offline |  | Single Stream | Single Stream | Offline | Offline | Offline | Offline |
|  | Accuracy | Performance | Accuracy | Performance | Accuracy | Performance |  | Accuracy | Performance | Accuracy | Performance | Accuracy | Performance |  | Accuracy | Performance | Accuracy | Performance | Accuracy | Performance |  | Accuracy | Performance | Accuracy | Performance | Accuracy | Performance |
|  | Batch Size 1 | Batch Size 1 | Batch Size 1 | Batch Size 1 | Batch Size >1 | Batch Size >1 |  | Batch Size 1 | Batch Size 1 | Batch Size 1 | Batch Size 1 | Batch Size >1 | Batch Size >1 |  | Batch Size 1 | Batch Size 1 | Batch Size 1 | Batch Size 1 | Batch Size >1 | Batch Size >1 |  | Batch Size 1 | Batch Size 1 | Batch Size 1 | Batch Size 1 | Batch Size >1 | Batch Size >1 |
| rcnn-nas-lowproposals-coco | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 |  | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 |  | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 |  | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 |
| rcnn-resnet50-lowproposals-coco | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 |  | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 |  | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 |  | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 |
| rcnn-resnet101-lowproposals-coco | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 |  | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 |  | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 |  | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 |
| rcnn-inception-resnet-v2-lowproposals-coco | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 |  | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 |  | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 |  | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 |
| rcnn-inception-v2-coco | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 |  | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 |  | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 |  | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 |
| ssd-inception-v2-coco | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 |  | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 |  | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 |  | 🟥 | 🟥 | 🟥 | 🟥 | 🟥 | 🟥 |
| ssd_mobilenet_v1_coco | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 |  | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 |  | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 |  | 🟥 | 🟥 | 🟥 | 🟥 | 🟥 | 🟥 |
| ssd_mobilenet_v1_quantized_coco | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 |  | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 |  | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 |  | 🟥 | 🟥 | 🟥 | 🟥 | 🟥 | 🟥 |
| ssd-mobilenet-v1-fpn-sbp-coco | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 |  | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 |  | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 |  | 🟥 | 🟥 | 🟥 | 🟥 | 🟥 | 🟥 |
| ssd-resnet50-v1-fpn-sbp-coco | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 |  | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 |  | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 |  | 🟥 | 🟥 | 🟥 | 🟥 | 🟥 | 🟥 |
| ssdlite-mobilenet-v2-coco | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 |  | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 |  | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 |  | 🟥 | 🟥 | 🟥 | 🟥 | 🟥 | 🟥 |
| yolo-v3-coco | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 |  | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 |  | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 |  | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 | 🟩 |
| ssd_resnet50_v1_fpn_640x640 | 🟩 | 🟩 | 🟥 | 🟥 | 🟥 | 🟥 |  | 🟩 | 🟩 | 🟩 | 🟥 | 🟥 | 🟥 |  | 🟩 | 🟩 | 🟩 | 🟩 | 🟥 | 🟥 |  | 🟥 | 🟥 | 🟥 | 🟥 | 🟥 | 🟥 |
| ssd_resnet50_v1_fpn_1024x1024 | 🟩 | 🟩 | 🟥 | 🟥 | 🟥 | 🟥 |  | 🟩 | 🟩 | 🟩 | 🟥 | 🟥 | 🟥 |  | 🟩 | 🟩 | 🟩 | 🟩 | 🟥 | 🟥 |  | 🟥 | 🟥 | 🟥 | 🟥 | 🟥 | 🟥 |
| ssd_resnet101_v1_fpn_640x640 | 🟩 | 🟩 | 🟥 | 🟥 | 🟥 | 🟥 |  | 🟩 | 🟩 | 🟩 | 🟥 | 🟥 | 🟥 |  | 🟩 | 🟩 | 🟩 | 🟩 | 🟥 | 🟥 |  | 🟥 | 🟥 | 🟥 | 🟥 | 🟥 | 🟥 |
| ssd_resnet101_v1_fpn_1024x1024 | 🟩 | 🟩 | 🟥 | 🟥 | 🟥 | 🟥 |  | 🟩 | 🟩 | 🟩 | 🟥 | 🟥 | 🟥 |  | 🟩 | 🟩 | 🟩 | 🟩 | 🟥 | 🟥 |  | 🟥 | 🟥 | 🟥 | 🟥 | 🟥 | 🟥 |
| ssd_resnet152_v1_fpn_640x640 | 🟩 | 🟩 | 🟥 | 🟥 | 🟥 | 🟥 |  | 🟩 | 🟩 | 🟩 | 🟥 | 🟥 | 🟥 |  | 🟩 | 🟩 | 🟩 | 🟩 | 🟥 | 🟥 |  | 🟥 | 🟥 | 🟥 | 🟥 | 🟥 | 🟥 |
| ssd_resnet152_v1_fpn_1024x1024 | 🟩 | 🟩 | 🟥 | 🟥 | 🟥 | 🟥 |  | 🟩 | 🟩 | 🟩 | 🟥 | 🟥 | 🟥 |  | 🟩 | 🟩 | 🟩 | 🟩 | 🟥 | 🟥 |  | 🟥 | 🟥 | 🟥 | 🟥 | 🟥 | 🟥 |
| ssd_mobilenet_v2_320x320 | 🟩 | 🟩 | 🟥 | 🟥 | 🟥 | 🟥 |  | 🟩 | 🟩 | 🟩 | 🟥 | 🟥 | 🟥 |  | 🟩 | 🟩 | 🟩 | 🟩 | 🟥 | 🟥 |  | 🟥 | 🟥 | 🟥 | 🟥 | 🟥 | 🟥 |
| ssd_mobilenet_v1_fpn_640x640 | 🟩 | 🟩 | 🟥 | 🟥 | 🟥 | 🟥 |  | 🟩 | 🟩 | 🟩 | 🟥 | 🟥 | 🟥 |  | 🟩 | 🟩 | 🟩 | 🟩 | 🟥 | 🟥 |  | 🟥 | 🟥 | 🟥 | 🟥 | 🟥 | 🟥 |
| ssd_mobilenet_v2_fpnlite_320x320 | 🟩 | 🟩 | 🟥 | 🟥 | 🟥 | 🟥 |  | 🟩 | 🟩 | 🟩 | 🟥 | 🟥 | 🟥 |  | 🟩 | 🟩 | 🟩 | 🟩 | 🟥 | 🟥 |  | 🟥 | 🟥 | 🟥 | 🟥 | 🟥 | 🟥 |
| ssd_mobilenet_v2_fpnlite_640x640 | 🟩 | 🟩 | 🟥 | 🟥 | 🟥 | 🟥 |  | 🟩 | 🟩 | 🟩 | 🟥 | 🟥 | 🟥 |  | 🟩 | 🟩 | 🟩 | 🟩 | 🟥 | 🟥 |  | 🟥 | 🟥 | 🟥 | 🟥 | 🟥 | 🟥 |

🟩 Supported
🟥 Not supported

</details>
<br>

# Setting up the Environment
# Building the Docker image

**NB:** The
[TensorRT 21.06](https://docs.nvidia.com/deeplearning/tensorrt/container-release-notes/rel_21-06.html#rel_21-06)
release is the last one to support TensorRT 7.2, needed by TensorFlow 2.7.

```
export CK_IMAGE_NAME=mlperf-inference-vision SDK_VER=21.06-py3 TF_VER=2.7.0
cd $(ck find program:$CK_IMAGE_NAME) && ./build.sh
```

<details>
<summary>Click to expand</summary>

```
Successfully built 9c39ebef9ad2
Successfully tagged krai/mlperf-inference-vision:21.06-py3_tf-2.7.0

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
krai/mlperf-inference-vision   21.06-py3_tf-2.7.0   362d3cd6ddd5   8 minutes ago   16.6GB
```