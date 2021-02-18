# MLPerf Inference - Object Detection - ONNX

## Pre-requisites

### CK repository (this one)

<pre>
&dollar; ck pull repo --url=https://github.com/krai/ck-mlperf
</pre>

### ONNX libraries

<pre>
&dollar; ck install package --tags=python-package,onnx
&dollar; ck install package --tags=python-package,onnxruntime
</pre>

### ONNX Object Detection model

Install SSD-ResNet34:

<pre>
&dollar; ck install package --tags=model,onnx,mlperf,ssd-resnet
</pre>

### Datasets

<pre>
&dollar; ck install package --tags=dataset,object-detection,preprocessed,side.1200
</pre>

## Running

<pre>
&dollar; ck run program:object-detection-onnx-py --env.CK_BATCH_COUNT=5000
</pre>

### Program parameters

#### `CK_BATCH_COUNT`

The number of images to be processed.

Default: `1`.

#### `CK_SKIP_IMAGES`

The number of skipped images.

Default: `0`.
