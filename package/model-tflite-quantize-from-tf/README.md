# Quantize TF models to TFLite models

A package that quantize TF models to TFLite models.

Currently, only the following [MLPerf](http://github.com/mlperf/inference)
Image Classification models are supported:
- [ResNet](#resnet)

And, only these quantization modes are supported:
- `int 8`
- `fp 16`

<a name="resnet"></a>
## ResNet
Quantize to `int 8`
```
$ ck install package --tags=image-classification,tf,quantized-to,tflite,resnet-int8
```
Quantize to `fp 16`
```
$ ck install package --tags=image-classification,tf,quantized-to,tflite,resnet-fp16
```
