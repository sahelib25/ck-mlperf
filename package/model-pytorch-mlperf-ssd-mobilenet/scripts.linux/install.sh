#!/bin/bash

FILESUFFIX=""
cp ${CK_ENV_MLPERF_INFERENCE_VISION}/"classification_and_detection/python/models/ssd_mobilenet_v1.py" ${INSTALL_DIR}/
if [[ -n ${WITHOUT_ABP_NMS} ]]; then
  cp ${ORIGINAL_PACKAGE_DIR}/ssdMobileNetV1WithoutABPNMS.patch ${INSTALL_DIR}/
  patch ssd_mobilenet_v1.py ssdMobileNetV1WithoutABPNMS.patch
  FILESUFFIX="_Without_ABP_NMS"
fi


python3 -W ignore ssd_mobilenet_v1.py --input_model ./ssd_mobilenet_v1.pytorch

echo "The SSDMobileNetV1_300_300${FILESUFFIX}.onnx is generated"

if [[ -n ${SIMPLIFY_ONNX} ]]; then
  cp ${ORIGINAL_PACKAGE_DIR}/split_and_simplify.py ${INSTALL_DIR}/
  echo "Running the split_and_simplify.py to remove NMS and generate the simplified graph"
  echo "python3 -W ignore split_and_simplify.py ./SSDMobileNetV1_300_300${FILESUFFIX}.onnx"
  python3 -W ignore split_and_simplify.py  "./SSDMobileNetV1_300_300${FILESUFFIX}.onnx"
  rm ./SSDMobileNetV1_300_300${FILESUFFIX}.onnx
fi
exit 0     
