#!/bin/bash

FILESUFFIX=""
cp ${ORIGINAL_PACKAGE_DIR}/flatlabels.txt ${INSTALL_DIR}
cp ${CK_ENV_MLPERF_INFERENCE_VISION}/"classification_and_detection/python/models/ssd_mobilenet_v1.py" ${INSTALL_DIR}/
if [[ -n ${WITHOUT_ABP_NMS} ]]; then
  cp ${ORIGINAL_PACKAGE_DIR}/ssdMobileNetV1WithoutABPNMS.patch ${INSTALL_DIR}/
  patch ssd_mobilenet_v1.py ssdMobileNetV1WithoutABPNMS.patch
  FILESUFFIX="_Without_ABP_NMS"
fi


python3 -W ignore ssd_mobilenet_v1.py --input_model ./ssd_mobilenet_v1.pytorch

echo "The SSDMobileNetV1_300_300${FILESUFFIX}.onnx is generated"

if [[ -n ${SIMPLIFY_ONNX} ]]; then
  cp ${ORIGINAL_PACKAGE_DIR}/splitAndSimplifyONNXGraph.py ${INSTALL_DIR}/
  echo "Running the splitAndSimplifyONNXGraph.py Script to generate the simplified graph"
  python3 -W ignore splitAndSimplifyONNXGraph.py --onnx_input "./SSDMobileNetV1_300_300${FILESUFFIX}.onnx" --onnx_output "./SSDMobileNetV1_300_300${FILESUFFIX}_batchsize_sim.onnx"
fi
exit 0     
