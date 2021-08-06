#!/bin/bash

# Copyright (c) 2021 Krai Ltd.
#
# SPDX-License-Identifier: BSD-3-Clause.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software without
# specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

function exit_if_error() {
    if [ "${?}" != "0" ]; then exit 1; fi
}


FILESUFFIX=""
cp ${CK_ENV_MLPERF_INFERENCE_VISION}/"classification_and_detection/python/models/ssd_mobilenet_v1.py" ${INSTALL_DIR}/
if [[ -n ${WITHOUT_ABP_NMS} ]]; then
  cp ${ORIGINAL_PACKAGE_DIR}/ssdMobileNetV1WithoutABPNMS.patch ${INSTALL_DIR}/
  patch ssd_mobilenet_v1.py ssdMobileNetV1WithoutABPNMS.patch
  FILESUFFIX="_Without_ABP_NMS"
fi


${CK_ENV_COMPILER_PYTHON_FILE} -W ignore ssd_mobilenet_v1.py --input_model ./ssd_mobilenet_v1.pytorch

echo "The SSDMobileNetV1_300_300${FILESUFFIX}.onnx is generated"

if [[ -n ${SIMPLIFY_ONNX} ]]; then
  cp ${ORIGINAL_PACKAGE_DIR}/split_and_simplify.py ${INSTALL_DIR}/
  echo "Running the split_and_simplify.py to remove NMS and generate the simplified graph"
  echo "${CK_ENV_COMPILER_PYTHON_FILE} -W ignore split_and_simplify.py ./SSDMobileNetV1_300_300${FILESUFFIX}.onnx"
  ${CK_ENV_COMPILER_PYTHON_FILE} -W ignore split_and_simplify.py  "./SSDMobileNetV1_300_300${FILESUFFIX}.onnx"
  rm ./SSDMobileNetV1_300_300${FILESUFFIX}.onnx
fi
exit_if_error
exit 0     
