#
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
#

# execution example in virtual env:
#   ck virtual env --tags=onnx,python-package --shell_cmd="python3 split_and_simplify.py ssd-mobilenet.onnx"

import sys
import onnx
import onnx.numpy_helper as onh
from onnx import helper
from onnxsim import simplify

def SimplifyOnnxModel(model):
    simplifiedModel, check = simplify(model)
    assert check, 'assert check failed'
    return simplifiedModel

def ModifyReshapeNodes(model):
    for node in model.graph.node:
        if node.op_type == "Reshape":
            for i, t in enumerate(model.graph.initializer):
                if t.name == node.input[1]:
                    shapeInfo = onh.to_array(t)
                    shapeCopy = shapeInfo.copy()
                    shapeCopy[0] = 0
                    rawNumpydata = onh.from_array(shapeCopy)
                    model.graph.initializer[i].raw_data = rawNumpydata.raw_data
    return model

onnx_filename = sys.argv[1]

try:
  input_names = ['input.1']
  output_names = ['616', '615']

  onnx.utils.extract_model(onnx_filename, onnx_filename, input_names, output_names)
except:
  pass


model_def = onnx.load(onnx_filename)

model_def = SimplifyOnnxModel(model_def)
model_def = ModifyReshapeNodes(model_def)
onnx.save(model_def, "./simplified.onnx")
