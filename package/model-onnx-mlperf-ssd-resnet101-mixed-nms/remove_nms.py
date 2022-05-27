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

import os
import re
import sys

import onnx_graphsurgeon as gs
import onnx

import numpy as np

onnx_filename_in     = sys.argv[1]
onnx_filename_top    = sys.argv[2]
onnx_filename_bottom = sys.argv[3]

from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto

###############################################################################
# split out the top of the graph removing nms
###############################################################################
model = onnx.load(onnx_filename_in)

graph_def = model.graph
nodes = graph_def.node

# remove the nodes just after the point in the graph we care about
for node in nodes:
    if node.name == "StatefulPartitionedCall/Postprocessor/Reshape_2" or node.name == "StatefulPartitionedCall/Postprocessor/convert_scores":
        graph_def.node.remove(node)

# remove the current output nodes from the graph
for i in range(8):
    del graph_def.output[0]

# create new output nodes
scores = helper.make_tensor_value_info('StatefulPartitionedCall/Postprocessor/Decode/transpose_1:0', TensorProto.FLOAT, [1,51150,4])
boxes = helper.make_tensor_value_info('StatefulPartitionedCall/concat_1:0', TensorProto.FLOAT, [1,51150,91])

graph_def.output.extend([boxes])
graph_def.output.extend([scores])

# use graphsurgeon to change input cast to float and remove the remaining unconnected cruft
surgeon = gs.import_onnx(model)

cast_nodes = [node for node in surgeon.nodes if node.name == "StatefulPartitionedCall/Cast"]
cast_nodes[0].inputs[0].dtype = np.float32

surgeon.cleanup()

# export the file
model = gs.export_onnx(surgeon)
onnx.save(model, onnx_filename_top)


###############################################################################
# split out the bottom of the graph containing the nms
###############################################################################
model = onnx.load(onnx_filename_in)

graph_def = model.graph
nodes = graph_def.node

# remove the nodes just before the point in the graph we care about
for node in nodes:
    if node.name == "StatefulPartitionedCall/Postprocessor/Decode/transpose_1" or node.name == "StatefulPartitionedCall/concat_1":
        graph_def.node.remove(node)

# remove the current input node from the graph
del graph_def.input[0]

# create new input nodes
scores = helper.make_tensor_value_info('StatefulPartitionedCall/concat_1:0', TensorProto.FLOAT, [1,51150,91])
boxes = helper.make_tensor_value_info('StatefulPartitionedCall/Postprocessor/Decode/transpose_1:0', TensorProto.FLOAT, [1,51150,4])

graph_def.input.extend([boxes])
graph_def.input.extend([scores])

# use graphsurgeon to remove the remaining unconnected cruft
surgeon = gs.import_onnx(model)
surgeon.cleanup()

# export the file
model = gs.export_onnx(surgeon)
onnx.save(model, onnx_filename_bottom)


'''
input_names = ['StatefulPartitionedCall/concat_1:0', 'StatefulPartitionedCall/Postprocessor/Decode/transpose_1:0']
output_names = ['detection_boxes', 'detection_classes', 'detection_scores']

onnx.utils.extract_model("intermediate_model.onnx", "model_tail.onnx", input_names, output_names)

import sys
sys.exit()

input_names = ['input_tensor:0']
output_names = ['StatefulPartitionedCall/concat_1:0', 'StatefulPartitionedCall/Postprocessor/Decode/transpose_1:0']

onnx.utils.extract_model("intermediate_model.onnx", "intermediate_model.onnx", input_names, output_names)


model = onnx.load("./intermediate_model.onnx")

graph = gs.import_onnx(model)

collectCastNodes = [node for node in graph.nodes if node.name == "StatefulPartitionedCall/Cast"]

collectCastNodes[0].inputs[0].dtype = np.float32
onnx.save(gs.export_onnx(graph), onnx_filename)
'''