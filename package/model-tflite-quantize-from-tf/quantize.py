#!/usr/bin/env python3

import os
import shutil
import pathlib

def frozengraph2savedmodel(graph_pb, export_dir, input_layer_name, output_layer_name):
    import tensorflow.compat.v1 as tf
    from tensorflow.python.saved_model import signature_constants
    from tensorflow.python.saved_model import tag_constants

    if os.path.exists(export_dir):
        shutil.rmtree(export_dir)
    os.mkdir(export_dir)

    builder = tf.saved_model.builder.SavedModelBuilder(export_dir)

    with tf.gfile.GFile(graph_pb, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    sigs = {}

    with tf.Session(graph=tf.Graph()) as sess:
        tf.import_graph_def(graph_def, name="")
        g = tf.get_default_graph()
        image_tensor = g.get_tensor_by_name(input_layer_name)
        softmax_tensor = g.get_tensor_by_name(output_layer_name)

        sigs[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = \
            tf.saved_model.signature_def_utils.predict_signature_def(
                {"in": image_tensor},
                {"out": softmax_tensor})

        builder.add_meta_graph_and_variables(sess,
                                             [tag_constants.SERVING],
                                             signature_def_map=sigs)
    builder.save()

def quantize(saved_model_filepath, tflite_model_file, quantization_level):
    import tensorflow as tf
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_filepath)

    if quantization_level == "int8":
        converter.representative_dataset = representative_dataset
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # Enforce full integer quantization for all ops including the input and output
        # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        # converter.target_spec.supported_types = [tf.int8]
        # converter.inference_type = tf.int8
        # converter.inference_input_type = tf.int8
        # converter.inference_output_type = tf.int8
    else:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]

    tflite_model = converter.convert()
    tflite_model_file = pathlib.Path(tflite_model_file)
    tflite_model_file.write_bytes(tflite_model)

    if os.path.exists(saved_model_filepath):
        shutil.rmtree(saved_model_filepath)
        

def representative_dataset():
    import tensorflow as tf
    import glob
    import numpy as np

    representative_dataset_path = os.environ['CK_DATASET_IMAGENET_CALIBRATION_ROOT']
    height                      = int(os.environ['MODEL_IMAGE_HEIGHT'])
    width                       = int(os.environ['MODEL_IMAGE_WIDTH'])

    for file in glob.glob(representative_dataset_path + '/*.JPEG'):
        data = tf.keras.preprocessing.image.load_img(file)
        data = tf.keras.preprocessing.image.img_to_array(data)
        data = tf.image.resize(data, (height, width), method='nearest')
        data = np.array(data).reshape(1, height, width, 3).astype(np.float32)
        yield [data]

if __name__ == '__main__':

    pb_model_filepath       = os.environ['CK_ENV_TENSORFLOW_MODEL_TF_FROZEN_FILEPATH']
    saved_model_filepath    = os.path.join(os.environ['INSTALL_DIR'], "tmp")
    tflite_model_filepath   = os.path.join(os.environ['INSTALL_DIR'], os.environ['PACKAGE_NAME'])
    quantization_level      = os.environ['MODEL_QUANTIZATION_LEVEL']
    input_layer_name        = os.environ['MODEL_INPUT_LAYER_NAME']
    output_layer_name       = os.environ['MODEL_OUTPUT_LAYER_NAME']

    frozengraph2savedmodel(pb_model_filepath, saved_model_filepath, input_layer_name, output_layer_name)
    quantize(saved_model_filepath, tflite_model_filepath, quantization_level)