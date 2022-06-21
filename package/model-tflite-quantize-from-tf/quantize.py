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
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
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
    representative_dataset_path = os.environ['CK_ENV_DATASET_IMAGENET_PREPROCESSED_DIR']
    representative_dataset_path = pathlib.Path(representative_dataset_path)

    for image in representative_dataset_path.glob('*/*.rgb8'):
        data = tf.keras.preprocessing.image.load_img(image)
        data = tf.keras.preprocessing.image.img_to_array(data)
        print(data.shape)
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