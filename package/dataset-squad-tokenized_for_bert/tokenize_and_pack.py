#!/usr/bin/env python3

import os
import sys

from transformers import BertTokenizer

# Input and output file paths:
squad_original_path     = sys.argv[1]
tokenization_vocab_path = sys.argv[2]
tokenized_squad_path    = sys.argv[3]

# Tokenization parameters:
max_seq_length          = int(sys.argv[4])
max_query_length        = int(sys.argv[5])
doc_stride              = int(sys.argv[6])

convert_to_raw          = sys.argv[7] == "yes"
first_100               = sys.argv[8] == "yes"

BERT_CODE_ROOT=os.environ['CK_ENV_MLPERF_INFERENCE']+'/language/bert'

sys.path.insert(0, BERT_CODE_ROOT)
sys.path.insert(0, BERT_CODE_ROOT + '/DeepLearningExamples/TensorFlow/LanguageModeling/BERT')

from create_squad_data import read_squad_examples, convert_examples_to_features

print("Creating the tokenizer from {} ...".format(tokenization_vocab_path))
tokenizer = BertTokenizer(tokenization_vocab_path)

print("Reading examples from {} ...".format(squad_original_path))
eval_examples = read_squad_examples(input_file=squad_original_path, is_training=False, version_2_with_negative=False)

if first_100:
    eval_examples = eval_examples[0:100]

eval_features = []
def append_feature(feature):
    eval_features.append(feature)

print("Tokenizing examples to features (max_seq_length={}, max_query_length={}, doc_stride={}) ...".format(max_seq_length, max_query_length, doc_stride))
convert_examples_to_features(
    examples=eval_examples,
    tokenizer=tokenizer,
    max_seq_length=max_seq_length,
    doc_stride=doc_stride,
    max_query_length=max_query_length,
    is_training=False,
    output_fn=append_feature,
    verbose_logging=True)


if convert_to_raw:

    import numpy as np

    print("Recording features to {} ...".format(tokenized_squad_path + "*.raw"))

    num_features = len(eval_features)
    input_ids = np.zeros((num_features, max_seq_length), dtype=np.int64)
    input_mask = np.zeros((num_features, max_seq_length), dtype=np.int64)
    segment_ids = np.zeros((num_features, max_seq_length), dtype=np.int64)

    for idx, feature in enumerate(eval_features):

        if len(feature.input_ids) != 384:
            print(len(feature.input_ids))
        input_ids[idx, :] = np.array(feature.input_ids, dtype=np.int64)
        input_mask[idx, :] = np.array(feature.input_mask, dtype=np.int64)
        segment_ids[idx, :] = np.array(feature.segment_ids, dtype=np.int64)

    input_ids.astype('int64').tofile(tokenized_squad_path + "_input_ids.raw")
    input_mask.astype('int64').tofile(tokenized_squad_path + "_input_mask.raw")
    segment_ids.astype('int64').tofile(tokenized_squad_path + "_segment_ids.raw")

else: # pickle

    import pickle

    print("Recording features to {} ...".format(tokenized_squad_path + ".pickle"))
    with open(tokenized_squad_path + ".pickle", 'wb') as cache_file:

        pickle.dump(eval_features, cache_file)


