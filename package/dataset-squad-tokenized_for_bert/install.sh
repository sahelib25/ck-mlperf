#!/bin/bash

env

"$CK_ENV_COMPILER_PYTHON_FILE" "${PACKAGE_DIR}/tokenize_and_pack.py" \
    "$CK_ENV_DATASET_SQUAD_ORIGINAL" \
    "$CK_ENV_DATASET_TOKENIZATION_VOCAB" \
    "${INSTALL_DIR}/bert_tokenized_squad_v1_1" \
    "$DATASET_MAX_SEQ_LENGTH" \
    "$DATASET_MAX_QUERY_LENGTH" \
    "$DATASET_DOC_STRIDE" \
    "$DATASET_RAW" \
    "$DATASET_FIRST_100" \
