#!/bin/bash

if [ "$DATASET_CALIBRATION" == "yes" ]
then
wget https://raw.githubusercontent.com/mlcommons/inference/master/calibration/SQuAD-v1.1/bert-calibration.txt -P ${INSTALL_DIR}/calib/
DATASET_CALIBRATION_FILE=${INSTALL_DIR}/calib/bert-calibration.txt
else
DATASET_CALIBRATION_FILE=""
fi

"$CK_ENV_COMPILER_PYTHON_FILE" "${PACKAGE_DIR}/tokenize_and_pack.py" \
    "$CK_ENV_DATASET_SQUAD_ORIGINAL" \
    "$CK_ENV_DATASET_TOKENIZATION_VOCAB" \
    "${INSTALL_DIR}/bert_tokenized_squad_v1_1" \
    "$DATASET_MAX_SEQ_LENGTH" \
    "$DATASET_MAX_QUERY_LENGTH" \
    "$DATASET_DOC_STRIDE" \
    "$DATASET_RAW" \
    "$DATASET_CALIBRATION_FILE" \
