#!/bin/bash

# requirements -

#  pip install fiftyone

# if centos7, also -
#  pip install fiftyone-db-rhel7

rm -rf ${INSTALL_DIR}/open-images-v6-mlperf

cd ${CK_ENV_MLPERF_INFERENCE}/vision/classification_and_detection/tools/

if [ "${DATASET_CALIBRATION}" == "yes" ]; then
    ln -s -f ${CK_ENV_MLPERF_INFERENCE}/calibration .
    ./openimages_calibration_mlperf.sh -d ${INSTALL_DIR}
else # validation
    ./openimages_mlperf.sh -d ${INSTALL_DIR}
fi


