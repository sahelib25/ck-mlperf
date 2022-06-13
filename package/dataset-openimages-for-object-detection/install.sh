#!/bin/bash

# requirements -

#  pip install fiftyone

# if centos7, also -
#  pip install fiftyone-db-rhel7

cd ${CK_ENV_MLPERF_INFERENCE}/vision/classification_and_detection/tools/
./openimages_mlperf.sh

mv ../open-images-v6-mlperf ${INSTALL_DIR}
