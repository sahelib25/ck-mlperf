ARG BASE_IMAGE
ARG SDK_VER
FROM ${BASE_IMAGE}:${SDK_VER}

# Fixed group id = 1500 (recommended to be the same as that of 'qaic').
ARG GROUP_ID
# Fixed user id = 2000
ARG USER_ID
# Use CK <= 1.55.5 with known behaviour.
ARG CK_VER

LABEL maintainer="Anton Lokhmotov <anton@krai.ai>"

# Use the Bash shell.
SHELL ["/bin/bash", "-c"]

# Allow stepping into the Bash shell interactively.
ENTRYPOINT ["/bin/bash", "-c"]

# Install known system dependencies and immediately clean up to make the image smaller.
# CK needs: git, wget, zip.
# TF needs: curl.
# TF Object Detection API needs ProtoBuf 3.0 which needs CMake.
ENV DEBIAN_FRONTEND=noninteractive
RUN apt update -y\
 && apt install -y apt-utils\
 && apt upgrade -y\
 && apt install -y\
 git wget zip libz-dev\
 curl\
 cmake\
 python3 python3-pip\
 vim\
 && apt clean

# Create a non-root user with a fixed group id 1500 and a fixed user id 2000
# (hopefully distinct from any host user id for security reasons).
# See the README for the gory details.
RUN groupadd -g ${GROUP_ID} kraig
RUN useradd -u ${USER_ID} -g kraig --create-home --shell /bin/bash krai
USER krai:kraig
WORKDIR /home/krai

# Install Collective Knowledge (CK). Make it group-executable.
ENV CK_ROOT="/home/krai/CK" \
    CK_REPOS="/home/krai/CK_REPOS" \
    CK_TOOLS="/home/krai/CK_TOOLS" \
    PATH="${CK_ROOT}/bin:/home/krai/.local/bin:${PATH}" \
    CK_CC=gcc \
    CK_PYTHON=python3 \
    GIT_USER="krai" \
    GIT_EMAIL="info@krai.ai" \
    LANG=C.UTF-8
RUN mkdir -p ${CK_ROOT} ${CK_REPOS} ${CK_TOOLS}
RUN git config --global user.name ${GIT_USER} && git config --global user.email ${GIT_EMAIL}
RUN git clone --branch V${CK_VER} https://github.com/ctuning/ck.git ${CK_ROOT}
RUN cd ${CK_ROOT}\
 && ${CK_PYTHON} setup.py install --user\
 && ${CK_PYTHON} -c "import ck.kernel as ck; print ('Collective Knowledge v%s' % ck.__version__)"\
 && chmod -R g+rx /home/krai/.local

# Explicitly create a CK experiment entry, a folder that will be mounted
# (with '--volume=<folder_for_results>:/home/krai/CK_REPOS/local/experiment').
# as a shared volume between the host and the container, and make it group-writable.
# For consistency, use the "canonical" uid from ck-analytics:module:experiment.
RUN ck create_entry --data_uoa=experiment --data_uid=bc0409fb61f0aa82 --path=${CK_REPOS}/local\
 && chmod -R g+w ${CK_REPOS}/local/experiment

# Pull CK repositories (including ck-mlperf, ck-env, ck-autotuning, ck-tensorflow, ck-docker).
RUN ck pull repo --url=http://github.com/krai/ck-object-detection.git

# Use generic Linux settings with dummy frequency setting scripts.
RUN ck detect platform.os --platform_init_uoa=generic-linux-dummy

# Detect C/C++ compiler (gcc).
RUN ck detect soft:compiler.gcc --full_path=$(which ${CK_CC})

# Detect Python.
RUN ck detect soft:compiler.python --full_path=$(which ${CK_PYTHON})
# Install the latest Python package installer (pip) and some dependencies.
RUN ${CK_PYTHON} -m pip install --ignore-installed pip setuptools --user


#-----------------------------------------------------------------------------#
# Step 1. Install prebuilt TensorFlow v2.6 (via pip).
#-----------------------------------------------------------------------------#
RUN ck install package --tags=lib,tensorflow,python-package --force_version=2.6.0
# Remove NumPy v1.19.5 which conflicts with Python COCO API.
RUN find /home/krai/CK_TOOLS/lib-python-tensorflow* -maxdepth 5 -name 'numpy*' \
    -exec rm -r {} \;
# Install Openvino Inference Engine
RUN ${CK_PYTHON} -m pip install --user tensorflow==2.5.1
RUN ck install package --tags=lib,tensorflow,python-package,openvino
#-----------------------------------------------------------------------------#


#-----------------------------------------------------------------------------#
# Step 2. Install TensorFlow v2.0 with CUDA/TensorRT support from sources.
#-----------------------------------------------------------------------------#
# # Install TF dependencies that currently have no CK packages.
# RUN ${CK_PYTHON} -m pip install --user absl-py==0.9.0              # >=0.7.0
# RUN ${CK_PYTHON} -m pip install --user astor==0.7.1                # >=0.6.0
# RUN ${CK_PYTHON} -m pip install --user cachetools==4.0.0           # <5.0,>=2.0.0
# RUN ${CK_PYTHON} -m pip install --user gast==0.2.2                 # ==0.2.2 (exact)
# RUN ${CK_PYTHON} -m pip install --user google-auth==1.13.1         # <2,>=1.6.3
# RUN ${CK_PYTHON} -m pip install --user google-auth-oauthlib==0.4.1 # <0.5,>=0.4.1
# RUN ${CK_PYTHON} -m pip install --user google-pasta==0.2.0         # >=0.1.6
# RUN ${CK_PYTHON} -m pip install --user grpcio==1.27.2              # >=1.8.6
# RUN ${CK_PYTHON} -m pip install --user h5py==2.10.0                # exact?
# RUN ${CK_PYTHON} -m pip install --user keras-applications==1.0.8   # >=1.0.8
# RUN ${CK_PYTHON} -m pip install --user keras-preprocessing==1.1.0  # >=1.0.5
# RUN ${CK_PYTHON} -m pip install --user markdown==3.2.1             # >=2.6.8
# RUN ${CK_PYTHON} -m pip install --user numpy==1.18.2               # <2.0,>=1.16.0
# RUN ${CK_PYTHON} -m pip install --user oauthlib==3.1.0             # ?
# RUN ${CK_PYTHON} -m pip install --user opt-einsum==3.2.0           # >=2.3.2
# RUN ${CK_PYTHON} -m pip install --user protobuf==3.11.3            # >=3.8.0
# RUN ${CK_PYTHON} -m pip install --user pyasn1==0.4.8               # <0.5.0,>=0.4.6
# RUN ${CK_PYTHON} -m pip install --user pyasn1-modules==0.2.8       # >=0.2.1
# RUN ${CK_PYTHON} -m pip install --user requests==2.23.0            # <3,>=2.21.0
# RUN ${CK_PYTHON} -m pip install --user requests-oauthlib==1.3.0    # >=0.7.0
# RUN ${CK_PYTHON} -m pip install --user rsa==4.0                    # <4.1,>=3.1.4
# RUN ${CK_PYTHON} -m pip install --user scipy==1.4.1                # ==1.4.1 (exact)
# RUN ${CK_PYTHON} -m pip install --user setuptools==46.1.3          # >=41.0.0
# RUN ${CK_PYTHON} -m pip install --user six==1.14.0                 # >=1.12.0
# RUN ${CK_PYTHON} -m pip install --user tensorboard==2.1.1          # <2.2.0,>=2.1.0 (almost exact)
# RUN ${CK_PYTHON} -m pip install --user tensorflow-estimator==2.1.0 # <2.2.0,>=2.1.0 (almost exact)
# RUN ${CK_PYTHON} -m pip install --user termcolor==1.1.0            # >=1.1.0
# RUN ${CK_PYTHON} -m pip install --user wheel==0.30.0               # >=0.26
# RUN ${CK_PYTHON} -m pip install --user werkzeug==1.0.1             # >=0.11.15
# RUN ${CK_PYTHON} -m pip install --user wrapt==1.11.2               # >=1.11.1

# # Detect TF CUDA dependencies.
# RUN ck detect soft:compiler.cuda --full_path=/usr/local/cuda/bin/nvcc \
#  && ck detect soft:lib.cublas --full_path=/usr/local/cuda-11.4/targets/x86_64-linux/lib/libcublas.so \
#  && ck detect soft:lib.cudnn
# # Install TF build dependencies.
# RUN ck install ck-env:package:jdk-8u131-universal
# RUN ck install ck-env:package:tool-bazel-0.27.1-linux
# # Build TF from sources with TensorRT support. Remove the Bazel cache (~16 GB) once done.
# RUN ck pull repo:ck-env
# RUN ck install package --tags=lib,tensorflow,vsrc,vcuda,v2.1.0 \
# --env.CK_TF_NEED_TENSORRT=YES --cus.version=2.1.0\
#  && rm -rf ~/.cache/bazel
# -----------------------------------------------------------------------------#


#-----------------------------------------------------------------------------#
# Step 3. Install the COCO 2017 validation dataset (5,000 images).
#-----------------------------------------------------------------------------#
# Download the dataset to the default path. Remove all training annotations (~765 MB).
RUN echo | ck install package --tags=object-detection,dataset,coco.2017,val,original,full\
 && ck virtual env --tags=object-detection,dataset,coco.2017,val,original,full --shell_cmd=\
'rm $CK_ENV_DATASET_COCO_LABELS_DIR/*train2017.json'
# Install Python COCO API.
RUN ck install package --tags=lib,python-package,cython
RUN echo "" | ck install package --tags=lib,python-package,numpy
RUN ck install package --tags=lib,python-package,matplotlib
RUN ck install package --tags=tool,coco,api
#-----------------------------------------------------------------------------#


#-----------------------------------------------------------------------------#
# Step 4. Install the object detection models.
#-----------------------------------------------------------------------------#
# Install TF model API, but remove useless API files to free up space.
RUN ck detect soft --tags=cmake --full_path=/usr/local/bin/cmake
RUN ck install package --tags=model,tensorflow,api\
 && ck virtual env --tags=model,tensorflow,api --shell_cmd=\
'cd $CK_ENV_TENSORFLOW_MODELS;\
 mv object_detection ..;\
 rm * -r;\
 mv ../object_detection .;\
 cd ..;\
 rm official -rf;\
 rm samples -rf;\
 rm tutorials -rf;\
 rm .git -rf'

RUN ck install package --tags=rcnn,nas,lowproposals,vcoco\
 && ck install package --tags=rcnn,resnet50,lowproposals\
 && ck install package --tags=rcnn,resnet101,lowproposals\
 && ck install package --tags=rcnn,inception-resnet-v2,lowproposals\
 && ck install package --tags=rcnn,inception-v2\
 && ck install package --tags=ssd,inception-v2\
 && ck install package --tags=ssd,mobilenet-v1,non-quantized,mlperf,tf\
 && ck install package --tags=ssd,mobilenet-v1,quantized,mlperf,tf\
 && ck install package --tags=ssd,mobilenet-v1,fpn\
 && ck install package --tags=ssd,resnet50,fpn\
 && ck install package --tags=ssdlite,mobilenet-v2,vcoco\
 && ck install package --tags=yolo-v3
#-----------------------------------------------------------------------------#


#-----------------------------------------------------------------------------#
# Step 5. Make final preparations to run the Object Detection TF-Python program.
#-----------------------------------------------------------------------------#
# Install remaining Python dependencies of the program.
# (Some have already been installed.)
RUN ck install package --tags=lib,python-package,cv2,opencv-python-headless
# Allow the program create tmp files when running under an external user.
RUN chmod -R g+rwx ${CK_REPOS}/ck-tensorflow/program/object-detection-tf-py/
#-----------------------------------------------------------------------------#


#=============================================================================#
# NB: The above is shared with object-detection-tf-py.tensorrt.ubuntu-18.04
#=============================================================================#


#-----------------------------------------------------------------------------#
# Step 6. Make final preparations to run the official vision app with CK mods.
#-----------------------------------------------------------------------------#
# NB: Apparently, we still need Pillow to run the official vision app.
RUN ck install package --tags=lib,python-package,pillow
# NB: While Abseil has already been installed above, we install and register it
# with CK here as well, as it is needed for LoadGen.
RUN ck install package --tags=lib,python-package,absl
# Install LoadGen from the official MLPerf Inference repo.
RUN ck install package --tags=mlperf,inference,source,upstream.master
RUN ck install package --tags=lib,python-package,mlperf,loadgen
# Install the official vision app with CK modifications and make it use
# the default LoadGen config file.
RUN ck install package --tags=mlperf,inference,source,krai.vision_with_ck
RUN ck detect  soft    --tags=loadgen,config,from.inference.master
# Allow the program to create tmp files when running under an external user.
RUN chmod -R g+rwx `ck find program:mlperf-inference-vision`
#-----------------------------------------------------------------------------#


#-----------------------------------------------------------------------------#
# Run the official MLPerf Inference vision app
# under the Offline scenario
# in the Accuracy mode on the first 50 images of the COCO 2017 validation dataset
# with the TensorFlow CPU backend and the SSD-MobileNet-FPN model.
#-----------------------------------------------------------------------------#
CMD ["ck run program:mlperf-inference-vision --cmd_key=direct --skip_print_timers \
--env.CK_LOADGEN_SCENARIO=Offline --env.CK_LOADGEN_MODE='--accuracy' --env.CK_LOADGEN_EXTRA_PARAMS='--count 50' \
--env.CK_LOADGEN_BACKEND=tensorflow --env.CUDA_VISIBLE_DEVICES=-1 --dep_add_tags.lib-tensorflow=vpip \
--env.CK_LOADGEN_REF_PROFILE=default_tf_object_det_zoo --env.CK_METRIC_TYPE=COCO --dep_add_tags.weights=ssd,mobilenet-v1,fpn"]
