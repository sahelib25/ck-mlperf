#! /bin/bash

function exit_if_error() {
    if [ "${?}" != "0" ]; then exit 1; fi
}

if [[ -n ${PATCH_CHECKER} ]]; then
  CHECKER_FILE=${INSTALL_DIR}/build/onnx/checker.py
  sed -i "s/C.check_model(protobuf_string)/#C.check_model(protobuf_string)/g" ${CHECKER_FILE}
  exit_if_error
  echo "Patching done"
else 
  echo "No patching"
fi

return 0
