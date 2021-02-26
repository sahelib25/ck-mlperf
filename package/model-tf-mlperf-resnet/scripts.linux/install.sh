#!/bin/bash

fix_input_shape=${_FIX_INPUT_SHAPE:-NO}
echo "Fix input shape? ${fix_input_shape}"
echo

if [ "${fix_input_shape}" == "YES" ]; then
  echo "${CK_ENV_COMPILER_PYTHON_FILE} ${ORIGINAL_PACKAGE_DIR}/fix_input_shape.py --input_graph ${_MODEL_INPUT_FILE_NAME} --input_name ${_TENSORFLOW_MODEL_INPUT_LAYER_NAME} --type b --output_graph ${_MODEL_INPUT_FILE_NAME}"
  "${CK_ENV_COMPILER_PYTHON_FILE}" "${ORIGINAL_PACKAGE_DIR}/fix_input_shape.py" --input_graph "${_MODEL_INPUT_FILE_NAME}" --input_name "${_TENSORFLOW_MODEL_INPUT_LAYER_NAME}" --type b --output_graph "${_MODEL_INPUT_FIXED_FILE_NAME}"
fi

echo "Done."
exit 0
