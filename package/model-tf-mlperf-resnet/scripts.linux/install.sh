#!/bin/bash

fix_input_shape=${_FIX_INPUT_SHAPE:-NO}
echo "Fix input shape? ${fix_input_shape}"
echo

if [ "${fix_input_shape}" == "YES" ]; then
  "${CK_ENV_COMPILER_PYTHON_FILE}" "${ORIGINAL_PACKAGE_DIR}/fix_input_shape.py" "${INSTALL_DIR}"
fi

echo "Done."
exit 0
