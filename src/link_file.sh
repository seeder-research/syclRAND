#!/bin/bash
FNAME=$1
COMPILER=compute++
COMPILE_FLAGS="-Wall -fpic -O3 -DNDEBUG -shared"
OPENCL_LIBS="/usr/local/cuda/lib64"
LOAD_FLAGS="-L${OPENCL_LIBS}"
FILESRC=${FNAME}.cpp
if [ -f ${FILESRC} ] && [ -f ${FNAME}.sycl ]; then
    echo Running \"${COMPILER} ${COMPILE_FLAGS} ${LOAD_FLAGS} -include ${FNAME}.sycl -x c++ ${FILESRC} -o lib${FNAME}.so -lOpenCL\"
    ${COMPILER} ${COMPILE_FLAGS} ${LOAD_FLAGS} -include ${FNAME}.sycl -x c++ ${FILESRC} -o lib${FNAME}.so -lOpenCL
fi
if [ -f lib${FNAME}.so ]; then
    echo "SUCCESS!"
else
    echo "FAILED: Could not generate lib${FNAME}.so!"
fi
