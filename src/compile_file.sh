#!/bin/bash
FNAME=$1
BC=$2
if [ ${BC} -eq "" ]; then
    echo "USAGE: compile_file.sh <src_name> <bitcode>"
    echo "<src_name>: this is the cpp file name WITHOUT the extenion. The script expects the extension to be .cpp"
    echo "<bitcode>: this is the SYCL bitcode to compile the device code."
    echo "           This bitcode should have the following taxonomy:"
    echo "           NVIDIA CUDA GPU: ptx64"
    echo "           AMD GPU: spir64"
    echo "           Intel GPU: spirv64"
fi
COMPILER=compute++
BINTYPE="-sycl-target ${BC}"
COMPILE_FLAGS="-Wall -fpic -DNDEBUG -sycl -no-serial-memop"
FILESRC=${FNAME}.cpp
if [ -f ${FILESRC} ]; then
    echo Running \"${COMPILER} ${COMPILE_FLAGS} ${BINTYPE} -sycl-ih ${FNAME}.sycl ${FILESRC}\"
    ${COMPILER} ${COMPILE_FLAGS} ${BINTYPE} -sycl-ih ${FNAME}.sycl ${FILESRC}
fi
if [ -f ${FNAME}.sycl ]; then
    echo "SUCCESS!"
else
    echo "FAILED: Could not generate SYCL integration header!"
fi
