#!/bin/bash
if ls *.so 1> /dev/null 2>&1; then
    rm *.so
fi
if ls *.s 1> /dev/null 2>&1; then
    rm *.s
fi
if ls *.sycl 1> /dev/null 2>&1; then
    rm *.sycl
fi
if ls *.bc 1> /dev/null 2>&1; then
    rm *.bc
fi
