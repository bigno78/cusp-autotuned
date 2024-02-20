#!/bin/bash

KTT_PATH="../KTT-2.1"

time -p \
    nvcc -DCUSP_PATH=$(realpath .) \
     -I . -I "$KTT_PATH/Source" \
     -l cuda -l ktt -L "$KTT_PATH/Build/x86_64_Release/" \
     --linker-options=-rpath,$(realpath "$KTT_PATH/Build/x86_64_Release/") \
     -std=c++17 -g -O3 -lineinfo \
     -DTIME_CSR=1 \
     -o run ftest.cu

