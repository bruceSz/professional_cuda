#!/bin/bash

usage() {
    echo $0 target_cu
    exit 1
}

if [ $# -eq 0 ];then
    usage
fi

src=$1

CUDA_BIN=/usr/local/cuda-10.0/bin/nvcc
$CUDA_BIN -arch=sm_35 -rdc=true $src   /usr/local/cuda/lib64/libcudadevrt.a 
