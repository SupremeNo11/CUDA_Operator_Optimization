#!/bin/bash
# 查询当前设备的 CUDA 架构 (compute capability)
# 用法: bash query_arch.sh
# 输出: 设备名称和架构号，例如: NVIDIA GeForce GTX 1050 Ti, 61
# 用于设置 CMAKE_CUDA_ARCHITECTURES

nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>/dev/null | \
  awk -F',' '{gsub(/\./,"",$2); print $1","$2}'
