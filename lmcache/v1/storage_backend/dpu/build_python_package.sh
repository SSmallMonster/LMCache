#!/bin/bash

# DPU Cache构建脚本

set -e

echo "Building DPU Cache Python Package..."

# 构建C共享库
echo "Step 1: Building C shared library..."
mkdir -p build
cd build
cmake ..
make dpu_cache -j$(nproc)
cd ..

# 检查共享库是否生成
if [ ! -f "build/libdpu_cache.so" ]; then
    echo "Error: libdpu_cache.so not found!"
    exit 1
fi

echo "Step 2: Setting up Python package..."

# 安装Python包（开发模式）
pip install -e .
echo "Build completed successfully!"