#!/bin/bash

# 安全训练脚本 - 避免worker崩溃
# 使用方法: ./scripts/train_safe.sh Task103_vesspe 0

if [ $# -ne 2 ]; then
    echo "使用方法: $0 <task_name> <fold>"
    echo "例如: $0 Task103_vesspe 0"
    exit 1
fi

TASK=$1
FOLD=$2

# 设置保守的环境变量
export OMP_NUM_THREADS=2  # 减少OpenMP线程数
export det_num_threads=8  # 减少数据加载线程数
export det_verbose=1

# 设置CUDA环境变量
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256

echo "========================================="
echo "开始安全训练模式"
echo "任务: $TASK"
echo "折数: $FOLD"
echo "OMP线程数: $OMP_NUM_THREADS"
echo "数据加载线程数: $det_num_threads"
echo "Batch Size: 6 (安全模式)"
echo "========================================="

# 启动训练
python scripts/train.py \
    task=$TASK \
    exp.fold=$FOLD \
    train=v004_safe \
    trainer_cfg.benchmark=True \
    trainer_cfg.deterministic=False

echo "========================================="
echo "训练完成"
echo "=========================================" 