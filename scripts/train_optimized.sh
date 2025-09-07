#!/bin/bash

# 优化训练脚本 - 提高GPU利用率和训练速度
# 使用方法: ./scripts/train_optimized.sh Task103_vesspe 0

# 检查参数
if [ $# -ne 2 ]; then
    echo "使用方法: $0 <task_name> <fold>"
    echo "例如: $0 Task103_vesspe 0"
    exit 1
fi

TASK=$1
FOLD=$2

# 设置环境变量以优化性能
export OMP_NUM_THREADS=4  # 减少OpenMP线程数，避免与数据加载线程冲突
export det_num_threads=16  # 增加数据加载线程数
export det_verbose=1

# 设置CUDA相关环境变量
export CUDA_LAUNCH_BLOCKING=0  # 启用异步CUDA操作
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512  # 优化CUDA内存分配

# 打印配置信息
echo "========================================="
echo "开始优化训练"
echo "任务: $TASK"
echo "折数: $FOLD"
echo "OMP线程数: $OMP_NUM_THREADS"
echo "数据加载线程数: $det_num_threads"
echo "========================================="

# 启动训练
python scripts/train.py \
    task=$TASK \
    exp.fold=$FOLD \
    trainer_cfg.benchmark=True \
    trainer_cfg.deterministic=False \
    augment_cfg.num_threads=16 \
    augment_cfg.num_cached_per_thread=4 \
    augment_cfg.batch_size=8

echo "========================================="
echo "训练完成"
echo "=========================================" 