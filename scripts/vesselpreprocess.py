"""
Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import sys
import shutil
import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from loguru import logger


def convert_nii_to_npz(input_file, output_file):
    """
    将nii.gz文件转换为npz文件

    Args:
        input_file: 输入nii.gz文件路径
        output_file: 输出npz文件路径
    """
    try:
        # 加载nii.gz文件
        vessel_img = sitk.ReadImage(str(input_file))
        vessel_npy = sitk.GetArrayFromImage(vessel_img)
        
        # 确保是二值的
        vessel_npy = (vessel_npy > 0).astype(np.uint8)
        
        # 添加通道维度
        vessel_npy = vessel_npy[np.newaxis]
        
        # 保存为npz
        np.savez_compressed(output_file, data=vessel_npy)
        
        logger.info(f"转换成功: {input_file} -> {output_file}")
        return True
    except Exception as e:
        logger.error(f"转换失败 {input_file}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def main():
    """
    nndet_vesselprep命令的入口点

    只执行一个功能：将raw_splitted/vesselsTr目录中的nii.gz文件转换为npz文件，
    并保存到preprocessed/D3V001_3d/vesselsTr目录下

    用法:
        nndet_vesselprep Task102_vesspe [--det_data DIR] [--options]
    """
    import os
    import sys
    import argparse
    from pathlib import Path
    
    from loguru import logger
    
    # 配置日志记录
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="血管掩码转换工具")
    parser.add_argument("task_id", type=str, help="任务ID，例如：Task102_vesspe或102")
    parser.add_argument("--det_data", type=str, default=os.getenv("det_data", "."), 
                       help="数据根目录，默认使用det_data环境变量")
    parser.add_argument("--processes", "-p", type=int, default=8, help="并行处理的进程数")
    parser.add_argument("--overwrite", action="store_true", help="是否覆盖现有文件")
    
    args = parser.parse_args()
    
    # 处理任务ID
    task_id = args.task_id
    if task_id.startswith("Task"):
        task_id = task_id.replace("Task", "")
    task_name = f"Task{task_id}"
    
    # 设置路径
    data_root = Path(args.det_data)
    task_dir = data_root / task_name
    
    # 源目录：raw_splitted/vesselsTr
    source_dir = task_dir / "raw_splitted" / "vesselsTr"
    
    # 目标目录：preprocessed/D3V001_3d/vesselsTr
    target_dir = task_dir / "preprocessed" / "D3V001_3d" / "vesselsTr"
    
    # 确保目标目录存在
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # 检查源目录是否存在
    if not source_dir.exists() or not source_dir.is_dir():
        logger.error(f"源目录不存在: {source_dir}")
        return 1
    
    # 获取所有nii.gz文件
    nii_files = list(source_dir.glob("*.nii.gz"))
    
    if not nii_files:
        logger.warning(f"在 {source_dir} 中未找到nii.gz文件")
        return 1
    
    logger.info(f"开始处理 {len(nii_files)} 个nii.gz文件，从 {source_dir} 到 {target_dir}")
    
    # 转换每个文件
    success_count = 0
    for nii_file in nii_files:
        # 获取案例ID
        case_id = nii_file.stem
        if "_vessel" in case_id:
            case_id = case_id.split("_vessel")[0]
        
        # 设置目标文件路径
        npz_file = target_dir / f"{case_id}.npz"
        
        # 检查是否需要覆盖
        if npz_file.exists() and not args.overwrite:
            logger.info(f"跳过已存在的文件: {npz_file}")
            success_count += 1
            continue
        
        # 执行转换
        if convert_nii_to_npz(nii_file, npz_file):
            success_count += 1
    
    # 输出结果
    logger.info(f"处理完成: 成功 {success_count}/{len(nii_files)} 个文件")
    
    if success_count == len(nii_files):
        logger.info("所有文件处理成功")
        return 0
    else:
        logger.warning(f"部分文件处理失败: {len(nii_files) - success_count} 个")
        return 1


if __name__ == "__main__":
    sys.exit(main())