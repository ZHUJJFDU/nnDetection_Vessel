#!/usr/bin/env python3
"""
脚本用于提取肺栓塞包围盒区域
从labelsTr中读取每个非零标签值的包围盒，扩展4个像素，
然后从对应的图像和标签中分割出这些区域并保存
"""

import os
import json
import numpy as np
import SimpleITK as sitk
from pathlib import Path
from loguru import logger
from tqdm import tqdm

def get_bounding_box_3d(mask, label_id):
    """
    从3D掩码中获取指定标签的包围盒
    
    Args:
        mask: 3D标签掩码 [Z, Y, X]
        label_id: 标签ID
        
    Returns:
        tuple: (z_min, z_max, y_min, y_max, x_min, x_max)
    """
    # 找到该标签的所有像素位置
    coords = np.where(mask == label_id)
    
    if len(coords[0]) == 0:
        return None
    
    # 计算包围盒
    z_min, z_max = int(np.min(coords[0])), int(np.max(coords[0])) + 1
    y_min, y_max = int(np.min(coords[1])), int(np.max(coords[1])) + 1  
    x_min, x_max = int(np.min(coords[2])), int(np.max(coords[2])) + 1
    
    return (z_min, z_max, y_min, y_max, x_min, x_max)

def expand_bbox(bbox, expansion=4, shape=None):
    """
    扩展包围盒
    
    Args:
        bbox: (z_min, z_max, y_min, y_max, x_min, x_max)
        expansion: 扩展像素数
        shape: 图像形状 (Z, Y, X)，用于边界检查
        
    Returns:
        tuple: 扩展后的包围盒
    """
    z_min, z_max, y_min, y_max, x_min, x_max = bbox
    
    # 扩展包围盒
    z_min = max(0, z_min - expansion)
    y_min = max(0, y_min - expansion)
    x_min = max(0, x_min - expansion)
    
    if shape is not None:
        z_max = min(shape[0], z_max + expansion)
        y_max = min(shape[1], y_max + expansion)
        x_max = min(shape[2], x_max + expansion)
    else:
        z_max = z_max + expansion
        y_max = y_max + expansion
        x_max = x_max + expansion
    
    return (z_min, z_max, y_min, y_max, x_min, x_max)

def crop_and_save_region(image_data, label_data, bbox, case_id, label_id, 
                        output_image_dir, output_label_dir, image_props, label_props):
    """
    根据包围盒裁剪并保存图像和标签区域
    
    Args:
        image_data: 图像数据
        label_data: 标签数据
        bbox: 包围盒 (z_min, z_max, y_min, y_max, x_min, x_max)
        case_id: 病例ID
        label_id: 标签ID
        output_image_dir: 图像输出目录
        output_label_dir: 标签输出目录
        image_props: 图像的ITK属性
        label_props: 标签的ITK属性
    """
    z_min, z_max, y_min, y_max, x_min, x_max = bbox
    
    # 裁剪图像和标签
    cropped_image = image_data[z_min:z_max, y_min:y_max, x_min:x_max]
    cropped_label = label_data[z_min:z_max, y_min:y_max, x_min:x_max]
    
    # 创建输出文件名
    image_filename = f"case_{case_id}_box_{label_id}_0000.nii.gz"
    label_filename = f"case_{case_id}_box_{label_id}.nii.gz"
    
    # 转换为ITK图像
    cropped_image_itk = sitk.GetImageFromArray(cropped_image)
    cropped_label_itk = sitk.GetImageFromArray(cropped_label)
    
    # 计算新的原点（考虑裁剪偏移）
    original_spacing = image_props['spacing']
    original_origin = image_props['origin']
    original_direction = image_props['direction']
    
    # 计算偏移量（注意ITK坐标系的顺序）
    offset = np.array([x_min, y_min, z_min]) * np.array(original_spacing)
    new_origin = np.array(original_origin) + offset
    
    # 设置图像属性
    cropped_image_itk.SetSpacing(original_spacing)
    cropped_image_itk.SetOrigin(tuple(new_origin))
    cropped_image_itk.SetDirection(original_direction)
    
    cropped_label_itk.SetSpacing(original_spacing)
    cropped_label_itk.SetOrigin(tuple(new_origin))
    cropped_label_itk.SetDirection(original_direction)
    
    # 保存文件
    sitk.WriteImage(cropped_image_itk, str(output_image_dir / image_filename))
    sitk.WriteImage(cropped_label_itk, str(output_label_dir / label_filename))
    
    logger.info(f"保存 {image_filename} 和 {label_filename}, 大小: {cropped_image.shape}")

def process_case(case_id, images_dir, labels_dir, output_image_dir, output_label_dir):
    """
    处理单个病例
    
    Args:
        case_id: 病例ID（不带扩展名）
        images_dir: 图像目录
        labels_dir: 标签目录  
        output_image_dir: 输出图像目录
        output_label_dir: 输出标签目录
    """
    logger.info(f"处理病例: {case_id}")
    
    # 构建文件路径
    image_path = images_dir / f"{case_id}_0000.nii.gz"
    label_path = labels_dir / f"{case_id}.nii.gz"
    json_path = labels_dir / f"{case_id}.json"
    
    # 检查文件是否存在
    if not image_path.exists():
        logger.warning(f"图像文件不存在: {image_path}")
        return
    
    if not label_path.exists():
        logger.warning(f"标签文件不存在: {label_path}")
        return
    
    if not json_path.exists():
        logger.warning(f"JSON文件不存在: {json_path}")
        return
    
    # 加载图像和标签
    try:
        image_itk = sitk.ReadImage(str(image_path))
        label_itk = sitk.ReadImage(str(label_path))
        
        image_data = sitk.GetArrayFromImage(image_itk)  # [Z, Y, X]
        label_data = sitk.GetArrayFromImage(label_itk)  # [Z, Y, X]
        
        # 获取图像属性
        image_props = {
            'spacing': image_itk.GetSpacing(),
            'origin': image_itk.GetOrigin(), 
            'direction': image_itk.GetDirection()
        }
        
        label_props = {
            'spacing': label_itk.GetSpacing(),
            'origin': label_itk.GetOrigin(),
            'direction': label_itk.GetDirection()
        }
        
        logger.info(f"加载图像: {image_data.shape}, 标签: {label_data.shape}")
        
    except Exception as e:
        logger.error(f"加载文件失败 {case_id}: {e}")
        return
    
    # 加载JSON标注
    try:
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        instances = json_data.get('instances', {})
    except Exception as e:
        logger.error(f"加载JSON文件失败 {case_id}: {e}")
        return
    
    # 获取标签数据中的唯一值
    unique_labels = np.unique(label_data)
    unique_labels = unique_labels[unique_labels > 0]  # 排除背景
    
    logger.info(f"发现 {len(unique_labels)} 个非零标签: {unique_labels}")
    
    # 处理每个非零标签
    for label_id in unique_labels:
        # 获取包围盒
        bbox = get_bounding_box_3d(label_data, label_id)
        if bbox is None:
            continue
            
        logger.info(f"标签 {label_id} 原始包围盒: {bbox}")
        
        # 扩展包围盒
        expanded_bbox = expand_bbox(bbox, expansion=4, shape=label_data.shape)
        logger.info(f"标签 {label_id} 扩展包围盒: {expanded_bbox}")
        
        # 裁剪并保存区域
        try:
            crop_and_save_region(
                image_data, label_data, expanded_bbox, 
                case_id, label_id, output_image_dir, output_label_dir,
                image_props, label_props
            )
        except Exception as e:
            logger.error(f"处理标签 {label_id} 失败: {e}")

def main():
    """主函数"""
    # 设置路径
    base_dir = Path("det_data/Task102_vesspe200/raw_splitted")
    images_dir = base_dir / "imagesTr"
    labels_dir = base_dir / "labelsTr"
    
    output_image_dir = Path("/home/zjc/data/image")
    output_label_dir = Path("/home/zjc/data/label")
    
    # 确保输出目录存在
    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取所有标签文件(.nii.gz格式)
    label_files = list(labels_dir.glob("*.nii.gz"))
    case_ids = [f.name.replace('.nii.gz', '') for f in label_files]  # 移除.nii.gz后缀，得到纯case ID
    
    logger.info(f"找到 {len(case_ids)} 个病例")
    
    # 处理每个病例
    for case_id in tqdm(case_ids, desc="处理病例"):
        process_case(case_id, images_dir, labels_dir, output_image_dir, output_label_dir)
    
    logger.info("处理完成!")

if __name__ == "__main__":
    main() 