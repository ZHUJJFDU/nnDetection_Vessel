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

from pathlib import Path

import numpy as np
import SimpleITK as sitk
from itertools import product


from typing import Sequence, Union, Tuple


def create_circle_mask_itk(image_itk: sitk.Image,
                           world_centers: Sequence[Sequence[float]],
                           world_rads: Sequence[float],
                           ndim: int = 3,
                           ) -> sitk.Image:
    """
    根据中心点和半径创建带有圆形的ITK图像

    参数:
        image_itk: 原始图像（用于坐标系参考）
        world_centers: 世界坐标系中心点序列 (x, y, z)
        world_rads: 半径序列
        ndim: 空间维度数

    返回:
        sitk.Image: 带有圆形的掩码图像
    """
    image_np = sitk.GetArrayFromImage(image_itk)
    min_spacing = min(image_itk.GetSpacing())

    if image_np.ndim > ndim:
        image_np = image_np[0]
    mask_np = np.zeros_like(image_np).astype(np.uint8)

    for _id, (world_center, world_rad) in enumerate(zip(world_centers, world_rads), start=1):
        check_rad = (world_rad / min_spacing) * 1.5  # 添加一些缓冲区
        bounds = []
        center = image_itk.TransformPhysicalPointToContinuousIndex(world_center)[::-1]
        for ax, c in enumerate(center):
            bounds.append((
                max(0, int(c - check_rad)),
                min(mask_np.shape[ax], int(c + check_rad)),
            ))
        coord_box = product(*[list(range(b[0], b[1])) for b in bounds])

        # 遍历每个像素位置
        for coord in coord_box:
            world_coord = image_itk.TransformIndexToPhysicalPoint(tuple(reversed(coord)))  # 反转顺序为x, y, z（适用于sitk）
            dist = np.linalg.norm(np.array(world_coord) - np.array(world_center))
            if dist <= world_rad:
                mask_np[tuple(coord)] = _id
        assert mask_np.max() == _id

    mask_itk = sitk.GetImageFromArray(mask_np)
    mask_itk.SetOrigin(image_itk.GetOrigin())
    mask_itk.SetDirection(image_itk.GetDirection())
    mask_itk.SetSpacing(image_itk.GetSpacing())
    return mask_itk


def load_sitk(path: Union[Path, str], **kwargs) -> sitk.Image:
    """
    加载图像的函数接口，使用SimpleITK

    参数:
        path: 要加载的文件路径
        **kwargs: 传递给sitk.ReadImage的额外参数

    返回:
        sitk.Image: 加载的SimpleITK图像
    """
    return sitk.ReadImage(str(path), **kwargs)


def load_sitk_as_array(path: Union[Path, str], **kwargs) -> Tuple[np.ndarray, dict]:
    """
    加载SimpleITK图像并将其转换为数组的函数接口

    参数:
        path: 要加载的文件路径
        **kwargs: 传递给sitk.ReadImage的额外参数

    返回:
        np.ndarray: 加载的图像数据
        dict: 加载的元数据
    """
    img_itk = load_sitk(path, **kwargs)
    meta = {key: img_itk.GetMetaData(key) for key in img_itk.GetMetaDataKeys()}
    return sitk.GetArrayFromImage(img_itk), meta
