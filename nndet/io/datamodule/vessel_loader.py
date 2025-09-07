"""
血管分割数据加载器

这个模块提供了一个扩展的DataLoader3DFast类，增加了对血管分割数据的加载和处理。
"""

import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Union, Sequence, Tuple, Optional

from loguru import logger

from nndet.io.datamodule import DATALOADER_REGISTRY
from nndet.io.datamodule.bg_loader import DataLoader3DFast
from nndet.io.patching import save_get_crop


@DATALOADER_REGISTRY.register
class DataLoader3DVesselFast(DataLoader3DFast):
    """
    扩展的DataLoader3DFast类，增加了对血管分割数据的支持。
    
    这个类在生成批次时会额外加载血管分割数据，如果可用。
    对于没有血管分割数据的情况，会自动优雅降级，返回None作为vessel值。
    """
    
    def __init__(self, *args, **kwargs):
        """初始化血管分割数据加载器，参数与DataLoader3DFast相同"""
        super().__init__(*args, **kwargs)
        # 检查是否有血管分割数据可用
        self.has_vessel_data = self._check_vessel_data()
        # 如果有血管数据，确定血管数据的形状
        if self.has_vessel_data:
            self.vessel_shape_batch = self._determine_vessel_shape()
            logger.info(f"血管分割数据可用，形状: {self.vessel_shape_batch}")
    
    def _check_vessel_data(self) -> bool:
        """
        检查第一个样本是否有vessel_file键，并且文件是否存在
        
        Returns:
            bool: 是否有血管分割数据
        """
        if not self._data:
            logger.warning("数据为空，无法检查血管分割数据")
            return False
            
        k = list(self._data.keys())[0]
        has_vessel_key = 'vessel_file' in self._data[k]
        
        if not has_vessel_key:
            logger.warning(f"数据项 {k} 中没有'vessel_file'键")
            return False
            
        vessel_file = self._data[k]['vessel_file']
        if vessel_file is None:
            logger.warning(f"数据项 {k} 的'vessel_file'为None")
            return False
            
        vessel_path = Path(vessel_file)
        if not vessel_path.is_file():
            logger.warning(f"血管文件不存在: {vessel_file}")
            # 尝试检查常见的命名模式
            # 例如，把.npy替换为.nii.gz等可能的情况
            alt_paths = [
                vessel_path.with_suffix('.nii.gz'),
                vessel_path.with_name(vessel_path.stem + '_vessel.npy'),
                vessel_path.with_name(vessel_path.stem + '_vessel.nii.gz'),
            ]
            for alt_path in alt_paths:
                if alt_path.is_file():
                    logger.info(f"找到替代的血管文件: {alt_path}")
                    # 更新数据字典
                    self._data[k]['vessel_file'] = str(alt_path)
                    return True
            return False
            
        logger.info(f"找到血管文件: {vessel_file}")
        return True
    
    def _determine_vessel_shape(self) -> Tuple[int, ...]:
        """
        确定血管分割数据的形状
        
        Returns:
            Tuple[int, ...]: 血管数据的形状（包括批次维度）
        """
        k = list(self._data.keys())[0]
        vessel_file = self._data[k]['vessel_file']
        
        if not vessel_file or not Path(vessel_file).is_file():
            # 如果没有血管数据，使用与分割数据相同的形状
            return self.seg_shape_batch
        
        vessel = np.load(str(vessel_file), self.memmap_mode, allow_pickle=False)
        num_vessel_channels = vessel.shape[0]  # 通常为1，表示血管mask
        
        vessel_shape = (self.batch_size, num_vessel_channels, *self.patch_size_generator)
        return vessel_shape
    
    def generate_train_batch(self) -> Dict[str, Any]:
        """
        生成一个批次，包括标准的数据和分割，以及额外的血管分割数据（如果可用）
        
        Returns:
            Dict: 包含以下键的批次字典:
                `data` (np.ndarray): 数据
                `seg` (np.ndarray): 无序实例分割
                `vessel` (np.ndarray or None): 血管分割掩码，如果没有则为None
                `instances`, `properties`, `keys`: 与父类相同
        """
        # 获取基本批次数据
        batch_dict = super().generate_train_batch()
        
        # 如果没有血管数据，直接返回基本批次
        if not self.has_vessel_data:
            batch_dict['vessel'] = None
            return batch_dict
        
        # 创建血管批次数组
        vessel_batch = np.zeros(self.vessel_shape_batch, dtype=float)
        
        # 为批次中的每个样本加载血管数据
        selected_cases, selected_instances = self.select()
        for batch_idx, (case_id, instance_id) in enumerate(zip(selected_cases, selected_instances)):
            # 获取血管文件路径
            vessel_file = self._data[case_id].get('vessel_file')
            
            # 如果该样本有血管数据，则加载
            if vessel_file and Path(vessel_file).is_file():
                try:
                    logger.debug(f"加载血管文件: {vessel_file}")
                    case_vessel = np.load(vessel_file, self.memmap_mode, allow_pickle=True)
                    logger.debug(f"血管数据形状: {case_vessel.shape}")
                    
                    # 使用与数据和分割相同的裁剪区域
                    if instance_id < 0:
                        candidates = self.load_candidates(case_id=case_id, fg_crop=False)
                        crop = self.get_bg_crop(
                            case_data=np.zeros_like(case_vessel),  # 只需要相同形状
                            case_seg=np.zeros_like(case_vessel),  # 只需要相同形状
                            properties={},  # 不需要属性
                            case_id=case_id,
                            candidates=candidates,
                        )
                    else:
                        candidates = self.load_candidates(case_id=case_id, fg_crop=True)
                        crop = self.get_fg_crop(
                            case_data=np.zeros_like(case_vessel),  # 只需要相同形状
                            case_seg=np.zeros_like(case_vessel),  # 只需要相同形状
                            properties={},  # 不需要属性
                            case_id=case_id,
                            instance_id=instance_id,
                            candidates=candidates,
                        )
                    
                    # 裁剪血管数据
                    vessel_batch[batch_idx] = save_get_crop(
                        case_vessel,
                        crop=crop,
                        mode='constant',
                        constant_values=0,  # 用0填充
                    )[0]
                except Exception as e:
                    logger.error(f"加载血管数据失败 ({vessel_file}): {e}")
                    vessel_batch[batch_idx] = 0  # 如果加载失败，使用零填充
            else:
                logger.warning(f"案例 {case_id} 没有有效的血管文件")
        
        # 将血管数据添加到批次字典
        batch_dict['vessel'] = vessel_batch
        
        return batch_dict 