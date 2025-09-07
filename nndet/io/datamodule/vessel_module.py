"""
血管分割数据模块

这个模块提供了一个通用的数据加载机制，能够加载和处理血管分割标签，
同时保持与现有模型的兼容性。
"""

import os
from pathlib import Path
from collections import OrderedDict
from typing import List, Dict, Optional, Union, Any, Iterable

import numpy as np
from loguru import logger

from nndet.io.datamodule.bg_module import Datamodule
from nndet.io.load import load_pickle
from nndet.io.utils import load_dataset_id
from nndet.io.datamodule import MODULE_REGISTRY, DATALOADER_REGISTRY
from nndet.io.datamodule.vessel_loader import DataLoader3DVesselFast


@MODULE_REGISTRY.register
class VesselDatamodule(Datamodule):
    """
    血管分割数据模块
    
    此模块扩展了基本的Datamodule类，添加了对血管分割数据的支持。
    它能够兼容所有现有的模型，只有需要血管分割数据的模型才会使用它。
    """
    def __init__(self, 
                 plan: dict,
                 augment_cfg: dict,
                 data_dir: os.PathLike,
                 fold: int = 0,
                 vessel_dir: Optional[os.PathLike] = None,
                 **kwargs):
        """
        初始化血管分割数据模块
        
        Args:
            plan: 配置计划
            augment_cfg: 数据增强配置
            data_dir: 数据目录路径
            fold: 当前折数
            vessel_dir: 血管分割数据目录，如果为None，则尝试使用data_dir
        """
        # 保存血管分割目录 - 在调用super()之前设置，确保do_split可以使用
        self.vessel_dir = Path(vessel_dir) if vessel_dir is not None else Path(data_dir)
        
        # 标记是否有血管分割数据
        self.has_vessel_data = False
        
        # 调用父类初始化
        super().__init__(plan=plan, augment_cfg=augment_cfg, data_dir=data_dir, fold=fold, **kwargs)
        
        # 检查血管分割数据
        self._check_vessel_data()
        
    def _check_vessel_data(self):
        """检查是否存在血管分割数据"""
        # 检查数据集中的第一个样本
        if len(self.dataset_tr) > 0:
            first_key = list(self.dataset_tr.keys())[0]
            vessel_file = self.dataset_tr[first_key].get('vessel_file')
            
            if vessel_file and Path(vessel_file).exists():
                self.has_vessel_data = True
                logger.info(f"血管分割数据可用: {self.has_vessel_data}")
            else:
                logger.warning("未找到血管分割数据，将继续但模型无法使用血管引导注意力")
        
    def do_split(self) -> None:
        """
        加载或创建数据集划分，同时确保血管分割数据也被包含
        """
        # 首先执行原始的划分逻辑
        super().do_split()
        
        # 如果指定了单独的血管分割目录，则需要更新数据集
        if self.vessel_dir != self.data_dir:
            vessel_dataset = load_dataset_id(self.vessel_dir)
            
            # 更新训练集
            for key in self.dataset_tr:
                if key in vessel_dataset:
                    self.dataset_tr[key]['vessel_file'] = vessel_dataset[key].get('vessel_file')
            
            # 更新验证集
            for key in self.dataset_val:
                if key in vessel_dataset:
                    self.dataset_val[key]['vessel_file'] = vessel_dataset[key].get('vessel_file')
            
            # 检查更新后的数据集是否包含血管分割数据
            self._check_vessel_data()
    
    def train_dataloader(self) -> Iterable:
        """
        创建训练数据加载器，如果有血管分割数据，则使用DataLoader3DVesselFast
        
        Returns:
            Iterable: 用于训练的数据加载器
        """
        # 如果有血管分割数据，使用特殊的数据加载器
        if self.has_vessel_data:
            dataloader_cls = DataLoader3DVesselFast
            logger.info(f"使用血管分割数据加载器 DataLoader3DVesselFast 进行训练")
        else:
            # 否则使用常规数据加载器
            dataloader_cls = DATALOADER_REGISTRY.get(self.dataloader)
            logger.info(f"使用标准数据加载器 {self.dataloader} 进行训练")

        dl_tr = dataloader_cls(
            data=self.dataset_tr,
            batch_size=self.batch_size,
            patch_size_generator=self.patch_size_generator,
            patch_size_final=self.patch_size,
            oversample_foreground_percent=self.augment_cfg[
                "oversample_foreground_percent"],
            pad_mode="constant",
            num_batches_per_epoch=self.augment_cfg[
                "num_train_batches_per_epoch"],
            **self.dataloader_kwargs,
        )

        from nndet.io.datamodule.bg_module import get_augmenter
        tr_gen = get_augmenter(
            dataloader=dl_tr,
            transform=self.augmentation.get_training_transforms(),
            num_processes=min(int(self.augment_cfg.get('num_threads', 12)), 16) - 1,
            num_cached_per_queue=self.augment_cfg.get('num_cached_per_thread', 2),
            multiprocessing=self.augment_cfg.get("multiprocessing", True),
            seeds=None,
            pin_memory=True,
        )
        logger.info("TRAINING KEYS:\n %s" % (str(self.dataset_tr.keys())))
        return tr_gen
        
    def get_model_input_info(self) -> Dict[str, Any]:
        """
        获取模型输入信息，用于配置模型
        
        Returns:
            Dict: 包含模型输入信息的字典
        """
        info = super().get_model_input_info() if hasattr(super(), 'get_model_input_info') else {}
        
        # 添加血管分割信息
        info.update({
            'has_vessel_data': self.has_vessel_data,
        })
        
        return info
        
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'VesselDatamodule':
        """
        从配置创建数据模块
        
        Args:
            config: 配置字典
            
        Returns:
            VesselDatamodule: 创建的数据模块实例
        """
        vessel_dir = config.get('vessel_dir', None)
        
        return cls(
            plan=config.get('plan', {}),
            augment_cfg=config.get('augment_cfg', {}),
            data_dir=config.get('data_dir', ''),
            fold=config.get('fold', 0),
            vessel_dir=vessel_dir,
        ) 