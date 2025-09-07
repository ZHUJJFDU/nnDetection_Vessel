"""
数据模块工厂函数

这个模块提供了创建数据模块的工厂函数，能够根据模型类型自动选择合适的数据模块。
"""

from typing import Dict, Any, Optional, Union, Type

from loguru import logger
from pathlib import Path

from nndet.io.datamodule import MODULE_REGISTRY
from nndet.io.datamodule.bg_module import Datamodule
from nndet.io.datamodule.vessel_module import VesselDatamodule


def create_datamodule(config: Dict[str, Any], model_type: Optional[str] = None):
    """
    创建适合指定模型类型的数据模块
    
    Args:
        config: 数据模块配置
        model_type: 模型类型名称，例如'RetinaUNetV004'。如果为None，则根据vessel_dir自动选择
        
    Returns:
        数据模块实例
    """
    # 检查配置是否包含血管分割目录
    vessel_dir = config.get('vessel_dir', None)
    has_vessel_dir = vessel_dir is not None and Path(vessel_dir).exists()
    
    if has_vessel_dir:
        logger.info(f"指定了血管分割目录: {vessel_dir}")
    
    # 如果指定了模型类型，检查是否需要血管分割数据
    needs_vessel_data = False
    
    if model_type is not None:
        # 检查模型类型是否需要血管分割数据
        needs_vessel_data = model_type in ['RetinaUNetV004']
        
        if needs_vessel_data and not has_vessel_dir:
            logger.warning(f"模型 {model_type} 需要血管分割数据，但未指定vessel_dir，将使用标准数据模块")
    
    # 选择数据模块类型
    if (needs_vessel_data and has_vessel_dir) or (has_vessel_dir and model_type is None):
        logger.info(f"使用血管分割数据模块，血管分割目录: {vessel_dir}")
        module_cls = MODULE_REGISTRY.get('VesselDatamodule')
        # 创建包含vessel_dir的配置
        updated_config = config.copy()
        updated_config['vessel_dir'] = vessel_dir
    else:
        logger.info("使用标准数据模块")
        module_cls = Datamodule
        updated_config = config
    
    # 创建数据模块实例
    if hasattr(module_cls, 'from_config'):
        return module_cls.from_config(updated_config)
    else:
        return module_cls(
            plan=updated_config.get('plan', {}),
            augment_cfg=updated_config.get('augment_cfg', {}),
            data_dir=updated_config.get('data_dir', ''),
            fold=updated_config.get('fold', 0),
        )


def check_model_vessel_compatibility(model_type: str) -> bool:
    """
    检查模型类型是否兼容血管分割数据
    
    Args:
        model_type: 模型类型名称
        
    Returns:
        bool: 如果模型兼容血管分割数据，则为True
    """
    # 兼容血管分割数据的模型列表
    vessel_compatible_models = [
        'RetinaUNetV004',
        # 未来可以添加更多兼容模型
    ]
    
    return model_type in vessel_compatible_models 