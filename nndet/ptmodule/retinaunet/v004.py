"""
RetinaUNet V004 实现 - 带有血管分割引导注意力的版本
"""

from nndet.arch.vesselattention import VesselAttenRetinaUNetModule
from nndet.ptmodule.retinaunet.v003 import RetinaUNetV003CBAM

from nndet.core.boxes.matcher import ATSSMatcher
from nndet.arch.heads.classifier import FocalClassifier
from nndet.arch.heads.regressor import GIoURegressor
from nndet.arch.heads.comb import DetectionHeadHNMNative
from nndet.arch.heads.segmenter import DiCESegmenterFgBg
from nndet.arch.conv import ConvGroupRelu, ConvDilatedBatchRelu

from nndet.ptmodule import MODULE_REGISTRY


@MODULE_REGISTRY.register
class RetinaUNetV004(VesselAttenRetinaUNetModule):
    """
    带有血管引导注意力的RetinaUNet V004版本
    
    此版本使用以下组件：
    - 血管引导注意力机制
    - 批归一化的卷积块作为基础网络卷积
    - 分组归一化的卷积块作为头部网络卷积
    - ATSS匹配器进行锚点匹配
    - Focal Loss分类器进行目标分类
    - GIoU回归器进行边界框回归
    - 原生硬负例挖掘(HNM)的检测头部
    - 前景背景Dice分割器
    """
    # 基础网络使用的卷积块类型
    base_conv_cls = ConvDilatedBatchRelu  # 使用批归一化的卷积块
    
    # 头部网络使用的卷积块类型
    head_conv_cls = ConvGroupRelu  # 使用分组归一化的卷积块
    
    # 检测头部类型
    head_cls = DetectionHeadHNMNative  # 带有原生硬负例挖掘的检测头部
    
    # 分类器类型
    head_classifier_cls = FocalClassifier  # 使用Focal Loss的分类器
    
    # 回归器类型
    head_regressor_cls = GIoURegressor  # 使用GIoU损失的回归器
    
    # 匹配器类型
    matcher_cls = ATSSMatcher  # 自适应训练样本选择匹配器
    
    # 分割器类型
    segmenter_cls = DiCESegmenterFgBg  # 使用Dice损失的前景背景分割器
    
    # 注意力类型和参数
    attention_type = "vessel"  # 使用血管引导注意力
    attention_params = {
        "kernel_size": 7,      # 空间注意力的卷积核大小
        "use_vessel": True,    # 启用血管掩码指导
        "fusion_mode": "concatenation"  # 注意力融合模式：连接、相加、相乘
    }

