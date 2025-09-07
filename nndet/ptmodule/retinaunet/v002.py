from nndet.ptmodule.retinaunet.base import RetinaUNetModule

from nndet.core.boxes.matcher import ATSSMatcher, IoUMatcher
from nndet.arch.heads.classifier import BCECLassifier, CEClassifier, FocalClassifier
from nndet.arch.heads.regressor import GIoURegressor, L1Regressor
from nndet.arch.heads.comb import DetectionHeadHNMNative, DetectionHeadHNMNativeRegAll, DetectionHeadHNM, DetectionHeadHNMRegAll
from nndet.arch.heads.segmenter import DiCESegmenterFgBg
from nndet.arch.conv import ConvInstanceRelu, ConvGroupRelu, ConvBatchLeaky, ConvDilatedBatchRelu

from nndet.ptmodule import MODULE_REGISTRY


@MODULE_REGISTRY.register
class RetinaUNetV002(RetinaUNetModule):
    """
    RetinaUNet的V002版本实现
    
    这个版本使用特定的组件组合来构建检测网络：
    - 使用空洞卷积块作为基础网络组件，增强小目标检测能力
    - 使用分组归一化的卷积块作为检测头部组件
    - 采用原生硬负例挖掘(HNM)的检测头部
    - 使用二元交叉熵(BCE)作为分类器
    - 使用GIoU损失作为回归器
    - 使用ATSS匹配器进行锚点匹配
    - 使用前景背景的Dice分割器
    """
    # 基础网络使用的卷积块类型
    base_conv_cls = ConvDilatedBatchRelu  # 使用空洞卷积块增强小目标检测
    
    # 头部网络使用的卷积块类型
    head_conv_cls = ConvGroupRelu  # 使用分组归一化的卷积块
    
    # 检测头部类型
    head_cls = DetectionHeadHNMNative  # 带有原生硬负例挖掘的检测头部
    
    # 分类器类型
    head_classifier_cls = BCECLassifier  # 使用二元交叉熵损失的分类器
    
    # 回归器类型
    head_regressor_cls = GIoURegressor  # 使用GIoU损失的回归器
    
    # 匹配器类型
    matcher_cls = ATSSMatcher  # 自适应训练样本选择匹配器
    
    # 分割器类型
    segmenter_cls = DiCESegmenterFgBg  # 使用Dice损失的前景背景分割器
