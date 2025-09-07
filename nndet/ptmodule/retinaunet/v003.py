from nndet.arch.attention import AttenRetinaUNetModule

from nndet.core.boxes.matcher import ATSSMatcher, IoUMatcher
from nndet.arch.heads.classifier import BCECLassifier, CEClassifier, FocalClassifier
from nndet.arch.heads.regressor import GIoURegressor, L1Regressor
from nndet.arch.heads.comb import DetectionHeadHNMNative, DetectionHeadHNMNativeRegAll, DetectionHeadHNM, DetectionHeadHNMRegAll
from nndet.arch.heads.segmenter import DiCESegmenterFgBg
from nndet.arch.conv import ConvInstanceRelu, ConvGroupRelu, ConvBatchLeaky, ConvDilatedBatchRelu

from nndet.ptmodule import MODULE_REGISTRY


@MODULE_REGISTRY.register
class RetinaUNetV003CBAMnew(AttenRetinaUNetModule):
    """
    带有改进ECA通道注意力的RetinaUNet V003版本
    
    此版本使用以下组件：
    - ECA通道注意力机制替代原始CBAM中的通道注意力
    - 保留原始CBAM的空间注意力机制
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
    attention_type = "cbam_eca"  # 使用自定义的ECA-CBAM注意力
    attention_params = {
        "kernel_size": 7,  # 空间注意力的卷积核大小
    }


@MODULE_REGISTRY.register
class RetinaUNetV003CBAM(AttenRetinaUNetModule):
    """
    带有CBAM注意力机制的RetinaUNet V003版本
    
    此版本使用以下组件：
    - 批归一化的卷积块作为基础网络卷积
    - 分组归一化的卷积块作为头部网络卷积
    - CBAM注意力机制用于特征图增强
    - ATSS匹配器进行锚点匹配
    - BCE分类器进行目标分类
    - GIoU回归器进行边界框回归
    - 原生硬负例挖掘(HNM)的检测头部
    - 前景背景Dice分割器
    """
    # 基础网络使用的卷积块类型
    base_conv_cls = ConvDilatedBatchRelu # 使用批归一化的卷积块
    
    # 头部网络使用的卷积块类型
    head_conv_cls = ConvGroupRelu  # 使用分组归一化的卷积块
    
    # 检测头部类型
    head_cls = DetectionHeadHNMNative  # 带有原生硬负例挖掘的检测头部
    
    # 分类器类型
    head_classifier_cls = FocalClassifier  # 使用二元交叉熵损失的分类器
    
    # 回归器类型
    head_regressor_cls = GIoURegressor  # 使用GIoU损失的回归器
    
    # 匹配器类型
    matcher_cls = ATSSMatcher  # 自适应训练样本选择匹配器
    
    # 分割器类型
    segmenter_cls = DiCESegmenterFgBg  # 使用Dice损失的前景背景分割器
    
    # 注意力类型和参数
    attention_type = "cbam"  # CBAM = 通道注意力 + 空间注意力
    attention_params = {
        "reduction_ratio": 16,  # 通道减少比例
        "kernel_size": 7,       # 空间注意力的卷积核大小
    }


