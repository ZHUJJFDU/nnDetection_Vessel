"""
带有注意力机制的RetinaUNet模块

这个模块提供了一个带有注意力机制的RetinaUNet骨架，可以通过
子类化和设置不同的属性来实现各种注意力变体。
"""

from nndet.ptmodule.retinaunet.base import RetinaUNetModule
from nndet.arch.decoder import AttentionUFPNModular
from nndet.arch.conv import Generator


class AttenRetinaUNetModule(RetinaUNetModule):
    """
    带注意力机制的RetinaUNet模块基类
    
    这个类为RetinaUNet架构添加了注意力机制，具体的注意力类型
    可以通过子类化并覆盖attention_type和attention_params属性来设置。
    
    Attributes:
        decoder_cls (class): 解码器类，默认为AttentionUFPNModular
        attention_type (str): 注意力类型，可以是'cbam', 'channel', 'spatial'
        attention_params (dict): 注意力参数，不同注意力类型需要不同的参数
    """
    # 使用带注意力的解码器类
    decoder_cls = AttentionUFPNModular
    
    # 默认注意力类型和参数 (可以被子类覆盖)
    attention_type = None
    attention_params = {}
    
    @classmethod
    def _build_decoder(cls, plan_arch, model_cfg, encoder):
        """
        构建带有注意力机制的解码器
        
        此方法覆盖父类的_build_decoder方法，添加注意力类型和参数。
        
        Args:
            plan_arch: 架构规划
            model_cfg: 模型配置
            encoder: 编码器实例
            
        Returns:
            AttentionUFPNModular: 带有注意力机制的解码器
        """
        # 获取注意力类型和参数
        attention_type = cls.attention_type
        attention_params = cls.attention_params
        
        # 创建卷积生成器，与基类相同的方式
        conv = Generator(cls.base_conv_cls, plan_arch["dim"])
        
        # 构建解码器参数
        decoder_kwargs = {
            "conv": conv,
            "conv_kernels": plan_arch["conv_kernels"],
            "strides": encoder.get_strides(),
            "in_channels": encoder.get_channels(),
            "decoder_levels": plan_arch["decoder_levels"],
            "fixed_out_channels": plan_arch["fpn_channels"],
            # 加入注意力机制相关配置
            "attention_type": attention_type,
            "attention_params": attention_params,
        }
        
        # 更新model_cfg中的decoder_kwargs
        decoder_kwargs.update(model_cfg['decoder_kwargs'])
        
        # 创建并返回解码器
        decoder = cls.decoder_cls(**decoder_kwargs)
        
        return decoder 