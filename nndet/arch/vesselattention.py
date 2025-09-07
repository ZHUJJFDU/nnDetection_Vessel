"""
血管引导注意力的RetinaUNet模块

这个模块提供了一个带有血管引导注意力机制的RetinaUNet骨架，
能够利用血管分割掩码增强对关键结构的感知。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, Any, List, Optional, Union
from loguru import logger

from nndet.arch.decoder.attention_fpn import AttentionUFPNModular
from nndet.arch.attention import AttenRetinaUNetModule


class VascularGuidedAttention(nn.Module):
    """
    血管引导注意力模块
    
    该模块使用血管分割掩码来引导网络关注肺血管区域，
    从而提高肺栓塞检测性能。
    """
    def __init__(self, channels: int, dim: int = 3, reduction_ratio: int = 16):
        """
        初始化血管引导注意力模块
        
        Args:
            channels: 输入通道数
            dim: 空间维度
            reduction_ratio: 通道减少比例
        """
        super().__init__()
        self.dim = dim
        
        # 通道注意力部分
        self.avg_pool = self._get_adaptive_avg_pool(dim)
        self.max_pool = self._get_adaptive_max_pool(dim)
        
        reduced_channels = max(1, channels // reduction_ratio)
        self.fc = nn.Sequential(
            self._get_conv(dim, channels, reduced_channels, 1, 0),
            nn.ReLU(inplace=True),
            self._get_conv(dim, reduced_channels, channels, 1, 0)
        )
        
        # 用于处理血管掩码的卷积
        self.vessel_conv = nn.Sequential(
            self._get_conv(dim, 1, 16, 3, 1),
            nn.ReLU(inplace=True),
            self._get_conv(dim, 16, 1, 1, 0),
            nn.Sigmoid()
        )
        
        # 组合注意力的卷积
        self.fusion_conv = self._get_conv(dim, 2, 1, 3, 1)
        self.sigmoid = nn.Sigmoid()
    
    def _get_adaptive_avg_pool(self, dim):
        if dim == 3:
            return nn.AdaptiveAvgPool3d(1)
        else:
            return nn.AdaptiveAvgPool2d(1)
    
    def _get_adaptive_max_pool(self, dim):
        if dim == 3:
            return nn.AdaptiveMaxPool3d(1)
        else:
            return nn.AdaptiveMaxPool2d(1)
    
    def _get_conv(self, dim, in_ch, out_ch, kernel_size, padding):
        if dim == 3:
            return nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=True)
        else:
            return nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=True)
    
    def forward(self, x, vessel_mask=None):
        """
        前向传播
        
        Args:
            x: 输入特征图
            vessel_mask: 血管分割掩码，如果为None，则退化为普通通道注意力
            
        Returns:
            增强后的特征图
        """
        # 通道注意力
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        channel_attention = self.sigmoid(avg_out + max_out)
        
        # 应用通道注意力
        channel_refined = x * channel_attention
        
        # 如果有血管掩码，则应用血管引导注意力
        if vessel_mask is not None:
            # 确保血管掩码与特征图大小一致
            if vessel_mask.shape[2:] != x.shape[2:]:
                vessel_mask = F.interpolate(vessel_mask, size=x.shape[2:], mode='nearest')
            
            # 增强血管区域
            vessel_attention = self.vessel_conv(vessel_mask)
            
            # 组合通道注意力和血管注意力
            combined_attention = torch.cat([channel_attention.expand_as(x), vessel_attention], dim=1)
            attention_map = self.sigmoid(self.fusion_conv(combined_attention))
            
            # 应用组合注意力
            return x * attention_map
        
        # 如果没有血管掩码，则只返回通道注意力结果
        return channel_refined


class VesselAttentionFPN(AttentionUFPNModular):
    """
    带有血管引导注意力的特征金字塔网络
    
    此类扩展了AttentionUFPNModular，添加了对血管分割掩码的处理能力。
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 为每个级别创建血管引导注意力模块
        self.vessel_attention = nn.ModuleDict()
        for level in range(len(self.decoder_levels)):
            self.vessel_attention[f"P{level}"] = VascularGuidedAttention(
                channels=self.out_channels[level],
                dim=self.dim,
                reduction_ratio=self.attention_params.get("reduction_ratio", 16),
            )
    
    def forward(self, inp_seq: List[torch.Tensor], vessel_mask: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
        """
        带有血管引导注意力的前向传播
        
        Args:
            inp_seq: 特征图序列
            vessel_mask: 血管分割掩码
            
        Returns:
            List[torch.Tensor]: 增强后的特征图
        """
        # 正常FPN处理
        fpn_maps = self.forward_lateral(inp_seq)
        
        # 应用血管引导注意力
        enhanced_maps = []
        for idx, x in enumerate(fpn_maps):
            level_key = f"P{idx}"
            if level_key in self.vessel_attention and vessel_mask is not None:
                # 应用血管引导注意力
                enhanced_maps.append(self.vessel_attention[level_key](x, vessel_mask))
            else:
                enhanced_maps.append(x)
        
        # 继续常规FPN处理
        out_list = []  # 从最低分辨率到最高分辨率排序
        for idx, x in enumerate(reversed(enhanced_maps), 1):
            level = self.num_level - idx
            
            if idx != 1:  # 不是最后一层
                x = x + up  # 添加上采样特征
                
                if self.num_fusion > 0:
                    x = self.fusion_bottom_up[f"P{level}"](x)
                    
            if idx != self.num_level:  # 不是第一层
                up = self.up[f"P{level}"](x)
                
            out_list.append(x)
            
        return self.forward_out(reversed(out_list))


class VesselAttenRetinaUNetModule(AttenRetinaUNetModule):
    """
    带有血管引导注意力的RetinaUNet模块
    
    这个类扩展了AttenRetinaUNetModule，添加了处理血管分割掩码的能力，
    并将掩码传递给注意力机制用于增强特征学习。
    
    Attributes:
        attention_type (str): 应该设置为"vessel"以启用血管引导注意力
    """
    def __init__(self,
                 model_cfg: dict,
                 trainer_cfg: dict,
                 plan: dict,
                 **kwargs):
        """
        初始化血管引导注意力的RetinaUNet模块
        
        Args:
            model_cfg: 模型配置
            trainer_cfg: 训练器配置
            plan: 从规划阶段获得的参数
            **kwargs: 其他参数
        """
        super().__init__(
            model_cfg=model_cfg,
            trainer_cfg=trainer_cfg,
            plan=plan,
            **kwargs
        )
        
        # 检查注意力类型是否正确设置为vessel
        if self.attention_type != "vessel":
            logger.warning(f"VesselAttenRetinaUNetModule初始化时使用了非vessel注意力类型: {self.attention_type}")
    
    def forward(self, batch_dict: Union[Dict, torch.Tensor]) -> Union[Dict, torch.Tensor]:
        """
        前向传播，处理输入并通过网络传递
        
        Args:
            batch_dict: 批次字典或张量（用于模型摘要生成）
                如果是字典，应该包含'data'键和可选的'vessel'键
                
        Returns:
            Dict 或 Tensor: 处理后的输出
        """
        # 处理直接输入张量的情况（为模型摘要生成）
        if isinstance(batch_dict, torch.Tensor):
            # 创建一个临时的vessel_mask，形状匹配batch_dict
            # 这是为了模型摘要生成时避免通道不匹配错误
            logger.info(f"摘要生成: 创建临时vessel_mask，输入形状 {batch_dict.shape}")
            temp_vessel_mask = torch.ones((batch_dict.shape[0], 1, *batch_dict.shape[2:]), 
                                          device=batch_dict.device)
            
            # 编码器处理
            encoded_features = self.model.encoder(batch_dict)
            
            # 解码器处理（传递临时vessel_mask）
            features_maps_all = self.model.decoder(encoded_features, temp_vessel_mask)
            
            # 获取检测头使用的特征图
            feature_maps_head = [features_maps_all[i] for i in self.model.decoder_levels]
            
            # 检测头处理
            pred_detection = self.model.head(feature_maps_head)
            
            # 生成锚点
            anchors = self.model.anchor_generator(batch_dict, feature_maps_head)
            
            # 分割头处理（如果存在）
            pred_seg = self.model.segmenter(features_maps_all) if self.model.segmenter is not None else None
            
            # 返回结果
            return pred_detection
        
        # 处理正常训练时的字典输入
        if isinstance(batch_dict, dict):
            # 提取血管掩码 (如果存在)
            vessel_mask = None
            if 'vessel' in batch_dict and batch_dict['vessel'] is not None:
                if isinstance(batch_dict['vessel'], torch.Tensor):
                    vessel_mask = batch_dict['vessel']
                    logger.info(f"找到血管掩码，形状为 {vessel_mask.shape}")
                else:
                    logger.warning(f"batch_dict中的vessel不是张量类型，收到{type(batch_dict['vessel'])}，将被忽略")
            
            # 获取输入数据
            data = batch_dict["data"]
            
            # 编码器处理
            encoded_features = self.model.encoder(data)
            
            # 解码器处理（传递血管掩码）
            features_maps_all = self.model.decoder(encoded_features, vessel_mask)
            
            # 获取检测头使用的特征图
            feature_maps_head = [features_maps_all[i] for i in self.model.decoder_levels]
            
            # 检测头处理
            pred_detection = self.model.head(feature_maps_head)
            
            # 生成锚点
            anchors = self.model.anchor_generator(data, feature_maps_head)
            
            # 分割头处理（如果存在）
            pred_seg = self.model.segmenter(features_maps_all) if self.model.segmenter is not None else None
            
            # 返回结果
            return pred_detection, anchors, pred_seg
        
        logger.error(f"batch_dict应该是字典类型，但收到了{type(batch_dict)}")
        raise TypeError(f"batch_dict应该是字典类型，但收到了{type(batch_dict)}") 