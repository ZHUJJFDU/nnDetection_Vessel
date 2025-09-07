import torch
import torch.nn as nn

from typing import Union, Callable, Sequence, List, Optional, Dict, Any
from loguru import logger
import math

from nndet.arch.decoder.base import UFPNModular


class AttentionUFPNModular(UFPNModular):
    """
    UFPNModular with attention mechanisms specifically for skip connections
    """
    def __init__(self,
                 conv: Callable,
                 strides: Sequence[int],
                 in_channels: Sequence[int],
                 conv_kernels: Union[Sequence[Union[Sequence[int], int]], int],
                 decoder_levels: Union[Sequence[int], None],
                 fixed_out_channels: int,
                 min_out_channels: int = 8,
                 upsampling_mode: str = 'nearest',
                 num_lateral: int = 1,
                 norm_lateral: bool = False,
                 activation_lateral: bool = False,
                 num_out: int = 1,
                 norm_out: bool = False,
                 activation_out: bool = False,
                 num_fusion: int = 0,
                 norm_fusion: bool = False,
                 activation_fusion: bool = False,
                 attention_type: str = "cbam",
                 attention_params: Optional[Dict[str, Any]] = None,
                 ):
        """
        UFPNModular with attention mechanisms specifically for skip connections

        Args:
            conv: convolution module to use internally
            strides: define stride with respective to largest feature map
                (from lowest stride [highest res] to highest stride [lowest res])
            in_channels: number of channels of each feature maps
            conv_kernels: define convolution kernels for decoder levels
            decoder_levels: levels which are later used for detection.
                If None a normal fpn is used.
            fixed_out_channels: number of output channels in fixed layers
            min_out_channels: minimum number of feature channels for
                layers above decoder levels
            upsampling_mode: if `transpose` a transposed convolution is used
                for upsampling, otherwise it defines the method used in
                torch.interpolate followed by a 1x1 convolution to adjust
                the channels
            num_lateral: number of lateral convolutions
            norm_lateral: en-/disable normalization in lateral connections
            activation_lateral: en-/disable non linearity in lateral connections
            num_out: number of output convolutions
            norm_out: en-/disable normalization in output connections
            activation_out: en-/disable non linearity in out connections
            num_fusion: number of convolutions after elementwise addition of skip connections
            norm_fusion: en-/disable normalization in fusion convolutions
            activation_fusion: en-/disable non linearity in fusion convolutions
            attention_type: type of attention to use
                choices: "none", "channel", "spatial", "cbam", "se", "nonlocal", "cbam_eca", "vessel"
            attention_params: parameters for attention modules
        """
        super().__init__(conv=conv, strides=strides, in_channels=in_channels,
                         conv_kernels=conv_kernels, decoder_levels=decoder_levels,
                         fixed_out_channels=fixed_out_channels,
                         min_out_channels=min_out_channels,
                         upsampling_mode=upsampling_mode,
                         num_lateral=num_lateral,
                         norm_lateral=norm_lateral,
                         activation_lateral=activation_lateral,
                         num_out=num_out,
                         norm_out=norm_out,
                         activation_out=activation_out,
                         num_fusion=num_fusion,
                         norm_fusion=norm_fusion,
                         activation_fusion=activation_fusion,
                         )
        
        self.attention_type = attention_type
        self.attention_params = attention_params if attention_params is not None else {}
        
        # 为每个级别创建注意力模块（应用于skip connection）
        self.skip_attention = nn.ModuleDict()
        for level in range(1, self.num_level):  # 第一层没有skip connection
            self.skip_attention[f"P{level}"] = self._create_attention_module(
                attention_type, self.out_channels[level])
    
    def _create_attention_module(self, 
                                attention_type: str, 
                             channels: int) -> Optional[nn.Module]:
        """
        创建注意力模块
        
        Args:
            attention_type: 注意力类型
            channels: 输入通道数
            
        Returns:
            nn.Module: 注意力模块
        """
        reduction_ratio = self.attention_params.get("reduction_ratio", 16)
        kernel_size = self.attention_params.get("kernel_size", 7)
        use_scale = self.attention_params.get("use_scale", True)
        use_vessel = self.attention_params.get("use_vessel", True)
        fusion_mode = self.attention_params.get("fusion_mode", "concatenation")
            
        if attention_type == "none" or attention_type is None:
            return None
        elif attention_type == "channel":
            return self._create_channel_attention(channels, reduction_ratio)
        elif attention_type == "spatial":
            return self._create_spatial_attention(kernel_size)
        elif attention_type == "cbam":
            return self._create_cbam_attention(channels, reduction_ratio, kernel_size)
        elif attention_type == "cbam_eca":
            return self._create_cbam_eca_attention(channels, kernel_size)
        elif attention_type == "vessel":
            return self._create_vessel_attention(channels, kernel_size, use_vessel, fusion_mode)
        else:
            logger.warning(f"Unknown attention type: {attention_type}, using none")
            return None
    
    def _create_channel_attention(self, channels: int, reduction_ratio: int) -> nn.Module:
        """创建通道注意力模块"""
        return ChannelAttention(channels, reduction_ratio, self.dim)
    
    def _create_spatial_attention(self, kernel_size: int) -> nn.Module:
        """创建空间注意力模块"""
        return SpatialAttention(kernel_size, self.dim)
    
    def _create_cbam_attention(self, channels: int, reduction_ratio: int, kernel_size: int) -> nn.Module:
        """创建CBAM注意力模块"""
        return CBAM(channels, reduction_ratio, kernel_size, self.dim)
    
    def _create_cbam_eca_attention(self, channels: int, kernel_size: int) -> nn.Module:
        """创建带有ECA通道注意力的CBAM模块"""
        return CBAMwithECA(channels, kernel_size, self.dim)
    
    def _create_vessel_attention(self, channels: int, kernel_size: int, 
                                 use_vessel: bool = True, 
                                 fusion_mode: str = "concatenation") -> nn.Module:
        """创建血管引导注意力模块"""
        return VesselGuidedAttention(channels, kernel_size, use_vessel, fusion_mode, self.dim)
        
    def forward(self, inp_seq: Sequence[torch.Tensor], vessel_mask: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
        """
        Forward pass with attention-enhanced skip connections and optional vessel guidance
        
        Args:
            inp_seq: sequence with feature maps (largest to samllest)
            vessel_mask: optional vessel segmentation mask for guided attention

        Returns:
            List[Tensor]: resulting feature maps
        """
        fpn_maps = self.forward_lateral(inp_seq)

        # bottom up path way
        out_list = []  # sorted lowest to highest res
        for idx, x in enumerate(reversed(fpn_maps), 1):
            level = self.num_level - idx

            if idx != 1:
                # 添加注意力增强的skip connection
                if self.attention_type != "none" and self.attention_type is not None and f"P{level}" in self.skip_attention:
                    # 应用注意力到上采样特征
                    if self.attention_type == "vessel" and vessel_mask is not None:
                        attended_up = self.skip_attention[f"P{level}"](up, vessel_mask)
                    else:
                        attended_up = self.skip_attention[f"P{level}"](up)
                    x = x + attended_up
                else:
                    # 常规skip connection
                    x = x + up
                
                if self.num_fusion > 0:
                    x = self.fusion_bottom_up[f"P{level}"](x)

            if idx != self.num_level:
                up = self.up[f"P{level}"](x)

            out_list.append(x)
            
        return self.forward_out(reversed(out_list))


# 简单的注意力模块实现
class ChannelAttention(nn.Module):
    """通道注意力模块"""
    def __init__(self, channels: int, reduction_ratio: int = 16, dim: int = 3):
        super().__init__()
        self.avg_pool = self._get_adaptive_avg_pool(dim)
        self.max_pool = self._get_adaptive_max_pool(dim)
        
        reduced_channels = max(1, channels // reduction_ratio)
        self.fc = nn.Sequential(
            self._get_conv(dim, channels, reduced_channels, 1, 0),
            nn.ReLU(inplace=True),
            self._get_conv(dim, reduced_channels, channels, 1, 0)
        )
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
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)


class SpatialAttention(nn.Module):
    """空间注意力模块"""
    def __init__(self, kernel_size: int = 7, dim: int = 3):
        super().__init__()
        assert kernel_size in (3, 5, 7), "Kernel size must be 3, 5, or 7"
        padding = kernel_size // 2
        self.conv = self._get_conv(dim, 2, 1, kernel_size, padding)
        self.sigmoid = nn.Sigmoid()
        self.dim = dim
    
    def _get_conv(self, dim, in_ch, out_ch, kernel_size, padding):
        if dim == 3:
            return nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=True)
        else:
            return nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=True)
    
    def forward(self, x):
        # 计算平均值和最大值沿通道维度
        if self.dim == 3:
            avg_out = torch.mean(x, dim=1, keepdim=True)
            max_out, _ = torch.max(x, dim=1, keepdim=True)
        else:
            avg_out = torch.mean(x, dim=1, keepdim=True)
            max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # 连接平均值和最大值
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        
        return x * self.sigmoid(out)


class CBAM(nn.Module):
    """卷积块注意力模块 (CBAM)"""
    def __init__(self, channels: int, reduction_ratio: int = 16, kernel_size: int = 7, dim: int = 3):
        super().__init__()
        self.channel_att = ChannelAttention(channels, reduction_ratio, dim)
        self.spatial_att = SpatialAttention(kernel_size, dim)
    
    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


class ECAChannelAttention(nn.Module):
    """
    改进的ECA通道注意力模块，支持2D和3D数据
    
    使用归一化和高斯激活增强通道注意力机制
    """
    def __init__(self, channels: int, dim: int = 3, **kwargs):
        super().__init__()
        # 自适应计算k值（确保k为奇数）
        k = int(abs(math.log2(channels) / 2 + 0.5))
        k = k if k % 2 else k + 1  # 确保k是奇数
        
        # 选择适合维度的平均池化层
        self.avg_pool = self._get_adaptive_avg_pool(dim)
        
        # 1D卷积保持不变，因为它是在通道维度上操作
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k-1)//2, bias=False)
        self.eps = 1e-5  # 防止除零错误
        self.dim = dim

    def _get_adaptive_avg_pool(self, dim):
        """根据数据维度选择适当的全局平均池化层"""
        if dim == 3:
            return nn.AdaptiveAvgPool3d(1)
        else:
            return nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # 处理输入x，根据维度决定如何操作
        b, c = x.size()[:2]
        
        # 全局平均池化
        y = self.avg_pool(x)        # [N, C, 1, 1] 或 [N, C, 1, 1, 1]
        
        # 改变形状为1D卷积输入
        if self.dim == 3:
            y = y.squeeze(-1).squeeze(-1).squeeze(-1).unsqueeze(1)  # [N, 1, C]
        else:
            y = y.squeeze(-1).squeeze(-1).unsqueeze(1)  # [N, 1, C]
            
        # 1D卷积
        y = self.conv(y)            # [N, 1, C]
        
        # 计算通道维度上的均值和标准差
        mean = y.mean(dim=2, keepdim=True)
        std = y.std(dim=2, keepdim=True) + self.eps
        
        # 标准化
        y_normalized = (y - mean) / std
        
        # 使用高斯函数作为激活函数
        y = torch.exp(-0.5 * (y_normalized ** 2))
        
        # 将形状恢复到原始大小
        if self.dim == 3:
            y = y.squeeze(1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # [N, C, 1, 1, 1]
        else:
            y = y.squeeze(1).unsqueeze(-1).unsqueeze(-1)  # [N, C, 1, 1]
        
        # 应用通道注意力
        return x * y


class CBAMwithECA(nn.Module):
    """
    使用ECA通道注意力的CBAM模块
    
    结合了ECA通道注意力和标准空间注意力
    """
    def __init__(self, channels: int, kernel_size: int = 7, dim: int = 3):
        super().__init__()
        # 使用ECA通道注意力代替原始通道注意力
        self.channel_att = ECAChannelAttention(channels, dim)
        
        # 使用原有的空间注意力
        self.spatial_att = SpatialAttention(kernel_size, dim)
    
    def forward(self, x):
        # 与原始CBAM相同的流程：先通道注意力，再空间注意力
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x 


# 新增：血管引导注意力模块
class VesselGuidedAttention(nn.Module):
    """
    血管引导注意力模块
    
    结合通道注意力和血管引导的空间注意力，使用血管分割掩码作为额外输入
    增强对血管区域的感知
    """
    def __init__(self, channels: int, kernel_size: int = 7, 
                 use_vessel: bool = True, fusion_mode: str = "concatenation", 
                 dim: int = 3):
        super().__init__()
        self.dim = dim
        self.use_vessel = use_vessel
        self.fusion_mode = fusion_mode  # concatenation, addition, multiplication
        
        # 使用标准通道注意力
        self.channel_att = ChannelAttention(channels, 16, dim)
        
        # 血管引导空间注意力需要处理常规特征和血管掩码
        if fusion_mode == "concatenation":
            # 如果使用连接模式，空间注意力的输入通道数需要增加1（血管掩码）
            self.spatial_conv = self._get_conv(dim, 3, 1, kernel_size, kernel_size//2)
        else:
            # 如果使用加法或乘法模式，保持常规空间注意力的输入通道数
            self.spatial_conv = self._get_conv(dim, 2, 1, kernel_size, kernel_size//2)
            
        self.sigmoid = nn.Sigmoid()
        
        # 用于处理血管掩码的卷积（如有必要）
        if self.use_vessel and self.fusion_mode != "concatenation":
            self.vessel_conv = self._get_conv(dim, 1, 1, kernel_size, kernel_size//2)

        # 用于处理回退情况的标准空间注意力卷积
        self.fallback_spatial_conv = self._get_conv(dim, 2, 1, kernel_size, kernel_size//2)
    
    def _get_conv(self, dim, in_ch, out_ch, kernel_size, padding):
        if dim == 3:
            return nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=True)
        else:
            return nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=True)
    
    def forward(self, x, vessel_mask=None):
        # 应用通道注意力
        x_chan = self.channel_att(x)
        
        # 计算平均值和最大值沿通道维度
        avg_out = torch.mean(x_chan, dim=1, keepdim=True)
        max_out, _ = torch.max(x_chan, dim=1, keepdim=True)
        
        # 如果没有提供血管掩码或不使用血管掩码，则降级为常规CBAM
        if vessel_mask is None or not self.use_vessel:
            # 连接平均值和最大值
            spatial_in = torch.cat([avg_out, max_out], dim=1)
            
            # 使用回退空间注意力卷积
            spatial_att = self.sigmoid(self.fallback_spatial_conv(spatial_in))
            
            return x_chan * spatial_att
        
        # 检查和记录血管掩码
        if vessel_mask.dim() != x.dim():
            logger.warning(f"血管掩码维度 ({vessel_mask.dim()}) 与特征图 ({x.dim()}) 不匹配，尝试修正...")
            # 尝试添加缺失的维度
            if vessel_mask.dim() == x.dim() - 1 and vessel_mask.shape[0] == x.shape[0]:
                vessel_mask = vessel_mask.unsqueeze(1)
                logger.info(f"添加通道维度后血管掩码形状: {vessel_mask.shape}")
        
        if vessel_mask.shape[1] != 1:
            logger.warning(f"血管掩码通道数 ({vessel_mask.shape[1]}) 应为1，尝试修正...")
            if vessel_mask.shape[1] > 1:
                # 如果有多个通道，取第一个
                vessel_mask = vessel_mask[:, 0:1]
                logger.info(f"提取单通道后血管掩码形状: {vessel_mask.shape}")
        
        # 确保血管掩码与特征图形状匹配
        if vessel_mask.shape[2:] != x.shape[2:]:
            logger.info(f"调整血管掩码大小，从 {vessel_mask.shape[2:]} 到 {x.shape[2:]}")
            
            # 使用最近邻插值调整血管掩码大小
            if self.dim == 3:
                vessel_mask = torch.nn.functional.interpolate(
                    vessel_mask, size=x.shape[2:], mode='nearest')
            else:
                vessel_mask = torch.nn.functional.interpolate(
                    vessel_mask, size=x.shape[2:], mode='nearest')
            
            logger.info(f"调整后血管掩码形状: {vessel_mask.shape}")
        
        # 根据融合模式合并血管掩码和特征统计
        try:
            if self.fusion_mode == "concatenation":
                # 简单连接血管掩码、平均值和最大值
                spatial_in = torch.cat([avg_out, max_out, vessel_mask], dim=1)
                logger.debug(f"连接后空间输入形状: {spatial_in.shape}")
                spatial_att = self.sigmoid(self.spatial_conv(spatial_in))
            
            elif self.fusion_mode == "addition":
                # 处理血管掩码并与特征统计相加
                vessel_att = self.sigmoid(self.vessel_conv(vessel_mask))
                spatial_in = torch.cat([avg_out, max_out], dim=1)
                feature_att = self.sigmoid(self.spatial_conv(spatial_in))
                spatial_att = feature_att + vessel_att
                
            elif self.fusion_mode == "multiplication":
                # 处理血管掩码并与特征统计相乘
                vessel_att = self.sigmoid(self.vessel_conv(vessel_mask))
                spatial_in = torch.cat([avg_out, max_out], dim=1)
                feature_att = self.sigmoid(self.spatial_conv(spatial_in))
                spatial_att = feature_att * vessel_att
            
            else:
                # 默认使用连接模式
                spatial_in = torch.cat([avg_out, max_out, vessel_mask], dim=1)
                spatial_att = self.sigmoid(self.spatial_conv(spatial_in))
        except RuntimeError as e:
            logger.error(f"空间注意力处理错误: {e}")
            logger.error(f"形状信息 - avg_out: {avg_out.shape}, max_out: {max_out.shape}, vessel_mask: {vessel_mask.shape}")
            
            # 使用回退机制
            logger.info("使用回退处理...")
            spatial_in = torch.cat([avg_out, max_out], dim=1)
            spatial_att = self.sigmoid(self.fallback_spatial_conv(spatial_in))
        
        # 应用空间注意力
        return x_chan * spatial_att


