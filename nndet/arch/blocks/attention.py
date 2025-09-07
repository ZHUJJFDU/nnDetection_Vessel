import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class ChannelAttention3D(nn.Module):
    """
    3D Channel Attention module for medical images.
    
    Squeezes spatial information to capture channel-wise relationships.
    """
    def __init__(self, channels: int, reduction_ratio: int = 16):
        """
        Initialize 3D channel attention
        
        Args:
            channels (int): Number of input channels
            reduction_ratio (int): Channel reduction ratio for bottleneck
        """
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        
        reduced_channels = max(1, channels // reduction_ratio)
        self.fc = nn.Sequential(
            nn.Conv3d(channels, reduced_channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(reduced_channels, channels, kernel_size=1, bias=True)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for channel attention
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, C, D, H, W]
            
        Returns:
            torch.Tensor: Tensor with channel attention applied
        """
        # Average pooling branch
        avg_pool = self.avg_pool(x)
        avg_out = self.fc(avg_pool)
        
        # Max pooling branch
        max_pool = self.max_pool(x)
        max_out = self.fc(max_pool)
        
        # Combine branches
        out = avg_out + max_out
        attention = self.sigmoid(out)
        
        return x * attention


class SpatialAttention3D(nn.Module):
    """
    3D Spatial Attention module for medical images.
    
    Focuses on important spatial regions in the volume.
    """
    def __init__(self, kernel_size: int = 7):
        """
        Initialize 3D spatial attention
        
        Args:
            kernel_size (int): Size of the convolution kernel
        """
        super().__init__()
        assert kernel_size in (3, 5, 7), "Kernel size must be 3, 5, or 7"
        padding = kernel_size // 2
        
        self.conv = nn.Conv3d(2, 1, kernel_size=kernel_size, padding=padding, bias=True)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for spatial attention
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, C, D, H, W]
            
        Returns:
            torch.Tensor: Tensor with spatial attention applied
        """
        # Generate spatial attention map
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention_map = torch.cat([avg_out, max_out], dim=1)
        
        attention_map = self.conv(attention_map)
        attention_map = self.sigmoid(attention_map)
        
        return x * attention_map


class CBAM3D(nn.Module):
    """
    3D Convolutional Block Attention Module (CBAM).
    
    Combines both channel and spatial attention.
    """
    def __init__(self, channels: int, reduction_ratio: int = 16, kernel_size: int = 7):
        """
        Initialize 3D CBAM
        
        Args:
            channels (int): Number of input channels
            reduction_ratio (int): Channel reduction ratio
            kernel_size (int): Kernel size for spatial attention
        """
        super().__init__()
        self.channel_attention = ChannelAttention3D(channels, reduction_ratio)
        self.spatial_attention = SpatialAttention3D(kernel_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for CBAM
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, C, D, H, W]
            
        Returns:
            torch.Tensor: Tensor with attention applied
        """
        # Apply channel attention first
        x = self.channel_attention(x)
        
        # Then apply spatial attention
        x = self.spatial_attention(x)
        
        return x


class SELayer3D(nn.Module):
    """
    3D Squeeze-and-Excitation block.
    
    Focuses on channel relationships in 3D volumes.
    """
    def __init__(self, channels: int, reduction_ratio: int = 16):
        """
        Initialize 3D SE block
        
        Args:
            channels (int): Number of input channels
            reduction_ratio (int): Channel reduction ratio
        """
        super().__init__()
        reduced_channels = max(1, channels // reduction_ratio)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Conv3d(channels, reduced_channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(reduced_channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for SE block
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, C, D, H, W]
            
        Returns:
            torch.Tensor: Tensor with attention applied
        """
        b, c, d, h, w = x.size()
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y


class NonLocalBlock3D(nn.Module):
    """
    3D Non-local Neural Networks block.
    
    Captures long-range dependencies in 3D volumes.
    """
    def __init__(self, channels: int, reduction_ratio: int = 2, use_scale: bool = True):
        """
        Initialize 3D non-local block
        
        Args:
            channels (int): Number of input channels
            reduction_ratio (int): Channel reduction ratio for internal operations
            use_scale (bool): Whether to scale the output by 1/sqrt(channels)
        """
        super().__init__()
        self.use_scale = use_scale
        self.scale_factor = 1.0 / (channels ** 0.5) if use_scale else 1.0
        
        reduced_channels = channels // reduction_ratio
        
        # Query, key, value projections
        self.query_conv = nn.Conv3d(channels, reduced_channels, kernel_size=1)
        self.key_conv = nn.Conv3d(channels, reduced_channels, kernel_size=1)
        self.value_conv = nn.Conv3d(channels, reduced_channels, kernel_size=1)
        
        # Output projection
        self.out_conv = nn.Conv3d(reduced_channels, channels, kernel_size=1)
        
        # Initialize weights
        nn.init.kaiming_normal_(self.query_conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.key_conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.value_conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.out_conv.weight, 0)
        nn.init.constant_(self.out_conv.bias, 0)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for non-local block
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, C, D, H, W]
            
        Returns:
            torch.Tensor: Tensor with non-local attention applied
        """
        batch_size = x.size(0)
        
        # Project to get query, key, value
        query = self.query_conv(x).view(batch_size, -1, x.size(2) * x.size(3) * x.size(4)).permute(0, 2, 1)
        key = self.key_conv(x).view(batch_size, -1, x.size(2) * x.size(3) * x.size(4))
        value = self.value_conv(x).view(batch_size, -1, x.size(2) * x.size(3) * x.size(4)).permute(0, 2, 1)
        
        # Calculate attention map
        attention = torch.bmm(query, key)
        if self.use_scale:
            attention = attention * self.scale_factor
        attention = F.softmax(attention, dim=-1)
        
        # Apply attention to value
        out = torch.bmm(attention, value).permute(0, 2, 1).contiguous()
        out = out.view(batch_size, -1, x.size(2), x.size(3), x.size(4))
        
        # Output projection
        out = self.out_conv(out)
        
        # Residual connection
        return x + out 