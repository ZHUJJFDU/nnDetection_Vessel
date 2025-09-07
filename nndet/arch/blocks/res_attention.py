"""
Copyright 2023 nnDetection

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
Enhanced residual blocks with various attention mechanisms for 3D medical imaging
"""

import torch
import torch.nn as nn

from typing import Sequence, Callable, Optional, Type, Union, Dict, Any
from functools import reduce 
from loguru import logger

from nndet.arch.conv import nd_pool, NdParam
from nndet.arch.blocks.attention import (
    ChannelAttention3D, SpatialAttention3D, CBAM3D, SELayer3D, NonLocalBlock3D
)


class ResBasicAttention(nn.Module):
    """
    Enhanced residual block with attention integration
    """
    def __init__(self,
                 conv: Callable,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: NdParam,
                 stride: NdParam,
                 padding: NdParam,
                 attention_type: str = "cbam",
                 attention_params: Optional[Dict[str, Any]] = None,
                 attention_position: str = "after_residual",
                 ):
        """
        Build a residual block with integrated attention mechanism
        
        Args:
            conv: generator for convolutions
            in_channels: number of input channels
            out_channels: number of output channels
            kernel_size: kernel size of convolutions
            stride: stride of first convolution
            padding: padding of convolutions
            attention_type: type of attention mechanism to use
                choices: "none", "channel", "spatial", "cbam", "se", "nonlocal"
            attention_params: parameters to pass to the attention module
            attention_position: where to apply the attention
                choices: "before_residual", "after_residual", "parallel"
        """
        super().__init__()
        logger.info(f"Creating ResBasicAttention block with {attention_type} attention")
        
        # Main path
        self.conv1 = conv(in_channels, out_channels, kernel_size=kernel_size,
                          padding=padding, stride=stride)
        self.conv2 = conv(out_channels, out_channels, kernel_size=kernel_size,
                          padding=padding, relu=None)
        self.relu = nn.ReLU(inplace=True)
        
        # Shortcut path
        stride_prod = (reduce((lambda x, y: x * y), stride)
                      if isinstance(stride, Sequence) else stride)
        if stride_prod > 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nd_pool("Avg", dim=conv.dim, kernel_size=stride, stride=stride),
                conv(in_channels, out_channels, kernel_size=1, relu=None),
            )
        else:
            self.shortcut = None
        
        # Attention module
        self.attention_position = attention_position
        self.attention = self._create_attention_module(
            attention_type, out_channels, attention_params
        )
        
        # Initialize weights
        self.init_weights()

    def _create_attention_module(self, 
                                attention_type: str, 
                                channels: int, 
                                params: Optional[Dict[str, Any]] = None
                                ) -> Optional[nn.Module]:
        """
        Create attention module based on specified type
        
        Args:
            attention_type: type of attention to create
            channels: number of channels
            params: additional parameters for attention module
            
        Returns:
            nn.Module or None: created attention module
        """
        if params is None:
            params = {}
            
        if attention_type.lower() == "none":
            return None
        elif attention_type.lower() == "channel":
            return ChannelAttention3D(channels, **params)
        elif attention_type.lower() == "spatial":
            return SpatialAttention3D(**params)
        elif attention_type.lower() == "cbam":
            return CBAM3D(channels, **params)
        elif attention_type.lower() == "se":
            return SELayer3D(channels, **params)
        elif attention_type.lower() == "nonlocal":
            return NonLocalBlock3D(channels, **params)
        else:
            logger.warning(f"Unknown attention type: {attention_type}, using none")
            return None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward input through the block
        
        Args:
            x (torch.Tensor): input tensor
            
        Returns:
            torch.Tensor: output tensor
        """
        residual = x
        
        # Main path
        out = self.conv1(x)
        out = self.conv2(out)
        
        # Apply attention before residual
        if self.attention_position == "before_residual" and self.attention is not None:
            out = self.attention(out)
        
        # Shortcut path
        if self.shortcut is not None:
            residual = self.shortcut(x)
        
        # Add residual connection
        out = out + residual
        
        # Apply attention after residual
        if self.attention_position == "after_residual" and self.attention is not None:
            out = self.attention(out)
        
        # Apply ReLU
        out = self.relu(out)
        
        return out

    def init_weights(self) -> None:
        """Initialize weights of the block"""
        try:
            torch.nn.init.zeros_(self.conv2.norm.weight)
        except:
            logger.info(f"Zero init of last norm layer {self.conv2.norm} failed")


class ResBottleneckAttention(nn.Module):
    """
    Enhanced bottleneck residual block with attention integration
    """
    def __init__(self,
                 conv: Callable,
                 in_channels: int,
                 internal_channels: int,
                 kernel_size: NdParam,
                 stride: NdParam,
                 padding: NdParam,
                 expansion: int = 4,
                 attention_type: str = "cbam",
                 attention_params: Optional[Dict[str, Any]] = None,
                 attention_position: str = "after_residual",
                 ):
        """
        Build a bottleneck residual block with integrated attention mechanism
        
        Args:
            conv: generator for convolutions
            in_channels: number of input channels
            internal_channels: number of internal channels to use
            kernel_size: kernel size of convolutions
            stride: stride of middle convolution
            padding: padding of convolutions
            expansion: expansion for last conv block
            attention_type: type of attention mechanism to use
                choices: "none", "channel", "spatial", "cbam", "se", "nonlocal"
            attention_params: parameters to pass to the attention module
            attention_position: where to apply the attention
                choices: "before_residual", "after_residual", "parallel"
        """
        super().__init__()
        logger.info(f"Creating ResBottleneckAttention block with {attention_type} attention")
        
        out_channels = internal_channels * expansion
        
        # Main path - bottleneck architecture
        self.conv1 = conv(in_channels, internal_channels,
                         kernel_size=1, padding=0, stride=1)
        self.conv2 = conv(internal_channels, internal_channels,
                         kernel_size=kernel_size, padding=padding, stride=stride)
        self.conv3 = conv(internal_channels, out_channels,
                         kernel_size=1, padding=0, relu=None, stride=1)
        self.relu = nn.ReLU(inplace=True)
        
        # Shortcut path
        stride_prod = (reduce((lambda x, y: x * y), stride)
                      if isinstance(stride, Sequence) else stride)
        if stride_prod > 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nd_pool("Avg", dim=conv.dim, kernel_size=stride, stride=stride),
                conv(in_channels, out_channels, kernel_size=1, relu=None),
            )
        else:
            self.shortcut = None
        
        # Attention module
        self.attention_position = attention_position
        self.attention = self._create_attention_module(
            attention_type, out_channels, attention_params
        )
        
        # Initialize weights
        self.init_weights()

    def _create_attention_module(self, 
                                attention_type: str, 
                                channels: int, 
                                params: Optional[Dict[str, Any]] = None
                                ) -> Optional[nn.Module]:
        """
        Create attention module based on specified type
        
        Args:
            attention_type: type of attention to create
            channels: number of channels
            params: additional parameters for attention module
            
        Returns:
            nn.Module or None: created attention module
        """
        if params is None:
            params = {}
            
        if attention_type.lower() == "none":
            return None
        elif attention_type.lower() == "channel":
            return ChannelAttention3D(channels, **params)
        elif attention_type.lower() == "spatial":
            return SpatialAttention3D(**params)
        elif attention_type.lower() == "cbam":
            return CBAM3D(channels, **params)
        elif attention_type.lower() == "se":
            return SELayer3D(channels, **params)
        elif attention_type.lower() == "nonlocal":
            return NonLocalBlock3D(channels, **params)
        else:
            logger.warning(f"Unknown attention type: {attention_type}, using none")
            return None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward input through the block
        
        Args:
            x (torch.Tensor): input tensor
            
        Returns:
            torch.Tensor: output tensor
        """
        residual = x
        
        # Main path
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        
        # Apply attention before residual
        if self.attention_position == "before_residual" and self.attention is not None:
            out = self.attention(out)
        
        # Shortcut path
        if self.shortcut is not None:
            residual = self.shortcut(x)
        
        # Add residual connection
        out = out + residual
        
        # Apply attention after residual
        if self.attention_position == "after_residual" and self.attention is not None:
            out = self.attention(out)
        
        # Apply ReLU
        out = self.relu(out)
        
        return out

    def init_weights(self) -> None:
        """Initialize weights of the block"""
        try:
            torch.nn.init.zeros_(self.conv3.norm.weight)
        except:
            logger.info(f"Zero init of last norm layer {self.conv3.norm} failed") 