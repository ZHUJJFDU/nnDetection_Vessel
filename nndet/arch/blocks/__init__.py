"""
Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany

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

from nndet.arch.blocks.basic import AbstractBlock, StackedConvBlock, \
    StackedResidualBlock, StackedConvBlock2
from nndet.arch.blocks.res import ResBasic, ResBottleneck
from nndet.arch.blocks.res_attention import ResBasicAttention, ResBottleneckAttention
from nndet.arch.blocks.attention import (
    ChannelAttention3D, SpatialAttention3D, CBAM3D, SELayer3D, NonLocalBlock3D
)
