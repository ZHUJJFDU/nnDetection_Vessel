import torch
import torch.nn as nn

from typing import Optional, Tuple, Callable, TypeVar
from abc import abstractmethod

from loguru import logger

from nndet.core.boxes import box_iou
from nndet.arch.layers.scale import Scale
from torch import Tensor

from nndet.losses import SmoothL1Loss, GIoULoss


CONV_TYPES = (nn.Conv2d, nn.Conv3d)


class Regressor(nn.Module):
    @abstractmethod
    def compute_loss(self, pred_deltas: Tensor, target_deltas: Tensor, **kwargs) -> Tensor:
        """
        计算回归损失（L1损失）

        参数:
            pred_deltas (Tensor): 预测的边界框偏移量 [N, dim * 2]
            target_deltas (Tensor): 目标边界框偏移量 [N, dim * 2]

        返回:
            Tensor: 损失值
        """
        raise NotImplementedError


class BaseRegressor(Regressor):
    def __init__(self,
                 conv,
                 in_channels: int,
                 internal_channels: int,
                 anchors_per_pos: int,
                 num_levels: int,
                 num_convs: int = 3,
                 add_norm: bool = True,
                 learn_scale: bool = False,
                 **kwargs,
                 ):
        """
        构建具有典型卷积结构的回归器头部的基类
        conv(in, internal) -> num_convs x conv(internal, internal) ->
        conv(internal, out)

        参数:
            conv: 处理单层的卷积模块
            in_channels: 输入通道数
            internal_channels: 内部使用的通道数
            anchors_per_pos: 每个位置的锚点数
            num_levels: 通过回归器的解码器级别数
            num_convs: 卷积数量
                input_conv -> num_convs -> output_convs
            add_norm: 启用/禁用内部层中的归一化层
            learn_scale: 是否为每个特征金字塔级别学习额外的标量值
            kwargs: 传递给第一个和内部卷积的关键字参数
        """
        super().__init__()
        self.dim = conv.dim
        self.num_levels = num_levels
        self.num_convs = num_convs
        self.learn_scale = learn_scale

        self.anchors_per_pos = anchors_per_pos

        self.in_channels = in_channels
        self.internal_channels = internal_channels

        self.conv_internal = self.build_conv_internal(conv, add_norm=add_norm, **kwargs)
        self.conv_out = self.build_conv_out(conv)

        if self.learn_scale:
            self.scales = self.build_scales()

        self.loss: Optional[nn.Module] = None
        self.init_weights()

    def build_conv_internal(self, conv, **kwargs):
        """
        构建内部卷积
        """
        _conv_internal = nn.Sequential()
        _conv_internal.add_module(
            name="c_in",
            module=conv(
                self.in_channels,
                self.internal_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                **kwargs,
            ))
        for i in range(self.num_convs):
            _conv_internal.add_module(
                name=f"c_internal{i}",
                module=conv(
                    self.internal_channels,
                    self.internal_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    **kwargs,
                ))
        return _conv_internal

    def build_conv_out(self, conv):
        """
        构建最终卷积
        """
        out_channels = self.anchors_per_pos * self.dim * 2
        return conv(
            self.internal_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            add_norm=False,
            add_act=False,
            bias=True,
        )

    def build_scales(self) -> nn.ModuleList:
        """
        为每个级别构建额外的标量值
        """
        logger.info("在回归器中学习级别特定的标量")
        return nn.ModuleList([Scale() for _ in range(self.num_levels)])

    def forward(self, x: torch.Tensor, level: int, **kwargs) -> torch.Tensor:
        """
        前向传播输入

        参数:
            x: 输入特征图，大小为 [N x C x Y x X x Z]
            level: 特征金字塔级别

        返回:
            torch.Tensor: 每个锚点的边界框回归值
                [N, n_anchors, dim*2]
        """
        bb_logits = self.conv_out(self.conv_internal(x))

        if self.learn_scale:
            bb_logits = self.scales[level](bb_logits)

        axes = (0, 2, 3, 1) if self.dim == 2 else (0, 2, 3, 4, 1)
        bb_logits = bb_logits.permute(*axes)
        bb_logits = bb_logits.contiguous()
        bb_logits = bb_logits.view(x.size()[0], -1, self.dim * 2)
        return bb_logits

    def compute_loss(self,
                     pred_deltas: Tensor,
                     target_deltas: Tensor,
                     **kwargs,
                     ) -> Tensor:
        """
        计算回归损失（L1损失）

        参数:
            pred_deltas: 预测的边界框偏移量 [N, dim * 2]
            target_deltas: 目标边界框偏移量 [N, dim * 2]

        返回:
            Tensor: 损失值
        """
        return self.loss(pred_deltas, target_deltas, **kwargs)

    def init_weights(self) -> None:
        """
        使用正态分布初始化权重（均值=0，标准差=0.01）
        """
        logger.info("重写回归器卷积权重初始化")
        for layer in self.modules():
            if isinstance(layer, CONV_TYPES):
                torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                if layer.bias is not None:
                    torch.nn.init.constant_(layer.bias, 0)


class L1Regressor(BaseRegressor):
    def __init__(self,
                 conv,
                 in_channels: int,
                 internal_channels: int,
                 anchors_per_pos: int,
                 num_levels: int,
                 num_convs: int = 3,
                 add_norm: bool = True,
                 beta: float = 1.,
                 reduction: Optional[str] = "sum",
                 loss_weight: float = 1.,
                 learn_scale: bool = False,
                 **kwargs,
                 ):
        """
        使用典型卷积结构和平滑L1损失构建回归器头部
        conv(in, internal) -> num_convs x conv(internal, internal) ->
        conv(internal, out)

        参数:
            conv: 处理单层的卷积模块
            in_channels: 输入通道数
            internal_channels: 内部使用的通道数
            anchors_per_pos: 每个位置的锚点数
            num_levels: 通过回归器的解码器级别数
            num_convs: 卷积数量
                input_conv -> num_convs -> output_convs
            add_norm: 启用/禁用内部层中的归一化层
            beta: L1到L2的变化点。
                对于beta值 < 1e-5，计算L1损失。
            reduction: 应用于损失的归约。'sum' | 'mean' | 'none'
            loss_weight: 平衡多个损失的标量
            learn_scale: 是否为每个特征金字塔级别学习额外的标量值
            kwargs: 传递给第一个和内部卷积的关键字参数
        """
        super().__init__(
            conv=conv,
            in_channels=in_channels,
            internal_channels=internal_channels,
            anchors_per_pos=anchors_per_pos,
            num_levels=num_levels,
            num_convs=num_convs,
            add_norm=add_norm,
            learn_scale=learn_scale,
            **kwargs,
            )

        self.loss = SmoothL1Loss(
            beta=beta,
            reduction=reduction,
            loss_weight=loss_weight,
            )


class GIoURegressor(BaseRegressor):
    def __init__(self,
                 conv,
                 in_channels: int,
                 internal_channels: int,
                 anchors_per_pos: int,
                 num_levels: int,
                 num_convs: int = 3,
                 add_norm: bool = True,
                 reduction: Optional[str] = "sum",
                 loss_weight: float = 1.,
                 learn_scale: bool = False,
                 **kwargs,
                 ):
        """
        使用典型卷积结构和GIoU损失构建回归器头部
        conv(in, internal) -> num_convs x conv(internal, internal) ->
        conv(internal, out)

        参数:
            conv: 处理单层的卷积模块
            in_channels: 输入通道数
            internal_channels: 内部使用的通道数
            anchors_per_pos: 每个位置的锚点数
            num_levels: 通过回归器的解码器级别数
            num_convs: 卷积数量
                input_conv -> num_convs -> output_convs
            add_norm: 启用/禁用内部层中的归一化层
            reduction: 应用于损失的归约。'sum' | 'mean' | 'none'
            loss_weight: 平衡多个损失的标量
            learn_scale: 是否为每个特征金字塔级别学习额外的标量值
            kwargs: 传递给第一个和内部卷积的关键字参数
        """
        super().__init__(
            conv=conv,
            in_channels=in_channels,
            internal_channels=internal_channels,
            anchors_per_pos=anchors_per_pos,
            num_levels=num_levels,
            num_convs=num_convs,
            add_norm=add_norm,
            learn_scale=learn_scale,
            **kwargs,
            )

        self.loss = GIoULoss(
            reduction=reduction,
            loss_weight=loss_weight,
            )


RegressorType = TypeVar('RegressorType', bound=Regressor)
