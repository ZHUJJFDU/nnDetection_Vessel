import torch
import math
import torch.nn as nn

from typing import Optional, TypeVar
from torch import Tensor
from abc import abstractmethod
from loguru import logger

from nndet.losses.classification import (
    FocalLossWithLogits,
    BCEWithLogitsLossOneHot,
    CrossEntropyLoss,
)

CONV_TYPES = (nn.Conv2d, nn.Conv3d)


class Classifier(nn.Module):
    @abstractmethod
    def compute_loss(self, pred_logits: Tensor, targets: Tensor, **kwargs) -> Tensor:
        """
        计算分类损失（交叉熵损失）

        参数:
            pred_logits (Tensor): 预测的logits
            targets (Tensor): 分类目标

        返回:
            Tensor: 分类损失
        """
        raise NotImplementedError

    @abstractmethod
    def box_logits_to_probs(self, box_logits: Tensor) -> Tensor:
        """
        将边界框logits转换为概率

        参数:
            box_logits (Tensor): 边界框logits [N, C], C=类别数量

        返回:
            Tensor: 概率
        """
        raise NotImplementedError


class BaseClassifier(Classifier):
    def __init__(self,
                 conv,
                 in_channels: int,
                 internal_channels: int,
                 num_classes: int,
                 anchors_per_pos: int,
                 num_levels: int,
                 num_convs: int = 3,
                 add_norm: bool = True,
                 **kwargs
                 ):
        """
        构建具有典型卷积结构的分类器头部的基类
        conv(in, internal) -> num_convs x conv(internal, internal) ->
        conv(internal, out)

        参数:
            conv: 处理单层的卷积模块
            in_channels: 输入通道数
            internal_channels: 内部使用的通道数
            num_classes: 前景类别数
            anchors_per_pos: 每个位置的锚点数
            num_levels: 通过分类器的解码器级别数
            num_convs: 卷积数量
                input_conv -> num_convs -> output_convs
            add_norm: 启用/禁用内部层中的归一化层
            kwargs: 传递给第一个和内部卷积的关键字参数

        注意:
            `self.loss` 需要在子类中重写
            `self.logits_convert_fn` 需要在子类中重写
        """
        super().__init__()
        self.dim = conv.dim
        self.num_levels = num_levels
        self.num_convs = num_convs

        self.num_classes = num_classes
        self.anchors_per_pos = anchors_per_pos

        self.in_channels = in_channels
        self.internal_channels = internal_channels

        self.conv_internal = self.build_conv_internal(conv, add_norm=add_norm, **kwargs)
        self.conv_out = self.build_conv_out(conv)

        self.loss: Optional[nn.Module] = None
        self.logits_convert_fn: Optional[nn.Module] = None
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
        out_channels = self.num_classes * self.anchors_per_pos
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

    def forward(self,
                x: torch.Tensor,
                level: int,
                **kwargs,
                ) -> torch.Tensor:
        """
        前向传播输入

        参数:
            x (torch.Tensor): 大小为(N x C x Y x X x Z)的输入特征图

        返回:
            torch.Tensor: 每个锚点的分类logits
                (N x anchors x num_classes)
        """
        class_logits = self.conv_out(self.conv_internal(x))

        axes = (0, 2, 3, 1) if self.dim == 2 else (0, 2, 3, 4, 1)
        class_logits = class_logits.permute(*axes)
        class_logits = class_logits.contiguous()
        class_logits = class_logits.view(x.size()[0], -1, self.num_classes)
        return class_logits

    def compute_loss(self, pred_logits: Tensor, targets: Tensor, **kwargs) -> Tensor:
        """
        带有交叉熵损失的基本分类器（通常在此之前应该进行硬负例挖掘）

        参数:
            pred_logits (Tensor): 预测的logits
            targets (Tensor): 分类目标

        返回:
            Tensor: 分类损失
        """
        return self.loss(pred_logits, targets.long(), **kwargs)

    def box_logits_to_probs(self, box_logits: Tensor) -> Tensor:
        """
        将边界框logits转换为概率

        参数:
            box_logits (Tensor): 边界框logits [N, C]
                N = 锚点数量, C=前景类别数量

        返回:
            Tensor: 概率
        """
        return self.logits_convert_fn(box_logits)

    def init_weights(self) -> None:
        """
        使用先验概率初始化权重
        """
        if self.prior_prob is not None:
            logger.info(f"初始化分类器权重: 先验概率 {self.prior_prob}")
            for layer in self.modules():
                if isinstance(layer, CONV_TYPES):
                    torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                    if layer.bias is not None:
                        torch.nn.init.constant_(layer.bias, 0)

            # 在模型初始化中使用先验概率来提高稳定性
            bias_value = -math.log((1 - self.prior_prob) / self.prior_prob)
            for layer in self.conv_out.modules():
                if isinstance(layer, CONV_TYPES):
                    torch.nn.init.constant_(layer.bias, bias_value)
        else:
            logger.info("初始化分类器权重: 使用卷积默认值")
  

class BCECLassifier(BaseClassifier):
    def __init__(self,
                 conv,
                 in_channels: int,
                 internal_channels: int,
                 num_classes: int,
                 anchors_per_pos: int,
                 num_levels: int,
                 num_convs: int = 3,
                 add_norm: bool = True,
                 prior_prob: Optional[float] = None,
                 weight: Optional[Tensor] = None,
                 reduction: str = "mean",
                 smoothing: float = 0.0,
                 loss_weight: float = 1.,
                 **kwargs
                 ):
        """
        使用基于Sigmoid的BCE损失计算和先验概率权重初始化的分类器头部
        conv(in, internal) -> num_convs x conv(internal, internal) ->
        conv(internal, out)

        参数:
            conv: 处理单层的卷积模块
            in_channels: 输入通道数
            internal_channels: 内部使用的通道数
            num_classes: 前景类别数
            anchors_per_pos: 每个位置的锚点数
            num_levels: 通过分类器的解码器级别数
            num_convs: 卷积数量
                input_conv -> num_convs -> output_convs
            add_norm: 启用/禁用内部层中的归一化层
            prior_prob: 使用给定的先验概率初始化最终卷积
            weight: BCEWithLogitsLoss中的权重（更多信息参见pytorch）
            reduction: 应用于损失的归约。'sum' | 'mean' | 'none'
            smoothing: 标签平滑
            loss_weight: 平衡多个损失的标量
            kwargs: 传递给第一个和内部卷积的关键字参数
        """
        self.prior_prob = prior_prob
        super().__init__(
            conv=conv,
            in_channels=in_channels,
            num_convs=num_convs,
            add_norm=add_norm,
            internal_channels=internal_channels,
            num_classes=num_classes,
            anchors_per_pos=anchors_per_pos,
            num_levels=num_levels,
            **kwargs,
            )

        self.loss = BCEWithLogitsLossOneHot(
            num_classes=num_classes,
            weight=weight,
            reduction=reduction,
            smoothing=smoothing,
            loss_weight=loss_weight,
            )
        self.logits_convert_fn = nn.Sigmoid()


class CEClassifier(BaseClassifier):
    def __init__(self,
                conv,
                in_channels: int,
                internal_channels: int,
                num_classes: int,
                anchors_per_pos: int,
                num_levels: int,
                num_convs: int = 3,
                add_norm: bool = True,
                prior_prob: Optional[float] = None,
                weight: Optional[Tensor] = None,
                reduction: str = "mean",
                loss_weight: float = 1.,
                **kwargs
                ):
        """
        使用基于Sigmoid的BCE损失计算和先验概率权重初始化的分类器头部
        conv(in, internal) -> num_convs x conv(internal, internal) ->
        conv(internal, out)

        参数:
            conv: 处理单层的卷积模块
            in_channels: 输入通道数
            internal_channels: 内部使用的通道数
            num_classes: 前景类别数
            anchors_per_pos: 每个位置的锚点数
            num_levels: 通过分类器的解码器级别数
            num_convs: 卷积数量
                input_conv -> num_convs -> output_convs
            add_norm: 启用/禁用内部层中的归一化层
            prior_prob: 使用给定的先验概率初始化最终卷积
            weight: 交叉熵损失中的权重（更多信息参见pytorch）
            reduction: 应用于损失的归约。'sum' | 'mean' | 'none'
            loss_weight: 平衡多个损失的标量
            kwargs: 传递给第一个和内部卷积的关键字参数
        """
        self.prior_prob = prior_prob
        super().__init__(
            conv=conv,
            in_channels=in_channels,
            num_convs=num_convs,
            add_norm=add_norm,
            internal_channels=internal_channels,
            num_classes=num_classes + 1, # 为背景添加一个通道
            anchors_per_pos=anchors_per_pos,
            num_levels=num_levels,
            **kwargs,
            )

        self.loss = CrossEntropyLoss(
            weight=weight,
            reduction=reduction,
            loss_weight=loss_weight,
            )
        self.logits_convert_fn = nn.Softmax(dim=1)

    def box_logits_to_probs(self, box_logits: Tensor) -> Tensor:
        """
        将边界框logits转换为概率

        参数:
            box_logits (Tensor): 边界框logits [N, C], C=类别数量

        返回:
            Tensor: 概率
        """
        return self.logits_convert_fn(box_logits)[:, 1:] # 移除背景预测


class FocalClassifier(BaseClassifier):
    def __init__(self,
                 conv,
                 in_channels: int,
                 internal_channels: int,
                 num_classes: int,
                 anchors_per_pos: int,
                 num_levels: int,
                 num_convs: int = 3,
                 add_norm: bool = True,
                 prior_prob: Optional[float] = None,
                 gamma: float = 2,
                 alpha: float = -1,
                 reduction: str = "sum",
                 loss_weight: float = 1.,
                 **kwargs
                 ):
        """
        使用基于Sigmoid的BCE损失计算和先验概率权重初始化的分类器头部
        conv(in, internal) -> num_convs x conv(internal, internal) ->
        conv(internal, out)

        参数:
            conv: 处理单层的卷积模块
            in_channels: 输入通道数
            internal_channels: 内部使用的通道数
            num_classes: 前景类别数
            anchors_per_pos: 每个位置的锚点数
            num_levels: 通过分类器的解码器级别数
            num_convs: 卷积数量
                input_conv -> num_convs -> output_convs
            add_norm: 启用/禁用内部层中的归一化层
            prior_prob: 使用给定的先验概率初始化最终卷积
            gamma: focal loss的gamma参数
            alpha: focal loss的alpha参数
            reduction: 应用于损失的归约。'sum' | 'mean' | 'none'
            loss_weight: 平衡多个损失的标量
            kwargs: 传递给第一个和内部卷积的关键字参数
        """
        self.prior_prob = prior_prob
        super().__init__(
            conv=conv,
            in_channels=in_channels,
            num_convs=num_convs,
            add_norm=add_norm,
            internal_channels=internal_channels,
            num_classes=num_classes,
            anchors_per_pos=anchors_per_pos,
            num_levels=num_levels,
            **kwargs,
            )

        self.loss = FocalLossWithLogits(
            gamma=gamma,
            alpha=alpha,
            reduction=reduction,
            loss_weight=loss_weight,
            )
        self.logits_convert_fn = nn.Sigmoid()


ClassifierType = TypeVar('ClassifierType', bound=Classifier)
