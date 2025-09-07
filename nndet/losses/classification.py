import torch
import torch.nn.functional as F
import torch.nn as nn

from torch import Tensor
from loguru import logger

from nndet.losses.base import reduction_helper
from nndet.utils import make_onehot_batch


def one_hot_smooth(data,
                   num_classes: int,
                   smoothing: float = 0.0,
                   ):
    targets = torch.empty(size=(*data.shape, num_classes), device=data.device)\
        .fill_(smoothing / num_classes)\
        .scatter_(-1, data.long().unsqueeze(-1), 1. - smoothing)
    return targets


@torch.jit.script
def focal_loss_with_logits(
        logits: torch.Tensor,
        target: torch.Tensor, gamma: float,
        alpha: float = -1,
        reduction: str = "mean",
        ) -> torch.Tensor:
    """
    Focal loss
    https://arxiv.org/abs/1708.02002

    参数:
        logits: 预测的logits [N, dims]
        target: (float)二元目标 [N, dims]
        gamma: 在focal loss中平衡简单和困难样本
        alpha: 平衡正样本和负样本 [0, 1] (增加alpha会增加前景类的权重(更好的召回率))
        reduction: 'mean'|'sum'|'none'
            mean: 整个批次上损失的平均值
            sum: 整个批次上损失的总和
            none: 不进行归约

    返回:
        torch.Tensor: 损失值

    另见:
        :class:`BFocalLossWithLogits`, :class:`FocalLossWithLogits`
    """
    bce_loss = F.binary_cross_entropy_with_logits(logits, target, reduction='none')

    p = torch.sigmoid(logits)
    pt = (p * target + (1 - p) * (1 - target))

    focal_term = (1. - pt).pow(gamma)
    loss = focal_term * bce_loss

    if alpha >= 0:
        alpha_t = (alpha * target + (1 - alpha) * (1 - target))
        loss = alpha_t * loss

    return reduction_helper(loss, reduction=reduction)


class FocalLossWithLogits(nn.Module):
    def __init__(self,
                 gamma: float = 2,
                 alpha: float = -1,
                 reduction: str = "sum",
                 loss_weight: float = 1.,
                 ):
        """
        多类别的Focal loss（使用one-hot编码和sigmoid）

        参数:
            gamma: 在focal loss中平衡简单和困难样本
            alpha: 平衡正样本和负样本 [0, 1] (增加alpha会增加前景类的权重(更好的召回率))
            reduction: 'mean'|'sum'|'none'
                mean: 整个批次上损失的平均值
                sum: 整个批次上损失的总和
                none: 不进行归约
            loss_weight: 平衡多个损失的标量
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                logits: torch.Tensor,
                targets: torch.Tensor,
                ) -> torch.Tensor:
        """
        计算损失

        参数:
            logits: 预测的logits [N, C, dims]，其中N是批次大小，
                C是类别数量，dims是任意空间维度
                (如果启用忽略背景，背景类应该位于通道0)
            targets: 编码为数字的目标 [N, dims]，其中N是批次大小，
                dims是任意空间维度

        返回:
            torch.Tensor: 损失值
        """
        n_classes = logits.shape[1] + 1
        target_onehot = make_onehot_batch(targets, n_classes=n_classes).float()
        target_onehot = target_onehot[:, 1:]

        return self.loss_weight * focal_loss_with_logits(
            logits, target_onehot,
            gamma=self.gamma,
            alpha=self.alpha,
            reduction=self.reduction,
            )


class BCEWithLogitsLossOneHot(torch.nn.BCEWithLogitsLoss):
    def __init__(self,
                 *args,
                 num_classes: int,
                 smoothing: float = 0.0,
                 loss_weight: float = 1.,
                 **kwargs,
                 ):
        """
        带有目标one-hot编码的BCE损失

        参数:
            num_classes: 类别数量
            smoothing: 标签平滑参数
            loss_weight: 平衡多个损失的标量
        """
        super().__init__(*args, **kwargs)
        self.smoothing = smoothing
        if smoothing > 0:
            logger.info(f"使用标签平滑，平滑参数为: {smoothing}")
        self.num_classes = num_classes
        self.loss_weight = loss_weight

    def forward(self,
                input: Tensor,
                target: Tensor,
                ) -> Tensor:
        """
        基于one-hot编码计算BCE损失

        参数:
            input: 所有前景类的logits [N, C]
                N是锚点数量，C是前景类数量
            target: 目标类别。0被视为背景，>0被视为前景类。
                [N]是锚点数量

        返回:
            Tensor: 最终损失
        """
        target_one_hot = one_hot_smooth(
            target, num_classes=self.num_classes + 1, smoothing=self.smoothing)  # [N, C + 1]
        target_one_hot = target_one_hot[:, 1:]  # 背景隐式编码

        return self.loss_weight * super().forward(input, target_one_hot.float())


class CrossEntropyLoss(torch.nn.CrossEntropyLoss):
    def __init__(self,
                 *args,
                 loss_weight: float = 1.,
                 **kwargs,
                 ) -> None:
        """
        与PyTorch的CE相同，但额外添加了损失权重以提供统一的API
        """
        super().__init__(*args, **kwargs)
        self.loss_weight = loss_weight

    def forward(self,
                input: Tensor,
                target: Tensor,
                ) -> Tensor:
        """
        与PyTorch的CE相同
        """
        return self.loss_weight * super().forward(input, target)
