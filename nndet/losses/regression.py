# Modifications licensed under:
# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
# SPDX-License-Identifier: Apache-2.0
#
# L1 loss from fvcore (https://github.com/facebookresearch/fvcore) licensed under
# SPDX-FileCopyrightText: 2019, Facebook, Inc
# SPDX-License-Identifier: Apache-2.0


from typing import Optional

import torch


__all__ = ["SmoothL1Loss", "smooth_l1_loss"]

from nndet.core.boxes.ops import generalized_box_iou
from nndet.losses.base import reduction_helper


class SmoothL1Loss(torch.nn.Module):
    def __init__(self,
                 beta: float,
                 reduction: Optional[str] = None,
                 loss_weight: float = 1.,
                 ):
        """
        函数的模块包装器

        参数:
            beta (float): L1转换为L2的变化点。
                对于beta值 < 1e-5，计算L1损失。
            reduction (str): 'none' | 'mean' | 'sum'
                 'none': 不对输出进行任何归约。
                 'mean': 输出将被平均。
                 'sum': 输出将被求和。
            loss_weight: 损失权重，用于平衡多个损失

        另见:
            :func:`smooth_l1_loss`
        """
        super().__init__()
        self.reduction = reduction
        self.beta = beta
        self.loss_weight = loss_weight

    def forward(self, inp: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算损失

        参数:
            inp (torch.Tensor): 预测张量（与目标形状相同）
            target (torch.Tensor): 目标张量

        返回:
            Tensor: 计算的损失
        """
        return self.loss_weight * reduction_helper(smooth_l1_loss(inp, target, self.beta), self.reduction)


def smooth_l1_loss(inp, target, beta: float):
    """
    来自 https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/smooth_l1_loss.py

    在Fast R-CNN论文中定义的平滑L1损失如下:
                  | 0.5 * x ** 2 / beta   如果 abs(x) < beta
    smoothl1(x) = |
                  | abs(x) - 0.5 * beta   否则,
    其中 x = input - target.
    平滑L1损失与Huber损失相关，Huber损失定义为:
               | 0.5 * x ** 2                  如果 abs(x) < beta
    huber(x) = |
               | beta * (abs(x) - 0.5 * beta)  否则
    平滑L1损失等于huber(x) / beta。这导致以下差异:
     - 当beta -> 0时，平滑L1损失收敛到L1损失，而Huber损失收敛到常数0损失。
     - 当beta -> +inf时，平滑L1收敛到常数0损失，而Huber损失收敛到L2损失。
     - 对于平滑L1损失，随着beta变化，损失的L1部分具有常数斜率1。对于Huber损失，L1部分的斜率是beta。
    平滑L1损失可以看作是完全的L1损失，但abs(x) < beta部分被一个二次函数替换，
    使得在abs(x) = beta处，其斜率为1。这个二次段在x = 0附近平滑了L1损失。

    参数:
        inp (Tensor): 任意形状的输入张量
        target (Tensor): 与输入形状相同的目标值张量
        beta (float): L1到L2的变化点。
            对于beta值 < 1e-5，计算L1损失。

    返回:
        Tensor: 应用归约选项后的损失。

    注意:
        PyTorch内置的"平滑L1损失"实现实际上并不实现平滑L1损失，
        也不实现Huber损失。它实现了两者相等的特殊情况（beta=1）。
        参见: https://pytorch.org/docs/stable/nn.html#torch.nn.SmoothL1Loss.
     """
    if beta < 1e-5:
        # 如果beta == 0，那么torch.where将在应用链式法则时导致梯度为nan，
        # 这是由于pytorch实现细节（False分支"0.5 * n ** 2 / 0"有一个
        # 零的传入梯度，而不是"没有梯度"）。为避免此问题，我们定义
        # beta的小值恰好为l1损失。
        loss = torch.abs(inp - target)
    else:
        n = torch.abs(inp - target)
        cond = n < beta
        loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    return loss


class GIoULoss(torch.nn.Module):
    def __init__(self,
                 reduction: Optional[str] = None,
                 eps: float = 1e-7,
                 loss_weight: float = 1.,
                 ):
        """
        广义IoU损失
        `Generalized Intersection over Union: A Metric and A Loss for Bounding
        Box Regression` https://arxiv.org/abs/1902.09630

        参数:
            reduction: 归约类型 'none' | 'mean' | 'sum'
            eps: 用于数值稳定性的小常数
            loss_weight: 损失权重，用于平衡多个损失

        注意:
            原始论文使用lambda=10来平衡PASCAL VOC和COCO的回归和分类损失
            （未针对COCO进行调整）

            `End-to-End Object Detection with Transformers` https://arxiv.org/abs/2005.12872
            "我们增强的Faster-RCNN+基线使用GIoU [38]损失以及标准l1损失进行边界框回归。
            我们进行了网格搜索以找到损失的最佳权重，最终模型仅使用GIoU损失，
            分别为框回归和建议回归任务使用权重20和1。"
        """
        super().__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """
        计算广义IoU损失

        参数:
            pred_boxes: 预测的边界框 (x1, y1, x2, y2, (z1, z2)) [N, dim * 2]
            target_boxes: 目标边界框 (x1, y1, x2, y2, (z1, z2)) [N, dim * 2]

        返回:
            Tensor: 损失值
        """
        loss = reduction_helper(
            torch.diag(generalized_box_iou(pred_boxes, target_boxes, eps=self.eps),
                       diagonal=0),
            reduction=self.reduction)
        return self.loss_weight * -1 * loss
