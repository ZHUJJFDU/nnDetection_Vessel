from loguru import logger
import torch
import torch.nn as nn
from torch import Tensor
from typing import Callable


def one_hot_smooth_batch(data, num_classes: int, smoothing: float = 0.0):
    shape = data.shape
    targets = torch.empty(size=(shape[0], num_classes, *shape[1:]), device=data.device)\
        .fill_(smoothing / num_classes)\
        .scatter_(1, data.long().unsqueeze(1), 1. - smoothing)
    return targets


def get_tp_fp_fn(net_output, gt, axes=None, mask=None, square=False):
    """
    net_output必须是(b, c, x, y(, z)))
    gt必须是标签图（形状为(b, 1, x, y(, z))或(b, x, y(, z))）或one-hot编码(b, c, x, y(, z))
    如果提供mask，它必须具有形状(b, 1, x, y(, z)))
    :param net_output: 网络输出
    :param gt: 真实标签
    :param axes: 轴
    :param mask: mask必须为有效像素值1，无效像素值0
    :param square: 如果为True，则fp、tp和fn在求和前会被平方
    :return: 真阳性、假阳性和假阴性
    """
    if axes is None:
        axes = tuple(range(2, len(net_output.size())))

    shp_x = net_output.shape
    shp_y = gt.shape

    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            gt = gt.view((shp_y[0], 1, *shp_y[1:]))

        if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
            # 如果是这种情况，那么gt可能已经是one-hot编码
            y_onehot = gt
        else:
            gt = gt.long()
            y_onehot = torch.zeros(shp_x)
            if net_output.device.type == "cuda":
                y_onehot = y_onehot.cuda(net_output.device.index)
            y_onehot.scatter_(1, gt, 1)

    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot

    if mask is not None:
        tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
        fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
        fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2

    tp = tp.sum(dim=axes, keepdim=False)
    fp = fp.sum(dim=axes, keepdim=False)
    fn = fn.sum(dim=axes, keepdim=False)
    return tp, fp, fn


class SoftDiceLoss(nn.Module):
    def __init__(self,
                 nonlin: Callable = None,
                 batch_dice: bool = False, 
                 do_bg: bool = False,
                 smooth_nom: float = 1e-5,
                 smooth_denom: float = 1e-5,
                 ):
        """
        软Dice损失
        
        参数:
            nonlin: 非线性激活函数
            batch_dice: 是否将批次视为伪体积
            do_bg: 是否在Dice计算中包含背景
            smooth_nom: 分子平滑项
            smooth_denom: 分母平滑项
        """
        super().__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.nonlin = nonlin
        self.smooth_nom = smooth_nom
        self.smooth_denom = smooth_denom
        logger.info(f"在Dice损失中使用batch_dice={self.batch_dice}和"
                    f"do_bg={self.do_bg}")

    def forward(self,
                inp: torch.Tensor,
                target: torch.Tensor,
                loss_mask: torch.Tensor=None,
                ):
        """
        计算损失
        
        参数:
            inp (torch.Tensor): 预测结果
            target (torch.Tensor): 真实标签
            loss_mask ([torch.Tensor], optional): 二元掩码。默认为None
        
        返回:
            torch.Tensor: 软Dice损失
        """
        shp_x = inp.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.nonlin is not None:
            inp = self.nonlin(inp)

        tp, fp, fn = get_tp_fp_fn(inp, target, axes, loss_mask, False)

        nominator = 2 * tp + self.smooth_nom
        denominator = 2 * tp + fp + fn + self.smooth_denom

        dc = nominator / denominator

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        dc = dc.mean()

        return 1 - dc


class TopKLoss(torch.nn.CrossEntropyLoss):
    def __init__(self,
                 topk: float,
                 loss_weight: float = 1.,
                 **kwargs,
                 ):
        """
        使用前k%的值计算交叉熵损失
        (输入应为softmax前的logits!)

        参数:
            topk: 用于损失计算的所有条目的百分比
            loss_weight: 平衡多个损失的标量
        """
        if "reduction" in kwargs:
            raise ValueError("TopKLoss不支持reduction参数。"
                             "该损失将始终返回平均值！")
        super().__init__(
            reduction="none",
            **kwargs,
        )
        if topk < 0 or topk > 1:
            raise ValueError("topk需要在[0, 1]范围内。")
        self.topk = topk
        self.loss_weight = loss_weight

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """
        计算CE损失，并使用前k%条目的平均值

        参数:
            input: 所有前景类的logits [N, C, *]
            target: 目标类别。0被视为背景，>0被视为前景类。[N, *]

        返回:
            Tensor: 最终损失
        """
        losses = super().forward(input, target)

        k = int(losses.numel() * self.topk)
        return self.loss_weight * losses.view(-1).topk(k=k, sorted=False)[0].mean()


class TopKLossSigmoid(torch.nn.BCEWithLogitsLoss):
    def __init__(self,
                 num_classes: int,
                 topk: float,
                 smoothing: float = 0.0,
                 loss_weight: float = 1.,
                 **kwargs,
                 ):
        """
        使用前k%的值计算带有one-hot编码的BCE损失
        (通过one-hot支持多类别，输入应为sigmoid前的logits!)

        参数:
            num_classes: 类别数量
            topk: 用于损失计算的所有条目的百分比
            smoothing: 标签平滑参数
            loss_weight: 平衡多个损失的标量
        """
        if "reduction" in kwargs:
            raise ValueError("TopKLoss不支持reduction参数。"
                             "该损失将始终返回平均值！")
        super().__init__(
            reduction="none",
            **kwargs,
        )
        self.smoothing = smoothing
        if smoothing > 0:
            logger.info(f"使用标签平滑，平滑参数为: {smoothing}")
        self.num_classes = num_classes

        self.topk = topk
        self.loss_weight = loss_weight

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """
        基于前景类的one-hot编码计算BCE损失，
        并使用前k%条目的平均值

        参数:
            input: 所有前景类的logits [N, C, *]
            target: 目标类别 [N, *]。目标将被one-hot编码，
                0被视为背景类并被移除。

        返回:
            Tensor: 最终损失
        """
        target_one_hot = one_hot_smooth_batch(
            target, num_classes=self.num_classes + 1, smoothing=self.smoothing)  # [N, C + 1]
        target_one_hot = target_one_hot[:, 1:]  # 背景隐式编码
        losses = super().forward(input, target_one_hot.float())

        k = int(losses.numel() * self.topk)
        return self.loss_weight * losses.view(-1).topk(k=k, sorted=False)[0].mean()

