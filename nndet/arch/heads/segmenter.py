import torch
import torch.nn as nn

from torch import Tensor
from typing import Dict, List, Union, Sequence, Optional, Tuple, TypeVar

from nndet.arch.conv import compute_padding_for_kernel, conv_kwargs_helper
from nndet.arch.heads.comb import AbstractHead
from nndet.arch.layers.interpolation import InterpolateToShapes
from nndet.losses.segmentation import SoftDiceLoss, TopKLoss


class Segmenter(AbstractHead):
    def __init__(self,
                 seg_classes: int,
                 in_channels: Sequence[int],
                 decoder_levels: Sequence[int],
                 **kwargs,
                 ):
        """
        分割头的抽象接口

        参数:
            seg_classes: 前景类别数量
                (!! 内部会+1用于背景类 !!)
            in_channels: 所有解码器级别的输入通道数
            decoder_levels: 用于检测的解码器级别
        """
        super().__init__()
        self.seg_classes = seg_classes + 1
        self.in_channels = in_channels
        self.decoder_levels = decoder_levels


class DiCESegmenter(Segmenter):
    def __init__(self,
                 conv,
                 seg_classes: int,
                 in_channels: Sequence[int],
                 decoder_levels: Sequence[int],
                 internal_channels: Optional[int] = None,
                 num_internal: int = 0,
                 add_norm: bool = True,
                 add_act: bool= True,
                 kernel_size: Union[int, Sequence[int]] = 3,
                 alpha: float = 0.5,
                 ce_kwargs: Optional[dict] = None,
                 dice_kwargs: Optional[dict] = None,
                 **kwargs,
                 ):
        """
        带有Dice和交叉熵损失的基本分割头
        (num_internal x conv [kernel_size]) -> 最终卷积 [1x1]

        参数:
            conv: 处理单层的卷积模块
            seg_classes: 前景类别数量
                (!! 内部会+1用于背景类 !!)
            in_channels: 所有解码器级别的输入通道数
            decoder_levels: 用于检测的解码器级别
            internal_channels: 内部卷积的通道数
            num_internal: 内部卷积的数量
            add_norm: 在内部卷积中添加归一化层
            add_act: 在内部卷积中添加激活层
            kernel_size: 卷积核大小
            alpha: 权衡Dice和交叉熵损失 (alpha * ce + (1-alpha) * soft_dice)
            ce_kwargs: 传递给交叉熵损失的关键字参数
            dice_kwargs: 传递给Dice损失的关键字参数
        """
        super().__init__(
            seg_classes=seg_classes,
            in_channels=in_channels,
            decoder_levels=decoder_levels,
            )
        self.num_internal = num_internal
        
        if internal_channels is None:
            self.internal_channels = self.in_channels[0]
        else:
            self.internal_channels = internal_channels

        self.conv_out = self.build_conv_out(conv)
        self.conv_intermediate = self.build_conv_internal(
            conv,
            kernel_size=kernel_size,
            add_norm=add_norm,
            add_act=add_act,
            **kwargs,
        )

        if dice_kwargs is None:
            dice_kwargs = {}
        dice_kwargs.setdefault("smooth_nom", 1e-5)
        dice_kwargs.setdefault("smooth_denom", 1e-5)
        dice_kwargs.setdefault("do_bg", False)
        self.dice_loss = SoftDiceLoss(nonlin=torch.nn.Softmax(dim=1), **dice_kwargs)

        if ce_kwargs is None:
            ce_kwargs = {}
        self.ce_loss = torch.nn.CrossEntropyLoss(**ce_kwargs)

        self.logits_convert_fn = nn.Softmax(dim=1)
        self.alpha = alpha

    def build_conv_out(self, conv) -> nn.Module:
        """
        构建输出卷积
        """
        _intermediate_channels = self.internal_channels if self.num_internal > 0 else self.in_channels[0]
        return conv(
            _intermediate_channels,
            self.seg_classes,
            kernel_size=1,
            padding=0,
            add_norm=None,
            add_act=None,
            bias=True,
            )

    def build_conv_internal(self,
                            conv,
                            kernel_size: Union[int, Tuple[int]],
                            add_norm: bool,
                            add_act: bool,
                            **kwargs,
                            ) -> Optional[nn.Module]:
        """
        构建内部卷积
        """
        padding = compute_padding_for_kernel(kernel_size)
        if self.num_internal > 0:
            _intermediate = torch.nn.Sequential()
            for i in range(self.num_internal):
                _intermediate.add_module(
                    f"c_intermediate{i}",
                    conv(
                        self.in_channels if i == 0 else self.internal_channels,
                        self.internal_channels,
                        kernel_size=kernel_size,
                        padding=padding,
                        stride=1,
                        add_norm=add_norm,
                        add_act=add_act,
                        **kwargs
                        )
                    )
        else:
            _intermediate = None
        return _intermediate

    def forward(self,
                x: List[torch.Tensor],
                ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        参数:
            x: 解码器产生的所有特征图，从最大到最小排列。

        返回:
            torch.Tensor: 结果
        """
        x = x[0]
        if self.conv_intermediate is not None:
            x = self.conv_intermediate(x)
        return {"seg_logits": self.conv_out(x)}

    def compute_loss(self,
                     pred_seg: Dict[str, torch.Tensor],
                     target: torch.Tensor,
                     ) -> Dict[str, torch.Tensor]:
        """
        计算加权的Dice和交叉熵损失

        参数:
            pred_seg: 分割预测结果
                `seg_logits`: 预测的logits
            target: 顶层的地面真实分割标签

        返回:
            Dict[str, torch.Tensor]: 计算的损失(包含在seg键中)
        """
        seg_logits = pred_seg["seg_logits"]
        return {
            "seg_ce": self.alpha * self.ce_loss(seg_logits, target.long()),
            "seg_dice": (1 - self.alpha) * self.dice_loss(seg_logits, target),
            }

    def postprocess_for_inference(self,
                                  prediction: Dict[str, torch.Tensor],
                                  *args, **kwargs,
                                  ) -> Dict[str, torch.Tensor]:
        """
        为推理后处理预测结果，例如将logits转换为概率

        参数:
            Dict[str, torch.Tensor]: 该头部的预测结果
                `seg_logits`: 预测的logits

        返回:
            Dict[str, torch.Tensor]: 后处理的预测结果
                `pred_seg`: 预测的概率 [N, C, dims]
        """
        return {"pred_seg": self.logits_convert_fn(prediction["seg_logits"])}


class DiCESegmenterFgBg(DiCESegmenter):
    def __init__(self,
                 conv,
                 seg_classes: int,
                 in_channels: Sequence[int],
                 decoder_levels: Sequence[int],
                 internal_channels: Optional[int] = None,
                 num_internal: int = 0,
                 add_norm: bool = True,
                 add_act: bool= True,
                 kernel_size: Union[int, Sequence[int]] = 3,
                 alpha: float = 0.5,
                 **kwargs,
                 ):
        """
        只区分前景和背景的带有Dice和交叉熵损失的基本分割头
        (num_internal x conv [kernel_size]) -> 最终卷积 [1x1]

        参数:
            conv: 处理单层的卷积模块
            seg_classes: 忽略!
            in_channels: 所有解码器级别的输入通道数
            decoder_levels: 用于检测的解码器级别
            internal_channels: 内部卷积的通道数
            num_internal: 内部卷积的数量
            add_norm: 在内部卷积中添加归一化层
            add_act: 在内部卷积中添加激活层
            kernel_size: 卷积核大小
            alpha: 权衡Dice和交叉熵损失 (alpha * ce + (1-alpha) * soft_dice)
        """
        super(DiCESegmenter, self).__init__(
            seg_classes=1,
            in_channels=in_channels,
            decoder_levels=decoder_levels,
            )
        self.num_internal = num_internal
        
        if internal_channels is None:
            self.internal_channels = self.in_channels[0]
        else:
            self.internal_channels = internal_channels

        self.conv_out = self.build_conv_out(conv)
        self.conv_intermediate = self.build_conv_internal(
            conv,
            kernel_size=kernel_size,
            add_norm=add_norm,
            add_act=add_act,
            **kwargs,
        )

        dice_kwargs = {"smooth_nom": 1e-5, "smooth_denom": 1e-5, "do_bg": True}
        self.dice_loss = SoftDiceLoss(nonlin=torch.nn.Softmax(dim=1), **dice_kwargs)
        self.ce_loss = torch.nn.CrossEntropyLoss()

        self.logits_convert_fn = nn.Softmax(dim=1)
        self.alpha = alpha

    def compute_loss(self,
                     pred_seg: Dict[str, torch.Tensor],
                     target: torch.Tensor,
                     ) -> Dict[str, torch.Tensor]:
        """
        计算加权的Dice和交叉熵损失

        参数:
            pred_seg: 分割预测结果
                `seg_logits`: 预测的logits
            target: 顶层的地面真实分割标签

        返回:
            Dict[str, torch.Tensor]: 计算的损失(包含在seg键中)
        """
        # 忽略所有大于0的类别，将它们全部视为前景
        target_seg = torch.zeros_like(target)
        target_seg[target > 0] = 1

        seg_logits = pred_seg["seg_logits"]
        return {
            "seg_ce": self.alpha * self.ce_loss(seg_logits, target_seg.long()),
            "seg_dice": (1 - self.alpha) * self.dice_loss(seg_logits, target_seg),
            }


class DiceTopKSegmenter(DiCESegmenter):
    def __init__(self,
                 conv,
                 seg_classes: int,
                 in_channels: Sequence[int],
                 decoder_levels: Sequence[int],
                 internal_channels: Optional[int] = None,
                 num_internal: int = 0,
                 add_norm: bool = True,
                 add_act: bool= True,
                 kernel_size: Union[int, Sequence[int]] = 3,
                 alpha: float = 0.5,
                 topk: float = 0.1,
                 **kwargs,
                 ):
        """
        带有Dice和TopK交叉熵损失的基本分割头，只考虑前K%的最困难体素
        (num_internal x conv [kernel_size]) -> 最终卷积 [1x1]

        参数:
            conv: 处理单层的卷积模块
            seg_classes: 前景类别数量
                (!! 内部会+1用于背景类 !!)
            in_channels: 所有解码器级别的输入通道数
            decoder_levels: 用于检测的解码器级别
            internal_channels: 内部卷积的通道数
            num_internal: 内部卷积的数量
            add_norm: 在内部卷积中添加归一化层
            add_act: 在内部卷积中添加激活层
            kernel_size: 卷积核大小
            alpha: 权衡Dice和交叉熵损失 (alpha * ce + (1-alpha) * soft_dice)
            topk: 要考虑的困难样本的比例
        """
        super(DiCESegmenter, self).__init__(
            seg_classes=seg_classes,
            in_channels=in_channels,
            decoder_levels=decoder_levels,
            )
        self.num_internal = num_internal
        
        if internal_channels is None:
            self.internal_channels = self.in_channels[0]
        else:
            self.internal_channels = internal_channels

        self.conv_out = self.build_conv_out(conv)
        self.conv_intermediate = self.build_conv_internal(
            conv,
            kernel_size=kernel_size,
            add_norm=add_norm,
            add_act=add_act,
            **kwargs,
        )

        dice_kwargs = {"smooth_nom": 1e-5, "smooth_denom": 1e-5, "do_bg": False}
        self.dice_loss = SoftDiceLoss(nonlin=torch.nn.Softmax(dim=1), **dice_kwargs)
        self.ce_loss = TopKLoss(k=topk)

        self.logits_convert_fn = nn.Softmax(dim=1)
        self.alpha = alpha


class DiceTopKSegmenterFgBg(DiCESegmenterFgBg):
    def __init__(self,
                 conv,
                 seg_classes: int,
                 in_channels: Sequence[int],
                 decoder_levels: Sequence[int],
                 internal_channels: Optional[int] = None,
                 num_internal: int = 0,
                 add_norm: bool = True,
                 add_act: bool= True,
                 kernel_size: Union[int, Sequence[int]] = 3,
                 alpha: float = 0.5,
                 topk: float = 0.1,
                 **kwargs,
                 ):
        """
        只区分前景和背景的带有Dice和TopK交叉熵损失的基本分割头
        (num_internal x conv [kernel_size]) -> 最终卷积 [1x1]

        参数:
            conv: 处理单层的卷积模块
            seg_classes: 忽略!
            in_channels: 所有解码器级别的输入通道数
            decoder_levels: 用于检测的解码器级别
            internal_channels: 内部卷积的通道数
            num_internal: 内部卷积的数量
            add_norm: 在内部卷积中添加归一化层
            add_act: 在内部卷积中添加激活层
            kernel_size: 卷积核大小
            alpha: 权衡Dice和交叉熵损失 (alpha * ce + (1-alpha) * soft_dice)
            topk: 要考虑的困难样本的比例
        """
        super(DiCESegmenter, self).__init__(
            seg_classes=1,
            in_channels=in_channels,
            decoder_levels=decoder_levels,
            )
        self.num_internal = num_internal
        
        if internal_channels is None:
            self.internal_channels = self.in_channels[0]
        else:
            self.internal_channels = internal_channels

        self.conv_out = self.build_conv_out(conv)
        self.conv_intermediate = self.build_conv_internal(
            conv,
            kernel_size=kernel_size,
            add_norm=add_norm,
            add_act=add_act,
            **kwargs,
        )

        dice_kwargs = {"smooth_nom": 1e-5, "smooth_denom": 1e-5, "do_bg": True}
        self.dice_loss = SoftDiceLoss(nonlin=torch.nn.Softmax(dim=1), **dice_kwargs)
        self.ce_loss = TopKLoss(k=topk)

        self.logits_convert_fn = nn.Softmax(dim=1)
        self.alpha = alpha


class DeepSupervisionSegmenterFGBG(DiCESegmenterFgBg):
    def __init__(self,
                 conv,
                 seg_classes: int,
                 in_channels: Sequence[int],
                 decoder_levels: Sequence[int],
                 internal_channels: Optional[int] = None,
                 num_internal: int = 0,
                 add_norm: bool = True,
                 add_act: bool= True,
                 kernel_size: Union[int, Sequence[int]] = 3,
                 alpha: float = 0.5,
                 dsv_weight: float = 1.,
                 **kwargs,
                 ):
        """
        带有Dice和交叉熵损失的深度监督前景-背景分割头
        (num_internal x conv [kernel_size]) -> 最终卷积 [1x1]
        为所有层级计算分割损失，然后将它们相加

        参数:
            conv: 处理单层的卷积模块
            seg_classes: 忽略!
            in_channels: 所有解码器级别的输入通道数
            decoder_levels: 用于检测的解码器级别
            internal_channels: 内部卷积的通道数
            num_internal: 内部卷积的数量
            add_norm: 在内部卷积中添加归一化层
            add_act: 在内部卷积中添加激活层
            kernel_size: 卷积核大小
            alpha: 权衡Dice和交叉熵损失 (alpha * ce + (1-alpha) * soft_dice)
            dsv_weight: 深度监督的权重
        """
        super(DiCESegmenter, self).__init__(
            seg_classes=1,
            in_channels=in_channels,
            decoder_levels=decoder_levels,
            )

        self.deep_supervision = True
        self.dim = len(in_channels)
        self.dsv_weight = dsv_weight
        self.num_dsv = len(decoder_levels)
        
        if internal_channels is None:
            self.internal_channels = self.in_channels[0]
        else:
            self.internal_channels = internal_channels

        # 为所有层级创建卷积
        self.conv_out = nn.ModuleList()
        self.conv_intermediate = nn.ModuleList()
        for level in range(self.num_dsv):
            self.conv_intermediate.append(self.build_conv_internal(
                conv,
                kernel_size=kernel_size,
                add_norm=add_norm,
                add_act=add_act,
                **kwargs,
            ))
            
            _intermediate_channels = self.internal_channels if num_internal > 0 else in_channels[level]
            self.conv_out.append(conv(
                _intermediate_channels,
                self.seg_classes,
                kernel_size=1,
                padding=0,
                add_norm=None,
                add_act=None,
                bias=True,
                ))

        # 如果所有特征图不同，需要一个插值模块
        self.interp = None
        if self.num_dsv > 1:
            self.interp = InterpolateToShapes()

        dice_kwargs = {"smooth_nom": 1e-5, "smooth_denom": 1e-5, "do_bg": True}
        self.dice_loss = SoftDiceLoss(nonlin=torch.nn.Softmax(dim=1), **dice_kwargs)
        self.ce_loss = torch.nn.CrossEntropyLoss()

        self.logits_convert_fn = nn.Softmax(dim=1)
        self.alpha = alpha

    def forward(self,
                x: List[torch.Tensor],
                ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        参数:
            x: 解码器产生的所有特征图，从最大到最小排列。

        返回:
            Dict[str, torch.Tensor]: 结果
                `seg_logits`: 最顶层的预测结果
                `seg_logits_deep0` - `seg_logits_deepN`: 中间层的预测结果
        """
        result = {}
        xs = [x[i] for i in self.decoder_levels]
        
        for idx, (_x, _conv_intermediate, _conv_out) in enumerate(zip(
                xs, self.conv_intermediate, self.conv_out)):
            if _conv_intermediate is not None:
                _x = _conv_intermediate(_x)
            
            if idx == 0:
                result["seg_logits"] = _conv_out(_x)
            else:
                result[f"seg_logits_deep{idx - 1}"] = _conv_out(_x)
        return result

    def compute_loss(self,
                     pred_seg: Dict[str, torch.Tensor],
                     target: torch.Tensor,
                     ) -> Dict[str, torch.Tensor]:
        """
        计算加权的Dice和交叉熵损失

        参数:
            pred_seg: 分割预测结果
                `seg_logits`: 预测的logits (最顶层)
                `seg_logits_deep0` - `seg_logits_deepN`: 中间层的预测结果
            target: 顶层的地面真实分割标签

        返回:
            Dict[str, torch.Tensor]: 计算的损失(包含在seg键中)
        """
        # 忽略所有大于0的类别，将它们全部视为前景
        target_seg = torch.zeros_like(target)
        target_seg[target > 0] = 1

        # 处理最顶层的分割
        result = {}
        dsv_cnt = 0
        for key, pred in pred_seg.items():
            if self.interp is not None and key != "seg_logits":
                pred = self.interp(pred, [pred_seg["seg_logits"].size()])

            if key == "seg_logits":
                result["seg_ce"] = self.alpha * self.ce_loss(pred, target_seg.long())
                result["seg_dice"] = (1 - self.alpha) * self.dice_loss(pred, target_seg)
            else:
                result[f"{key}_ce"] = self.dsv_weight * (self.alpha * self.ce_loss(pred, target_seg.long()))
                result[f"{key}_dice"] = self.dsv_weight * ((1 - self.alpha) * self.dice_loss(pred, target_seg))
                dsv_cnt += 1
        return result

    def _compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        内部计算损失方法

        参数:
            pred: 预测的logits
            target: 目标分割

        返回:
            torch.Tensor: 计算的损失
        """
        return self.alpha * self.ce_loss(pred, target.long()) + \
               (1 - self.alpha) * self.dice_loss(pred, target)


SegmenterType = TypeVar('SegmenterType', bound=Segmenter)
