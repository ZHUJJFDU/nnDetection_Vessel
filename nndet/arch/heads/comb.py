# 导入必要的库
import torch
import torch.nn as nn

from torch import Tensor
from typing import Dict, List, Tuple, Optional, TypeVar
from abc import abstractmethod

from nndet.core.boxes import BoxCoderND
from nndet.core.boxes.sampler import AbstractSampler
from nndet.arch.heads.classifier import Classifier
from nndet.arch.heads.regressor import Regressor


class AbstractHead(nn.Module):
    """
    提供一个抽象接口，用于接收输入并计算自身损失的模块
    """
    @abstractmethod
    def forward(self, x: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        计算前向传播
        
        参数：
            x: 特征图列表
        """
        raise NotImplementedError 

    @abstractmethod
    def compute_loss(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        """
        计算损失
        """
        raise NotImplementedError

    @abstractmethod
    def postprocess_for_inference(self,
                                  prediction: Dict[str, torch.Tensor],
                                  *args, **kwargs,
                                  ) -> Dict[str, torch.Tensor]:
        """
        为推理后处理预测结果，例如将logits转换为概率

        参数：
            Dict[str, torch.Tensor]: 该头部的预测结果
            List[torch.Tensor]: 每张图像的锚框
        """
        raise NotImplementedError


class DetectionHead(AbstractHead):
    def __init__(self,
                 classifier: Classifier,
                 regressor: Regressor,
                 coder: BoxCoderND,
                 ):
        """
        带有分类器和回归模块的检测头
        
        参数：
            classifier: 分类器模块
            regressor: 回归模块
        """
        super().__init__()
        self.classifier = classifier
        self.regressor = regressor
        self.coder = coder

    def forward(self,
                fmaps: List[torch.Tensor],
                ) -> Dict[str, torch.Tensor]:
        """
        通过头部模块前向传播特征图

        参数：
            fmaps: 头部模块的特征图列表

        返回：
            Dict[str, torch.Tensor]: 预测结果
                `box_deltas`(Tensor): 边界框偏移
                    [Num_Anchors_Batch, (dim * 2)]
                `box_logits`(Tensor): 分类logits
                    [Num_Anchors_Batch, (num_classes)]
        """
        logits, offsets = [], []
        for level, p in enumerate(fmaps):
            logits.append(self.classifier(p, level=level))
            offsets.append(self.regressor(p, level=level))

        sdim = fmaps[0].ndim - 2
        box_deltas = torch.cat(offsets, dim=1).reshape(-1, sdim * 2)
        box_logits = torch.cat(logits, dim=1).flatten(0, -2)
        return {"box_deltas": box_deltas, "box_logits": box_logits}

    @abstractmethod
    def compute_loss(self,
                     prediction: Dict[str, Tensor],
                     target_labels: List[Tensor],
                     matched_gt_boxes: List[Tensor],
                     anchors: List[Tensor],
                     ) -> Tuple[Dict[str, Tensor], torch.Tensor, torch.Tensor]:
        """
        计算回归和分类损失
        N个锚框覆盖所有图像；每张图像M个锚框 => sum(M) = N

        参数：
            prediction: 用于损失计算的检测预测结果
                `box_logits`: 每个锚框的分类logits [N]
                `box_deltas`: 每个锚框的偏移
                    (x1, y1, x2, y2, (z1, z2))[N, dim * 2]
            target_labels: 每个锚框的目标标签（每张图像）[M]
            matched_gt_boxes: 每个锚框匹配的gt框
                List[[N, dim *  2]], N=每张图像的锚框数量
            anchors: 每张图像的锚框 List[[N, dim *  2]]

        返回：
            Tensor: 包含损失的字典（reg为回归损失，cls为分类损失）
            Tensor: 采样的正样本锚框索引（连接后）
            Tensor: 采样的负样本锚框索引（连接后）
        """
        raise NotImplementedError

    def postprocess_for_inference(self,
                                  prediction: Dict[str, torch.Tensor],
                                  anchors: List[torch.Tensor],
                                  ) -> Dict[str, torch.Tensor]:
        """
        为推理后处理预测结果，例如将logits转换为概率

        参数：
            Dict[str, torch.Tensor]: 该头部的预测结果
                `box_logits`: 每个锚框的分类logits [N]
                `box_deltas`: 每个锚框的偏移
                    (x1, y1, x2, y2, (z1, z2))[N, dim * 2]
            List[torch.Tensor]: 每张图像的锚框
        """
        postprocess_predictions = {
            "pred_boxes": self.coder.decode(prediction["box_deltas"], anchors),
            "pred_probs": self.classifier.box_logits_to_probs(prediction["box_logits"]),
        }
        return postprocess_predictions


class DetectionHeadHNM(DetectionHead):
    def __init__(self,
                 classifier: Classifier,
                 regressor: Regressor,
                 coder: BoxCoderND,
                 sampler: AbstractSampler,
                 log_num_anchors: Optional[str] = "mllogger",
                 ):
        """
        带有分类器和回归模块的检测头。使用困难负样本挖掘来计算损失

        参数：
            classifier: 分类器模块
            regressor: 回归模块
            sampler (AbstractSampler): 用于选择正样本和负样本的采样器
            log_num_anchors (str): 要使用的日志记录器名称；如果为None，则不执行日志记录
        """
        super().__init__(classifier=classifier, regressor=regressor, coder=coder)

        self.logger = None # get_logger(log_num_anchors) if log_num_anchors is not None else None
        self.fg_bg_sampler = sampler

    def compute_loss(self,
                     prediction: Dict[str, Tensor],
                     target_labels: List[Tensor],
                     matched_gt_boxes: List[Tensor],
                     anchors: List[Tensor],
                     ) -> Tuple[Dict[str, Tensor], torch.Tensor, torch.Tensor]:
        """
        计算回归和分类损失
        N个锚框覆盖所有图像；每张图像M个锚框 => sum(M) = N

        参数：
            prediction: 用于损失计算的检测预测结果
                box_logits (Tensor): 每个锚框的分类logits
                    [N, num_classes]
                box_deltas (Tensor): 每个锚框的偏移
                    (x1, y1, x2, y2, (z1, z2))[N, dim * 2]
            target_labels (List[Tensor]): 每个锚框的目标标签
                （每张图像）[M]
            matched_gt_boxes: 每个锚框匹配的gt框
                List[[N, dim *  2]], N=每张图像的锚框数量
            anchors: 每张图像的锚框 List[[N, dim *  2]]

        返回：
            Tensor: 包含损失的字典（reg为回归损失，cls
                为分类损失）
            Tensor: 采样的正样本锚框索引（连接后）
            Tensor: 采样的负样本锚框索引（连接后）
        """
        box_logits, box_deltas = prediction["box_logits"], prediction["box_deltas"]

        losses = {}
        sampled_pos_inds, sampled_neg_inds = self.select_indices(target_labels, box_logits)
        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)

        target_labels = torch.cat(target_labels, dim=0)

        with torch.no_grad():
            batch_matched_gt_boxes = torch.cat(matched_gt_boxes, dim=0)
            batch_anchors = torch.cat(anchors, dim=0)
            target_deltas_sampled = self.coder.encode_single(
                batch_matched_gt_boxes[sampled_pos_inds], batch_anchors[sampled_pos_inds],
            )

        # target_deltas = self.coder.encode(matched_gt_boxes, anchors)
        # target_deltas_sampled = torch.cat(target_deltas, dim=0)[sampled_pos_inds]

        # assert len(batch_anchors) == len(batch_matched_gt_boxes)
        # assert len(batch_anchors) == len(box_deltas)
        # assert len(batch_anchors) == len(box_logits)
        # assert len(batch_anchors) == len(target_labels)

        if sampled_pos_inds.numel() > 0:
            losses["reg"] = self.regressor.compute_loss(
                box_deltas[sampled_pos_inds],
                target_deltas_sampled,
                ) / max(1, sampled_pos_inds.numel())

        losses["cls"] = self.classifier.compute_loss(
            box_logits[sampled_inds], target_labels[sampled_inds])
        return losses, sampled_pos_inds, sampled_neg_inds

    def select_indices(self,
                       target_labels: List[Tensor],
                       boxes_scores: Tensor,
                       ) -> Tuple[Tensor, Tensor]:
        """
        从目标标签中采样正样本和负样本锚框

        参数：
            target_labels (List[Tensor]): 每个锚框的目标标签
                （每张图像）[M]
            boxes_scores (Tensor): 每个锚框的分类logits
                [N, num_classes]

        返回：
            Tensor: 采样的正样本索引 [R]
            Tensor: 采样的负样本索引 [R]
        """
        boxes_max_fg_probs = self.classifier.box_logits_to_probs(boxes_scores)
        boxes_max_fg_probs = boxes_max_fg_probs.max(dim=1)[0]  # 搜索最大前景概率

        # 每张图像的正样本和负样本锚框索引
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(target_labels, boxes_max_fg_probs)
        sampled_pos_inds = torch.where(torch.cat(sampled_pos_inds, dim=0))[0]
        sampled_neg_inds = torch.where(torch.cat(sampled_neg_inds, dim=0))[0]

        # if self.logger:
        #     self.logger.add_scalar("train/num_pos", sampled_pos_inds.numel())
        #     self.logger.add_scalar("train/num_neg", sampled_neg_inds.numel())

        return sampled_pos_inds, sampled_neg_inds


class BoxHeadNoSampler(DetectionHead):
    def __init__(self,
                 classifier: Classifier,
                 regressor: Regressor,
                 coder: BoxCoderND,
                 log_num_anchors: Optional[str] = "mllogger",
                 **kwargs
                 ):
        """
        带有分类器和回归模块的检测头。使用所有
        前景锚框进行回归，并将所有锚框传递给分类器

        参数：
            classifier: 分类器模块
            regressor: 回归模块
            log_num_anchors (str): 要使用的日志记录器名称；如果为None，则不执行日志记录
        """
        super().__init__(classifier=classifier, regressor=regressor, coder=coder)
        self.logger = None # get_logger(log_num_anchors) if log_num_anchors is not None else None

    def compute_loss(self,
                     prediction: Dict[str, Tensor],
                     target_labels: List[Tensor],
                     matched_gt_boxes: List[Tensor],
                     anchors: List[Tensor],
                     ) -> Tuple[Dict[str, Tensor], torch.Tensor, Optional[torch.Tensor]]:
        """
        计算回归和分类损失
        N个锚框覆盖所有图像；每张图像M个锚框 => sum(M) = N

        参数：
            prediction: 用于损失计算的检测预测结果
                box_logits (Tensor): 每个锚框的分类logits
                    [N, num_classes]
                box_deltas (Tensor): 每个锚框的偏移
                    (x1, y1, x2, y2, (z1, z2))[N, dim * 2]
            target_labels: 每个锚框的目标标签（每张图像）[M]
            matched_gt_boxes: 每个锚框匹配的gt框
                List[[N, dim *  2]], N=每张图像的锚框数量
            anchors: 每张图像的锚框 List[[N, dim *  2]]

        返回：
            Tensor: 包含损失的字典（reg为回归损失，cls为
                分类损失）
            Tensor: 采样的正样本锚框索引（连接后）
            Tensor: 采样的负样本锚框索引（连接后）
        """
        box_logits, box_deltas = prediction["box_logits"], prediction["box_deltas"]

        target_labels = torch.cat(target_labels, dim=0)
        batch_anchors = torch.cat(anchors, dim=0)
        pred_boxes = self.coder.decode_single(box_deltas, batch_anchors)
        target_boxes = torch.cat(matched_gt_boxes, dim=0)

        sampled_inds = torch.where(target_labels >= 0)[0]
        sampled_pos_inds = torch.where(target_labels >= 1)[0]

        losses = {}
        if sampled_pos_inds.numel() > 0:
            losses["reg"] = self.regressor.compute_loss(
                pred_boxes[sampled_pos_inds],
                target_boxes[sampled_pos_inds],
            ) / max(1, sampled_pos_inds.numel())

        losses["cls"] = self.classifier.compute_loss(
            box_logits[sampled_inds],
            target_labels[sampled_inds],
            ) / max(1, sampled_pos_inds.numel())
        return losses, sampled_pos_inds, None


class DetectionHeadHNMNative(DetectionHeadHNM):
    def compute_loss(self,
                     prediction: Dict[str, Tensor],
                     target_labels: List[Tensor],
                     matched_gt_boxes: List[Tensor],
                     anchors: List[Tensor],
                     ) -> Tuple[Dict[str, Tensor], torch.Tensor, torch.Tensor]:
        """
        计算回归和分类损失
        N个锚框覆盖所有图像；每张图像M个锚框 => sum(M) = N

        该头部从网络中解码相对偏移，并直接在边界框上计算回归损失（例如用于GIoU损失）

        参数：
            prediction: 用于损失计算的检测预测结果
                box_logits (Tensor): 每个锚框的分类logits
                    [N, num_classes]
                box_deltas (Tensor): 每个锚框的偏移
                    (x1, y1, x2, y2, (z1, z2))[N, dim * 2]
            target_labels (List[Tensor]): 每个锚框的目标标签
                （每张图像）[M]
            matched_gt_boxes: 每个锚框匹配的gt框
                List[[N, dim *  2]], N=每张图像的锚框数量
            anchors: 每张图像的锚框 List[[N, dim *  2]]

        返回：
            Tensor: 包含损失的字典（reg为回归损失，cls为
                分类损失）
            Tensor: 采样的正样本锚框索引（连接后）
            Tensor: 采样的负样本锚框索引（连接后）
        """
        box_logits, box_deltas = prediction["box_logits"], prediction["box_deltas"]

        with torch.no_grad():
            losses = {}
            sampled_pos_inds, sampled_neg_inds = self.select_indices(target_labels, box_logits)
            sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)

        target_labels = torch.cat(target_labels, dim=0)
        batch_anchors = torch.cat(anchors, dim=0)
        pred_boxes_sampled = self.coder.decode_single(
            box_deltas[sampled_pos_inds], batch_anchors[sampled_pos_inds])

        target_boxes_sampled = torch.cat(matched_gt_boxes, dim=0)[sampled_pos_inds]

        if sampled_pos_inds.numel() > 0:
            losses["reg"] = self.regressor.compute_loss(
                pred_boxes_sampled,
                target_boxes_sampled,
                ) / max(1, sampled_pos_inds.numel())

        losses["cls"] = self.classifier.compute_loss(
            box_logits[sampled_inds], target_labels[sampled_inds])
        return losses, sampled_pos_inds, sampled_neg_inds


class DetectionHeadHNMNativeRegAll(DetectionHeadHNM):
    def compute_loss(self,
                     prediction: Dict[str, Tensor],
                     target_labels: List[Tensor],
                     matched_gt_boxes: List[Tensor],
                     anchors: List[Tensor],
                     ) -> Tuple[Dict[str, Tensor], torch.Tensor, torch.Tensor]:
        """
        计算回归和分类损失
        N个锚框覆盖所有图像；每张图像M个锚框 => sum(M) = N

        该头部从网络中解码相对偏移，并直接在边界框上计算回归损失（例如用于GIoU损失）

        参数：
            prediction: 用于损失计算的检测预测结果
                box_logits (Tensor): 每个锚框的分类logits
                    [N, num_classes]
                box_deltas (Tensor): 每个锚框的偏移
                    (x1, y1, x2, y2, (z1, z2))[N, dim * 2]
            target_labels (List[Tensor]): 每个锚框的目标标签
                （每张图像）[M]
            matched_gt_boxes: 每个锚框匹配的gt框
                List[[N, dim *  2]], N=每张图像的锚框数量
            anchors: 每张图像的锚框 List[[N, dim *  2]]

        返回：
            Tensor: 包含损失的字典（reg为回归损失，cls为
                分类损失）
            Tensor: 采样的正样本锚框索引（连接后）
            Tensor: 采样的负样本锚框索引（连接后）
        """
        box_logits, box_deltas = prediction["box_logits"], prediction["box_deltas"]

        losses = {}
        sampled_pos_inds, sampled_neg_inds = self.select_indices(target_labels, box_logits)
        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)

        target_labels = torch.cat(target_labels, dim=0)
        batch_anchors = torch.cat(anchors, dim=0)

        assert len(batch_anchors) == len(box_deltas)
        assert len(batch_anchors) == len(box_logits)
        assert len(batch_anchors) == len(target_labels)

        losses["cls"] = self.classifier.compute_loss(
            box_logits[sampled_inds], target_labels[sampled_inds])

        pos_inds = torch.where(target_labels >= 1)[0]
        pred_boxes = self.coder.decode_single(box_deltas[pos_inds], batch_anchors[pos_inds])
        target_boxes = torch.cat(matched_gt_boxes, dim=0)[pos_inds]

        if pos_inds.numel() > 0:
            losses["reg"] = self.regressor.compute_loss(
                pred_boxes,
                target_boxes,
                ) / max(1, pos_inds.numel())

        return losses, sampled_pos_inds, sampled_neg_inds


class DetectionHeadHNMRegAll(DetectionHeadHNM):
    def compute_loss(self,
                     prediction: Dict[str, Tensor],
                     target_labels: List[Tensor],
                     matched_gt_boxes: List[Tensor],
                     anchors: List[Tensor],
                     ) -> Tuple[Dict[str, Tensor], torch.Tensor, torch.Tensor]:
        """
        计算回归和分类损失
        N个锚框覆盖所有图像；每张图像M个锚框 => sum(M) = N

        参数：
            prediction: 用于损失计算的检测预测结果
                box_logits (Tensor): 每个锚框的分类logits
                    [N, num_classes]
                box_deltas (Tensor): 每个锚框的偏移
                    (x1, y1, x2, y2, (z1, z2))[N, dim * 2]
            target_labels (List[Tensor]): 每个锚框的目标标签
                （每张图像）[M]
            matched_gt_boxes: 每个锚框匹配的gt框
                List[[N, dim *  2]], N=每张图像的锚框数量
            anchors: 每张图像的锚框 List[[N, dim *  2]]

        返回：
            Tensor: 包含损失的字典（reg为回归损失，cls
                为分类损失）
            Tensor: 采样的正样本锚框索引（连接后）
            Tensor: 采样的负样本锚框索引（连接后）
        """
        box_logits, box_deltas = prediction["box_logits"], prediction["box_deltas"]

        losses = {}
        sampled_pos_inds, sampled_neg_inds = self.select_indices(target_labels, box_logits)
        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)
        target_labels = torch.cat(target_labels, dim=0)

        losses["cls"] = self.classifier.compute_loss(
            box_logits[sampled_inds], target_labels[sampled_inds])

        pos_inds = torch.where(target_labels >= 1)[0]
        with torch.no_grad():
            batch_matched_gt_boxes = torch.cat(matched_gt_boxes, dim=0)
            batch_anchors = torch.cat(anchors, dim=0)
            target_deltas_sampled = self.coder.encode_single(
                batch_matched_gt_boxes[pos_inds], batch_anchors[pos_inds],
            )

        assert len(batch_anchors) == len(batch_matched_gt_boxes)
        assert len(batch_anchors) == len(box_deltas)
        assert len(batch_anchors) == len(box_logits)
        assert len(batch_anchors) == len(target_labels)

        if pos_inds.numel() > 0:
            losses["reg"] = self.regressor.compute_loss(
                box_deltas[pos_inds],
                target_deltas_sampled,
                ) / max(1, pos_inds.numel())

        return losses, sampled_pos_inds, sampled_neg_inds


HeadType = TypeVar('HeadType', bound=AbstractHead)
