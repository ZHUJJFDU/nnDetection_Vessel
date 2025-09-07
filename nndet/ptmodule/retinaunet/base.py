from __future__ import annotations

import os
import copy
from collections import defaultdict
from pathlib import Path
from functools import partial
from typing import Callable, Hashable, Sequence, Dict, Any, Type

import torch
import numpy as np
from loguru import logger
from torchvision.models.detection.rpn import AnchorGenerator

from nndet.utils.tensor import to_numpy
from nndet.evaluator.det import BoxEvaluator
from nndet.evaluator.seg import SegmentationEvaluator

from nndet.core.retina import BaseRetinaNet
from nndet.core.boxes.matcher import IoUMatcher
from nndet.core.boxes.sampler import HardNegativeSamplerBatched
from nndet.core.boxes.coder import CoderType, BoxCoderND
from nndet.core.boxes.anchors import get_anchor_generator
from nndet.core.boxes.ops import box_iou
from nndet.core.boxes.anchors import AnchorGeneratorType

from nndet.ptmodule.base_module import LightningBaseModuleSWA, LightningBaseModule

from nndet.arch.conv import Generator, ConvInstanceRelu, ConvGroupRelu
from nndet.arch.blocks.basic import StackedConvBlock2
from nndet.arch.encoder.abstract import EncoderType
from nndet.arch.encoder.modular import Encoder
from nndet.arch.decoder.base import DecoderType, BaseUFPN, UFPNModular
from nndet.arch.heads.classifier import ClassifierType, CEClassifier
from nndet.arch.heads.regressor import RegressorType, L1Regressor
from nndet.arch.heads.comb import HeadType, DetectionHeadHNM
from nndet.arch.heads.segmenter import SegmenterType, DiCESegmenter

from nndet.training.optimizer import get_params_no_wd_on_norm
from nndet.training.learning_rate import LinearWarmupPolyLR

from nndet.inference.predictor import Predictor
from nndet.inference.sweeper import BoxSweeper
from nndet.inference.transforms import get_tta_transforms, Inference2D
from nndet.inference.loading import get_loader_fn
from nndet.inference.helper import predict_dir
from nndet.inference.ensembler.segmentation import SegmentationEnsembler
from nndet.inference.ensembler.detection import BoxEnsemblerSelective

from nndet.io.transforms import (
    Compose,
    Instances2Boxes,
    Instances2Segmentation,
    FindInstances,
    )


class RetinaUNetModule(LightningBaseModuleSWA):
    base_conv_cls = ConvInstanceRelu
    head_conv_cls = ConvGroupRelu
    block = StackedConvBlock2
    encoder_cls = Encoder
    decoder_cls = UFPNModular
    matcher_cls = IoUMatcher
    head_cls = DetectionHeadHNM
    head_classifier_cls = CEClassifier
    head_regressor_cls = L1Regressor
    head_sampler_cls = HardNegativeSamplerBatched
    segmenter_cls = DiCESegmenter

    def __init__(self,
                 model_cfg: dict,
                 trainer_cfg: dict,
                 plan: dict,
                 **kwargs
                 ):
        """
        RetinaUNet Lightning模块骨架
        
        参数:
            model_cfg: 模型配置。查看:method:`from_config_plan`
                获取更多信息
            trainer_cfg: 训练器信息
            plan: 包含从规划阶段获得的参数
        """
        super().__init__(
            model_cfg=model_cfg,
            trainer_cfg=trainer_cfg,
            plan=plan,
        )

        _classes = [f"class{c}" for c in range(plan["architecture"]["classifier_classes"])]
        self.box_evaluator = BoxEvaluator.create(
            classes=_classes,
            fast=True,
            save_dir=None,
            )
        self.seg_evaluator = SegmentationEvaluator.create()

        self.pre_trafo = Compose(
            FindInstances(
                instance_key="target",
                save_key="present_instances",
                ),
            Instances2Boxes(
                instance_key="target",
                map_key="instance_mapping",
                box_key="boxes",
                class_key="classes",
                present_instances="present_instances",
                ),
            Instances2Segmentation(
                instance_key="target",
                map_key="instance_mapping",
                present_instances="present_instances",
                )
            )

        self.eval_score_key = "mAP_IoU_0.10_0.50_0.05_MaxDet_100"

    def training_step(self, batch, batch_idx):
        """
        计算单个训练步骤
        有关更多信息，请参见:class:`BaseRetinaNet`
        """
        with torch.no_grad():
            batch = self.pre_trafo(**batch)

        losses, _ = self.model.train_step(
            images=batch["data"],
            targets={
                "target_boxes": batch["boxes"],
                "target_classes": batch["classes"],
                "target_seg": batch['target'][:, 0]  # 移除通道维度
                },
            evaluation=False,
            batch_num=batch_idx,
        )
        loss = sum(losses.values())
        return {"loss": loss, **{key: l.detach().item() for key, l in losses.items()}}

    def validation_step(self, batch, batch_idx):
        """
        计算单个验证步骤（与训练步骤相同，但有额外的预测处理）
        有关更多信息，请参见:class:`BaseRetinaNet`
        """
        with torch.no_grad():
            batch = self.pre_trafo(**batch)
            targets = {
                    "target_boxes": batch["boxes"],
                    "target_classes": batch["classes"],
                    "target_seg": batch['target'][:, 0]  # 移除通道维度
                }
            losses, prediction = self.model.train_step(
                images=batch["data"],
                targets=targets,
                evaluation=True,
                batch_num=batch_idx,
            )
            loss = sum(losses.values())

        self.evaluation_step(prediction=prediction, targets=targets)
        return {"loss": loss.detach().item(),
                **{key: l.detach().item() for key, l in losses.items()}}

    def evaluation_step(
        self,
        prediction: dict,
        targets: dict,
    ):
        """
        执行评估步骤，将预测和真实值添加到缓存机制中，以便在epoch结束时进行评估

        参数:
            prediction: 从模型获得的预测
                'pred_boxes': List[Tensor]: 每张图像的预测边界框
                    List[[R, dim * 2]]
                'pred_scores': List[Tensor]: 类别的预测概率
                    List[[R]]
                'pred_labels': List[Tensor]: 预测类别 List[[R]]
                'pred_seg': Tensor: 预测分割 [N, dims]
            targets: 真实值
                `target_boxes` (List[Tensor]): 真实边界框
                    (x1, y1, x2, y2, (z1, z2))[X, dim * 2], X=图像中真实框的数量
                `target_classes` (List[Tensor]): 每个框的真实类别
                    (类别从0开始) [X], X=图像中真实框的数量
                `target_seg` (Tensor): 分割真实值（如果在输入字典中找到seg）
        """
        pred_boxes = to_numpy(prediction["pred_boxes"])
        pred_classes = to_numpy(prediction["pred_labels"])
        pred_scores = to_numpy(prediction["pred_scores"])

        gt_boxes = to_numpy(targets["target_boxes"])
        gt_classes = to_numpy(targets["target_classes"])
        gt_ignore = None

        self.box_evaluator.run_online_evaluation(
            pred_boxes=pred_boxes,
            pred_classes=pred_classes,
            pred_scores=pred_scores,
            gt_boxes=gt_boxes,
            gt_classes=gt_classes,
            gt_ignore=gt_ignore,
            )

        pred_seg = to_numpy(prediction["pred_seg"])
        gt_seg = to_numpy(targets["target_seg"])

        self.seg_evaluator.run_online_evaluation(
            seg_probs=pred_seg,
            target=gt_seg,
            )

    def training_epoch_end(self, training_step_outputs):
        """
        将训练损失记录到loguru记录器
        """
        # 处理并记录损失
        vals = defaultdict(list)
        for _val in training_step_outputs:
            for _k, _v in _val.items():
                if _k == "loss":
                    vals[_k].append(_v.detach().item())
                else:
                    vals[_k].append(_v)

        for _key, _vals in vals.items():
            mean_val = np.mean(_vals)
            if _key == "loss":
                logger.info(f"训练损失达到: {mean_val:0.5f}")
            self.log(f"train_{_key}", mean_val, sync_dist=True)
        return super().training_epoch_end(training_step_outputs)

    def validation_epoch_end(self, validation_step_outputs):
        """
        将验证损失记录到loguru记录器
        """
        # 处理并记录损失
        vals = defaultdict(list)
        for _val in validation_step_outputs:
            for _k, _v in _val.items():
                vals[_k].append(_v)

        for _key, _vals in vals.items():
            mean_val = np.mean(_vals)
            if _key == "loss":
                logger.info(f"验证损失达到: {mean_val:0.5f}")
            self.log(f"val_{_key}", mean_val, sync_dist=True)

        # 处理并记录指标
        self.evaluation_end()
        return super().validation_epoch_end(validation_step_outputs)

    def evaluation_end(self):
        """
        使用`evaluation_step`中的缓存值执行epoch的评估
        """
        metric_scores, _ = self.box_evaluator.finish_online_evaluation()
        self.box_evaluator.reset()

        logger.info(f"mAP@0.1:0.5:0.05: {metric_scores['mAP_IoU_0.10_0.50_0.05_MaxDet_100']:0.3f}  "
                    f"AP@0.1: {metric_scores['AP_IoU_0.10_MaxDet_100']:0.3f}  "
                    f"AP@0.5: {metric_scores['AP_IoU_0.50_MaxDet_100']:0.3f}")

        seg_scores, _ = self.seg_evaluator.finish_online_evaluation()
        self.seg_evaluator.reset()
        metric_scores.update(seg_scores)

        logger.info(f"代理前景Dice系数: {seg_scores['seg_dice']:0.3f}")

        for key, item in metric_scores.items():
            self.log(f'{key}', item, on_step=None, on_epoch=True, prog_bar=False, logger=True)

    def configure_optimizers(self):
        """
        配置优化器和调度器
        基本配置是SGD与LinearWarmup和PolyLR学习率调度
        """
        # 配置优化器
        logger.info(f"运行: 初始学习率 {self.trainer_cfg['initial_lr']} "
                    f"权重衰减 {self.trainer_cfg['weight_decay']} "
                    f"SGD 动量为 {self.trainer_cfg['sgd_momentum']} 和 "
                    f"Nesterov {self.trainer_cfg['sgd_nesterov']}")
        wd_groups = get_params_no_wd_on_norm(self, weight_decay=self.trainer_cfg['weight_decay'])
        optimizer = torch.optim.SGD(
            wd_groups,
            self.trainer_cfg["initial_lr"],
            weight_decay=self.trainer_cfg["weight_decay"],
            momentum=self.trainer_cfg["sgd_momentum"],
            nesterov=self.trainer_cfg["sgd_nesterov"],
            )

        # 配置学习率调度器
        num_iterations = self.trainer_cfg["max_num_epochs"] * \
            self.trainer_cfg["num_train_batches_per_epoch"]
        scheduler = LinearWarmupPolyLR(
            optimizer=optimizer,
            warm_iterations=self.trainer_cfg["warm_iterations"],
            warm_lr=self.trainer_cfg["warm_lr"],
            poly_gamma=self.trainer_cfg["poly_gamma"],
            num_iterations=num_iterations
        )
        return [optimizer], {'scheduler': scheduler, 'interval': 'step'}

    @classmethod
    def from_config_plan(cls,
                         model_cfg: dict,
                         plan_arch: dict,
                         plan_anchors: dict,
                         log_num_anchors: str = None,
                         **kwargs,
                         ):
        """
        创建可配置的RetinaUNet

        参数:
            model_cfg: 模型配置
                查看示例配置获取更多信息
            plan_arch: 计划架构
                `dim` (int): 空间维度的数量
                `in_channels` (int): 输入通道数
                `classifier_classes` (int): 类别数量
                `seg_classes` (int): 分割类别数量
                `start_channels` (int): 编码器中的起始通道数
                `fpn_channels` (int): 用于FPN的通道数
                `head_channels` (int): 用于头部的通道数
                `decoder_levels` (int): 用于检测的解码器级别
            plan_anchors: 锚点的参数（参见
                :class:`AnchorGenerator` 获取更多信息）
                    `stride`: 步长
                    `aspect_ratios`: 宽高比
                    `sizes`: 2D锚点的大小
                    (`zsizes`: 3D的额外z尺寸)
            log_num_anchors: 要使用的日志记录器名称；如果为None，则不进行记录
            **kwargs:
        """
        logger.info(f"架构覆盖: {model_cfg['plan_arch_overwrites']} "
                    f"锚点覆盖: {model_cfg['plan_anchors_overwrites']}")
        logger.info(f"根据{plan_arch.get('arch_name', 'not_found')}的计划构建架构")
        plan_arch.update(model_cfg["plan_arch_overwrites"])
        plan_anchors.update(model_cfg["plan_anchors_overwrites"])
        logger.info(f"起始通道数: {plan_arch['start_channels']}; "
                    f"头部通道数: {plan_arch['head_channels']}; "
                    f"FPN通道数: {plan_arch['fpn_channels']}")

        _plan_anchors = copy.deepcopy(plan_anchors)
        coder = BoxCoderND(weights=(1.,) * (plan_arch["dim"] * 2))
        s_param = False if ("aspect_ratios" in _plan_anchors) and \
                           (_plan_anchors["aspect_ratios"] is not None) else True
        anchor_generator = get_anchor_generator(
            plan_arch["dim"], s_param=s_param)(**_plan_anchors)

        encoder = cls._build_encoder(
            plan_arch=plan_arch,
            model_cfg=model_cfg,
            )
        decoder = cls._build_decoder(
            encoder=encoder,
            plan_arch=plan_arch,
            model_cfg=model_cfg,
            )
        matcher = cls.matcher_cls(
            similarity_fn=box_iou,
            **model_cfg["matcher_kwargs"],
            )

        classifier = cls._build_head_classifier(
            plan_arch=plan_arch,
            model_cfg=model_cfg,
            anchor_generator=anchor_generator,
        )
        regressor = cls._build_head_regressor(
            plan_arch=plan_arch,
            model_cfg=model_cfg,
            anchor_generator=anchor_generator,
        )
        head = cls._build_head(
            plan_arch=plan_arch,
            model_cfg=model_cfg,
            classifier=classifier,
            regressor=regressor,
            coder=coder
        )
        segmenter = cls._build_segmenter(
            plan_arch=plan_arch,
            model_cfg=model_cfg,
            decoder=decoder,
        )

        detections_per_img = plan_arch.get("detections_per_img", 100)
        score_thresh = plan_arch.get("score_thresh", 0)
        topk_candidates = plan_arch.get("topk_candidates", 10000)
        remove_small_boxes = plan_arch.get("remove_small_boxes", 0.01)
        nms_thresh = plan_arch.get("nms_thresh", 0.6)

        logger.info(f"模型推理摘要: \n"
                    f"每张图像的检测数: {detections_per_img} \n"
                    f"分数阈值: {score_thresh} \n"
                    f"topk候选数: {topk_candidates} \n"
                    f"移除小框阈值: {remove_small_boxes} \n"
                    f"NMS阈值: {nms_thresh}",
                    )

        return BaseRetinaNet(
            dim=plan_arch["dim"],
            encoder=encoder,
            decoder=decoder,
            head=head,
            anchor_generator=anchor_generator,
            matcher=matcher,
            num_classes=plan_arch["classifier_classes"],
            decoder_levels=plan_arch["decoder_levels"],
            segmenter=segmenter,
            # model_max_instances_per_batch_element (在mdt中每张图像，每个类别；这里是每张图像)
            detections_per_img=detections_per_img,
            score_thresh=score_thresh,
            topk_candidates=topk_candidates,
            remove_small_boxes=remove_small_boxes,
            nms_thresh=nms_thresh,
        )

    @classmethod
    def _build_encoder(
        cls,
        plan_arch: dict,
        model_cfg: dict,
    ) -> EncoderType:
        """
        构建编码器网络

        参数:
            plan_arch: 架构设置
            model_cfg: 附加架构设置

        返回:
            EncoderType: 编码器实例
        """
        conv = Generator(cls.base_conv_cls, plan_arch["dim"])
        logger.info(f"构建:: 编码器 {cls.encoder_cls.__name__}: {model_cfg['encoder_kwargs']} ")
        encoder = cls.encoder_cls(
            conv=conv,
            conv_kernels=plan_arch["conv_kernels"],
            strides=plan_arch["strides"],
            block_cls=cls.block,
            in_channels=plan_arch["in_channels"],
            start_channels=plan_arch["start_channels"],
            stage_kwargs=None,
            max_channels=plan_arch.get("max_channels", 320),
            **model_cfg['encoder_kwargs'],
        )
        return encoder

    @classmethod
    def _build_decoder(
        cls,
        plan_arch: dict,
        model_cfg: dict,
        encoder: EncoderType,
    ) -> DecoderType:
        """
        构建解码器网络

        参数:
            plan_arch: 架构设置
            model_cfg: 附加架构设置

        返回:
            DecoderType: 解码器实例
        """
        conv = Generator(cls.base_conv_cls, plan_arch["dim"])
        logger.info(f"构建:: 解码器 {cls.decoder_cls.__name__}: {model_cfg['decoder_kwargs']}")
        decoder = cls.decoder_cls(
            conv=conv,
            conv_kernels=plan_arch["conv_kernels"],
            strides=encoder.get_strides(),
            in_channels=encoder.get_channels(),
            decoder_levels=plan_arch["decoder_levels"],
            fixed_out_channels=plan_arch["fpn_channels"],
            **model_cfg['decoder_kwargs'],
        )
        return decoder

    @classmethod
    def _build_head_classifier(
        cls,
        plan_arch: dict,
        model_cfg: dict,
        anchor_generator: AnchorGeneratorType,
    ) -> ClassifierType:
        """
        构建检测头的分类子网络

        参数:
            anchor_generator: 锚点生成器实例
            plan_arch: 架构设置
            model_cfg: 附加架构设置

        返回:
            ClassifierType: 分类实例
        """
        conv = Generator(cls.head_conv_cls, plan_arch["dim"])
        name = cls.head_classifier_cls.__name__
        kwargs = model_cfg['head_classifier_kwargs']

        logger.info(f"构建:: 分类器 {name}: {kwargs}")
        classifier = cls.head_classifier_cls(
            conv=conv,
            in_channels=plan_arch["fpn_channels"],
            internal_channels=plan_arch["head_channels"],
            num_classes=plan_arch["classifier_classes"],
            anchors_per_pos=anchor_generator.num_anchors_per_location()[0],
            num_levels=len(plan_arch["decoder_levels"]),
            **kwargs,
        )
        return classifier

    @classmethod
    def _build_head_regressor(
        cls,
        plan_arch: dict,
        model_cfg: dict,
        anchor_generator: AnchorGeneratorType,
    ) -> RegressorType:
        """
        构建检测头的回归子网络

        参数:
            plan_arch: 架构设置
            model_cfg: 附加架构设置
            anchor_generator: 锚点生成器实例

        返回:
            RegressorType: 分类实例
        """
        conv = Generator(cls.head_conv_cls, plan_arch["dim"])
        name = cls.head_regressor_cls.__name__
        kwargs = model_cfg['head_regressor_kwargs']

        logger.info(f"构建:: 回归器 {name}: {kwargs}")
        regressor = cls.head_regressor_cls(
            conv=conv,
            in_channels=plan_arch["fpn_channels"],
            internal_channels=plan_arch["head_channels"],
            anchors_per_pos=anchor_generator.num_anchors_per_location()[0],
            num_levels=len(plan_arch["decoder_levels"]),
            **kwargs,
        )
        return regressor

    @classmethod
    def _build_head(
        cls,
        plan_arch: dict,
        model_cfg: dict,
        classifier: ClassifierType,
        regressor: RegressorType,
        coder: CoderType,
    ) -> HeadType:
        """
        构建检测头

        参数:
            plan_arch: 架构设置
            model_cfg: 附加架构设置
            classifier: 分类器实例
            regressor: 回归器实例
            coder: 用于编码框的编码器实例

        返回:
            HeadType: 实例化的头部
        """
        head_name = cls.head_cls.__name__
        head_kwargs = model_cfg['head_kwargs']
        sampler_name = cls.head_sampler_cls.__name__
        sampler_kwargs = model_cfg['head_sampler_kwargs']

        logger.info(f"构建:: 头部 {head_name}: {head_kwargs} "
                    f"采样器 {sampler_name}: {sampler_kwargs}")
        sampler = cls.head_sampler_cls(**sampler_kwargs)
        head = cls.head_cls(
            classifier=classifier,
            regressor=regressor,
            coder=coder,
            sampler=sampler,
            log_num_anchors=None,
            **head_kwargs,
        )
        return head

    @classmethod
    def _build_segmenter(
        cls,
        plan_arch: dict,
        model_cfg: dict,
        decoder: DecoderType,
    ) -> SegmenterType:
        """
        构建分割头部

        参数:
            plan_arch: 架构设置
            model_cfg: 附加架构设置
            decoder: 解码器实例

        返回:
            SegmenterType: 分割头部
        """
        if cls.segmenter_cls is not None:
            name = cls.segmenter_cls.__name__
            kwargs = model_cfg['segmenter_kwargs']
            conv = Generator(cls.base_conv_cls, plan_arch["dim"])

            logger.info(f"构建:: 分割器 {name} {kwargs}")
            segmenter = cls.segmenter_cls(
                conv,
                seg_classes=plan_arch["seg_classes"],
                in_channels=decoder.get_channels(),
                decoder_levels=plan_arch["decoder_levels"],
                **kwargs,
            )
        else:
            segmenter = None
        return segmenter

    @staticmethod
    def get_ensembler_cls(key: Hashable, dim: int) -> Callable:
        """
        获取合并多个预测的合并器类
        需要在子类中重写！
        """
        _lookup = {
            2: {
                "boxes": None,
                "seg": None,
            },
            3: {
                "boxes": BoxEnsemblerSelective,
                "seg": SegmentationEnsembler,
                }
            }
        if dim == 2:
            raise NotImplementedError
        return _lookup[dim][key]

    @classmethod
    def get_predictor(cls,
                      plan: Dict,
                      models: Sequence[RetinaUNetModule],
                      num_tta_transforms: int = None,
                      do_seg: bool = False,
                      **kwargs,
                      ) -> Predictor:
        # 处理计划
        crop_size = plan["patch_size"]
        batch_size = plan["batch_size"]
        inferene_plan = plan.get("inference_plan", {})
        logger.info(f"为预测找到推理计划: {inferene_plan}")
        if num_tta_transforms is None:
            num_tta_transforms = 8 if plan["network_dim"] == 3 else 4

        # 设置
        tta_transforms, tta_inverse_transforms = \
            get_tta_transforms(num_tta_transforms, True)
        logger.info(f"预测使用 {len(tta_transforms)} 个TTA变换（一个虚拟变换）。")

        ensembler = {"boxes": partial(
            cls.get_ensembler_cls(key="boxes", dim=plan["network_dim"]).from_case,
            parameters=inferene_plan,
        )}
        if do_seg:
            ensembler["seg"] = partial(
                cls.get_ensembler_cls(key="seg", dim=plan["network_dim"]).from_case,
            )

        predictor = Predictor(
            ensembler=ensembler,
            models=models,
            crop_size=crop_size,
            tta_transforms=tta_transforms,
            tta_inverse_transforms=tta_inverse_transforms,
            batch_size=batch_size,
            **kwargs,
            )
        if plan["network_dim"] == 2:
            raise NotImplementedError
            predictor.pre_transform = Inference2D(["data"])
        return predictor

    def sweep(self,
              cfg: dict,
              save_dir: os.PathLike,
              train_data_dir: os.PathLike,
              case_ids: Sequence[str],
              run_prediction: bool = True,
              **kwargs,
              ) -> Dict[str, Any]:
        """
        扫描检测参数以找到最佳预测

        参数:
            cfg: 用于训练的配置
            save_dir: 用于训练的保存目录
            train_data_dir: 存放预处理训练/验证数据的目录
            case_ids: 准备和预测的案例标识符
            run_prediction: 预测案例
            **kwargs: 传递给预测函数的关键字参数

        返回:
            Dict: 推理计划
                例如（确切参数取决于用于预测的合并器类）
                `iou_thresh` (float): 最佳IoU阈值
                `score_thresh (float)`: 最佳分数阈值
                `no_overlap` (bool): 启用/禁用类独立NMS（ciNMS）
        """
        logger.info(f"在 {case_ids} 上运行参数扫描")

        train_data_dir = Path(train_data_dir)
        preprocessed_dir = train_data_dir.parent
        processed_eval_labels = preprocessed_dir / "labelsTr"

        _save_dir = save_dir / "sweep"
        _save_dir.mkdir(parents=True, exist_ok=True)

        prediction_dir = save_dir / "sweep_predictions"
        prediction_dir.mkdir(parents=True, exist_ok=True)

        if run_prediction:
            logger.info("使用默认设置预测案例...")
            predictor = predict_dir(
                source_dir=train_data_dir,
                target_dir=prediction_dir,
                cfg=cfg,
                plan=self.plan,
                source_models=save_dir,
                num_models=1,
                num_tta_transforms=None,
                case_ids=case_ids,
                save_state=True,
                model_fn=get_loader_fn(mode=self.trainer_cfg.get("sweep_ckpt", "last")),
                **kwargs,
                )

        logger.info("开始参数扫描...")
        ensembler_cls = self.get_ensembler_cls(key="boxes", dim=self.plan["network_dim"])
        sweeper = BoxSweeper(
            classes=[item for _, item in cfg["data"]["labels"].items()],
            pred_dir=prediction_dir,
            gt_dir=processed_eval_labels,
            target_metric=self.eval_score_key,
            ensembler_cls=ensembler_cls,
            save_dir=_save_dir,
            )
        inference_plan = sweeper.run_postprocessing_sweep()
        return inference_plan
