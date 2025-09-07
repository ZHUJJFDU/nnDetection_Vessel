from __future__ import annotations

import os
from time import time
from typing import Any, Callable, Dict, Optional, Sequence, Hashable, Type, TypeVar

import torch
import pytorch_lightning as pl
from pytorch_lightning.core.memory import ModelSummary
from loguru import logger

from nndet.io.load import save_txt
from nndet.inference.predictor import Predictor


class LightningBaseModule(pl.LightningModule):
    def __init__(self,
                 model_cfg: dict,
                 trainer_cfg: dict,
                 plan: dict,
                 **kwargs
                 ):
        """
        提供在nnDetection内部使用的基础模块。
        所有nnDetection的lightning模块都应该从这个类派生！

        参数:
            model_cfg: 模型配置。查看 :method:`from_config_plan` 获取更多信息
            trainer_cfg: 训练器信息
            plan: 包含从规划阶段导出的参数
        """
        super().__init__()
        self.model_cfg = model_cfg  # 存储模型配置
        self.trainer_cfg = trainer_cfg  # 存储训练器配置
        self.plan = plan  # 存储规划参数

        # 从配置和规划参数创建模型
        self.model = self.from_config_plan(
            model_cfg=self.model_cfg,
            plan_arch=self.plan["architecture"],
            plan_anchors=self.plan["anchors"],
        )

        # 设置示例输入数组形状，用于模型摘要和检查
        self.example_input_array_shape = (
            1, plan["architecture"]["in_channels"], *plan["patch_size"],
            )

        # 用于记录每个训练周期的时间
        self.epoch_start_tic = 0
        self.epoch_end_toc = 0

    @property
    def max_epochs(self):
        """
        训练的总周期数
        """
        return self.trainer_cfg["max_num_epochs"]

    def on_epoch_start(self) -> None:
        """
        记录周期开始时间
        """
        self.epoch_start_tic = time()
        return super().on_epoch_start()
    
    def validation_epoch_end(self, validation_step_outputs):
        """
        打印周期耗时
        (当在集群上禁用进度条时需要)
        """
        self.epoch_end_toc = time()
        logger.info(f"This epoch took {int(self.epoch_end_toc - self.epoch_start_tic)} s")
        return super().validation_epoch_end(validation_step_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        用于生成模型摘要
        不要(!)用于推理。这只会将输入通过网络前向传播，
        不包括检测特定的后处理！
        """
        return self.model(x)

    @property
    def example_input_array(self):
        """
        创建示例输入张量
        """
        return torch.zeros(*self.example_input_array_shape)

    def summarize(self, *args, **kwargs) -> Optional[ModelSummary]:
        """
        将模型摘要保存为txt文件
        """
        summary = super().summarize(*args, **kwargs)
        save_txt(summary, "./network")
        return summary

    def inference_step(self, batch: Any, **kwargs) -> Dict[str, Any]:
        """
        nnDetection预测器类使用的预测方法
        """
        return self.model.inference_step(batch, **kwargs)

    @classmethod
    def from_config_plan(cls,
                         model_cfg: dict,
                         plan_arch: dict,
                         plan_anchors: dict,
                         log_num_anchors: str = None,
                         **kwargs,
                         ):
        """
        用于生成模型
        需要在子类中实现
        """
        raise NotImplementedError

    @staticmethod
    def get_ensembler_cls(key: Hashable, dim: int) -> Callable:
        """
        获取用于组合多个预测结果的集成器类
        需要在子类中重写！
        """
        raise NotImplementedError

    @classmethod
    def get_predictor(cls,
                      plan: Dict,
                      models: Sequence[LightningBaseModule],
                      num_tta_transforms: int = None,
                      **kwargs
                      ) -> Type[Predictor]:
        """
        获取预测器
        需要在子类中重写！
        """
        raise NotImplementedError

    def sweep(self,
              cfg: dict,
              save_dir: os.PathLike,
              train_data_dir: os.PathLike,
              case_ids: Sequence[str],
              run_prediction: bool = True,
              ) -> Dict[str, Any]:
        """
        扫描参数以找到最佳预测结果
        需要在子类中重写！

        参数:
            cfg: 用于训练的配置
            save_dir: 用于训练的保存目录
            train_data_dir: 存放预处理训练/验证数据的目录
            case_ids: 要准备和预测的病例标识符
            run_prediction: 是否预测病例
            **kwargs: 传递给预测函数的关键字参数
        """
        raise NotImplementedError


class LightningBaseModuleSWA(LightningBaseModule):
    """
    支持随机权重平均(SWA)的Lightning基础模块
    SWA是一种通过平均训练后期权重来提高泛化能力的技术
    """
    @property
    def max_epochs(self):
        """
        训练的总周期数，包括SWA周期
        """
        return self.trainer_cfg["max_num_epochs"] + self.trainer_cfg["swa_epochs"]

    def configure_callbacks(self):
        """
        配置SWA回调，实现循环线性学习率调度
        """
        from nndet.training.swa import SWACycleLinear

        callbacks = []
        callbacks.append(
            SWACycleLinear(
                swa_epoch_start=self.trainer_cfg["max_num_epochs"],  # SWA开始的周期
                cycle_initial_lr=self.trainer_cfg["initial_lr"] / 10.,  # 循环初始学习率
                cycle_final_lr=self.trainer_cfg["initial_lr"] / 1000.,  # 循环结束学习率
                num_iterations_per_epoch=self.trainer_cfg["num_train_batches_per_epoch"],  # 每周期迭代次数
                )
            )
        return callbacks


# 定义绑定到LightningBaseModule的类型变量，用于类型注解
LightningBaseModuleType = TypeVar('LightningBaseModuleType', bound=LightningBaseModule)
