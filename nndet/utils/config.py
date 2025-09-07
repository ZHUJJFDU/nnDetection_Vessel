import json
import importlib
from pathlib import Path

import yaml
from omegaconf import OmegaConf
from hydra import compose as hydra_compose

from nndet.io.paths import Pathlike, get_task


def load_dataset_info(task_dir: Pathlike) -> dict:
    """
    从给定的任务目录加载数据集信息
    
    参数:
        task_dir: 特定任务目录的路径，例如 ../Task12_LIDC
        
    返回:
        dict: 加载的数据集信息。通常包括:
            `name` (str): 数据集名称
            `target_class` (str): 目标类别
    """
    task_dir = Path(task_dir)
    yaml_path = task_dir / "dataset.yaml"
    yaml_path_fallback = task_dir / "dataset.yml"
    json_path = task_dir / "dataset.json"

    # 尝试按优先顺序加载不同格式的数据集配置文件
    if yaml_path.is_file():
        with open(yaml_path, 'r') as f:
            data = yaml.full_load(f)
    elif yaml_path_fallback.is_file():
        with open(yaml_path_fallback, 'r') as f:
            data = yaml.full_load(f)
    elif json_path.is_file():
        with open(json_path, "r") as f:
            data = json.load(f)
    else:
        raise RuntimeError(f"在 {task_dir} 中未找到 dataset.json 或 dataset.yaml 文件")
    return data


def compose(task, *args, models: bool = False, **kwargs) -> dict:
    """
    为指定任务组合配置。
    
    参数:
        task: 任务名称
        *args: 传递给 hydra compose 的位置参数
        models: 是否加载模型特定的配置
        **kwargs: 传递给 hydra compose 的关键字参数
        
    返回:
        dict: 组合后的配置
    """
    # 使用 hydra 组合基础配置
    cfg = hydra_compose(*args, **kwargs)
    # print(OmegaConf.to_yaml(cfg))
    OmegaConf.set_struct(cfg, False)
    
    # 获取任务名称并设置配置
    task_name = get_task(task, name=True, models=models)
    cfg["task"] = task_name
    
    # 加载数据集信息并添加到配置
    cfg["data"] = load_dataset_info(get_task(task_name))

    # 处理额外导入
    for imp in cfg.get("additional_imports", []):
        print(f"发现额外导入模块 {imp}")
        importlib.import_module(imp)

    return cfg
