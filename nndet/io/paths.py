"""
Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import numpy as np

from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

Pathlike = Union[Path, str]


def subfiles(dir_path: Path, identifier: str, join: bool) -> List[str]:
    """
    获取所有路径

    参数:
        dir_path: 目录路径
        join: 是否返回dir_path+file_name而不仅仅是file_name
        identifier: 用于选择文件的正则表达式

    返回:
        List[str]: 找到的路径/文件名
    """
    paths = list(map(str, list(Path(os.path.expandvars(dir_path)).glob(identifier))))
    if not join:
        paths = [p.rsplit(os.path.sep, 1)[-1] for p in paths]
    return paths


def get_paths_raw_to_split(data_dir: Path, output_dir: Path,
                           subdirs: tuple = ("imagesTr", "imagesTs")) -> Tuple[
        List[Path], List[Path]]:
    """
    在子目录中搜索所有需要分割的*.nii.gz文件，
    并创建所有文件的源路径和目标路径列表
    （目标路径保留输出目录内的子文件夹）

    参数:
        data_dir (str): 数据所在的顶级目录
        output_dir (str): 分割数据的输出目录
        subdirs (Tuple[str]): 应该搜索数据的子目录

    返回:
        List[Path]: 源目录子文件夹中所有nii文件的路径
        List[Path]: 相应目标目录的路径
    """
    source_files, target_dirs = [], []

    for subdir in subdirs:
        sub_output_dir = output_dir / subdir
        if not sub_output_dir.is_dir():
            sub_output_dir.mkdir(parents=True)

        sub_data_dir = data_dir / subdir
        nii_files = list(sub_data_dir.glob('*.nii.gz'))
        nii_files = list(filter(lambda x: not x.name.startswith('.'), nii_files))
        nii_files.sort()
        for n in nii_files:
            source_files.append(n)
            target_dirs.append(sub_output_dir)
    return source_files, target_dirs


def get_paths_from_splitted_dir(
    num_modalities: int,
    splitted_4d_output_dir: Path,
    test: bool = False,
    labels: bool = True,
    remove_ids: Optional[Sequence[str]] = None,
    ) -> List[List[Path]]:
    """
    创建指向分割数据目录内所有病例（数据和标签；标签在最后位置）的列表

    参数:
        num_modalities (int): 模态数量
        splitted_4d_output_dir (Path): 4d分割数据所在的目录路径
        test: 是否从测试数据获取路径（如果为False，则搜索训练数据）
        labels: 是否在每个病例的最后位置添加标签路径
        remove_ids: 应从列表中删除的病例ID。如果为None，
            则不删除任何病例ID

    返回:
        List[List[Path]]: 所有分割文件的路径；
            每个病例包含其数据文件，标签文件在最后
    """
    data_subdir = "imagesTs" if test else "imagesTr"
    labels_subdir = "labelsTs" if test else "labelsTr"
    training_ids = get_case_ids_from_dir(
        splitted_4d_output_dir / data_subdir,
        unique=True,
        remove_modality=True,
        join=False,
    )
    if remove_ids is not None:
        training_ids = [i for i in training_ids if i not in remove_ids]

    result = []
    for t in training_ids:
        modalities = []
        for m in range(num_modalities):
            modalities.append(splitted_4d_output_dir / data_subdir / f"{t}_{m:04d}.nii.gz")
        if labels:
            modalities.append(
                splitted_4d_output_dir / labels_subdir / f"{t}.nii.gz")
        result.append(modalities)
    return result


def get_case_ids_from_dir(dir_path: Path, unique: bool = True,
                          remove_modality: bool = True, join: bool = False,
                          pattern="*.nii.gz") -> List[str]:
    """
    从目录中提取所有病例ID

    参数:
        dir_path: 包含文件的目录
        unique: 是否去除重复ID
        remove_modality: 是否从文件名中删除模态（文件名格式为 case_0000.nii.gz）
        join: 是否包含目录在结果中
        pattern: 用于选择文件的模式

    返回:
        List[str]: 病例ID列表
    """
    files = subfiles(dir_path, pattern, join=join)
    list_of_case_ids = [get_case_id_from_file(os.path.basename(i),
                                             remove_modality=remove_modality)
                       for i in files]
    if unique:
        list_of_case_ids = np.unique(list_of_case_ids).tolist()
    return list_of_case_ids


def get_case_id_from_path(file_path: Pathlike, remove_modality: bool = True) -> str:
    """
    从文件路径中提取病例ID

    参数:
        file_path: 文件路径
        remove_modality: 是否删除模态（例如从case_0000中删除_0000）

    返回:
        str: 病例ID
    """
    if isinstance(file_path, Path):
        file_path = str(file_path)
    file = os.path.basename(file_path)
    return get_case_id_from_file(file, remove_modality=remove_modality)


def get_case_id_from_file(file_name: str, remove_modality: bool = True) -> str:
    """
    从文件名中提取病例ID

    参数:
        file_name: 文件名
        remove_modality: 是否删除模态（例如从case_0000中删除_0000）

    返回:
        str: 病例ID
    """
    if remove_modality:
        case_id = file_name.split(".")
        return "_".join(case_id[0].rsplit("_", 1)[:-1])
    else:
        case_id = file_name.split(".")
        return case_id[0]


def get_task(task_id: str, name: bool = False, models: bool = False) -> Union[Path, str]:
    """
    根据任务ID获取任务名称或路径

    参数:
        task_id: 任务的ID
        name: 是否返回任务名称而不是路径
        models: 是否返回模型目录而不是训练目录

    返回:
        Union[Path, str]: 任务路径或名称
    
    异常:
        KeyError: 如果任务ID未知
    """
    if models:
        t = os.getenv("det_models")
    else:
        t = os.getenv("det_data")
    if t is None:
        raise ValueError("Framework not configured correctly! "
                         "Please set `det_data` and `det_models` as environment variables!")
    det_data = Path(t)
    all_tasks = [d.stem for d in det_data.iterdir() if d.is_dir() and "Task" in d.name]

    if str(task_id).startswith("Task"):
        task_id = str(task_id)[4:]
    all_tasks = [tn[4:] for tn in all_tasks]

    task_options_exact = [d for d in all_tasks if str(task_id) in d]
    task_number_id = [tn for tn in all_tasks if tn.split('_', 1)[0] == str(task_id)]
    task_name_id = [tn for tn in all_tasks if len(tn.split('_', 1)) > 1 and tn.split('_', 1)[1] == str(task_id)]

    if len(task_options_exact) == 1:
        result = det_data / f"Task{task_options_exact[0]}"
    elif len(task_number_id) == 1:
        result = det_data / f"Task{task_number_id[0]}"
    elif len(task_name_id) == 1:
        result = det_data / f"Task{task_name_id[0]}"
    else:
        raise KeyError(f"Unknown Task id {task_id}")
    
    if name:
        result = result.stem
    return result


def get_training_dir(model_dir: Pathlike, fold: int) -> Path:
    """
    Find training dir from a specific model dir

    Args:
        model_dir: path to model dir e.g. ../Task12_LIDC/RetinaUNetV0
        fold: fold to look for. if -1 look for consolidated dir

    Returns:
        Path: path to training dir
    """
    model_dir = Path(model_dir)
    identifier = f"fold{fold}" if fold != -1 else "consolidated"
    candidates = [p for p in model_dir.iterdir() if p.is_dir() and identifier in p.stem]
    if len(candidates) == 1:
        return candidates[0]
    else:
        raise ValueError(f"Found wrong number of training dirs {candidates} in {model_dir}")
