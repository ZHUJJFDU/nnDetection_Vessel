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
import pickle
import json
import yaml
import time
from contextlib import contextmanager
from itertools import repeat
from multiprocessing.pool import Pool
from collections import OrderedDict
from pathlib import Path
from typing import Sequence, Any, Tuple, Union
from zipfile import BadZipfile

import numpy as np
import SimpleITK as sitk
from loguru import logger

from nndet.io.paths import subfiles, Pathlike


__all__ = ["load_case_cropped", "load_case_from_list",
           "load_properties_of_cropped", "npy_dataset",
           "load_pickle", "load_json", "save_json", "save_pickle",
           "save_yaml", "load_npz_looped",
           ]


def load_case_from_list(data_files, seg_file=None) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    从文件路径列表加载一个病例的数据和标签

    参数:
        data_files (Sequence[Path]): 数据文件的路径
        seg_file (Path): 分割文件的路径（如果找到一个带有json后缀的第二个文件，
            它将被视为附加属性文件并自动加载）

    返回:
        np.ndarary: 加载的数据（float32类型）[C, X, Y, Z]
        np.ndarray: 加载的分割（如果没有提供分割文件，则为None）
            （float32类型）[1, X, Y, Z]
        dict: 文件的附加属性
            `original_size_of_raw_data`: 数据的原始形状（正确重排序）
            `original_spacing`: 原始间距（正确重排序）
            `list_of_data_files`: 数据文件的路径
            `seg_file`: 标签文件的路径
            `itk_origin`: 世界坐标系中的原点
            `itk_spacing`: 世界坐标系中的间距
            `itk_direction`: 世界坐标系中的方向
    """
    assert isinstance(data_files, Sequence), "case must be sequence"
    properties = OrderedDict()
    data_itk = [sitk.ReadImage(str(f)) for f in data_files]

    properties["original_size_of_raw_data"] = np.array(data_itk[0].GetSize())[[2, 1, 0]]
    properties["original_spacing"] = np.array(data_itk[0].GetSpacing())[[2, 1, 0]]
    properties["list_of_data_files"] = data_files
    properties["seg_file"] = seg_file

    properties["itk_origin"] = data_itk[0].GetOrigin()
    properties["itk_spacing"] = data_itk[0].GetSpacing()
    properties["itk_direction"] = data_itk[0].GetDirection()

    data_npy = np.stack([sitk.GetArrayFromImage(d) for d in data_itk])
    if seg_file is not None:
        seg_itk = sitk.ReadImage(str(seg_file))
        seg_npy = sitk.GetArrayFromImage(seg_itk)[None].astype(np.float32)

        seg_props_file = f"{str(seg_file).split('.')[0]}.json"
        if os.path.isfile(seg_props_file):
            properties_json = load_json(seg_props_file)

            # 将实例转换为正确的类型
            properties_json["instances"] = {
                str(key): int(item) for key, item in properties_json["instances"].items()}

            properties.update(properties_json)
    else:
        seg_npy = None
    return data_npy.astype(np.float32), seg_npy, properties


def load_properties_of_cropped(path: Path):
    """
    加载裁剪后的属性文件

    参数:
        path: 裁剪后数据的属性文件路径
    
    返回:
        属性字典
    """
    if not path.suffix == '.pkl':
        path = Path(str(path) + '.pkl')
    
    with open(path, 'rb') as f:
        properties = pickle.load(f)
    return properties


def load_case_cropped(folder: Path, case_id: str) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    从预处理（裁剪）的文件夹加载病例

    参数:
        folder: 数据集的根目录，需要包含imagesTr和labelsTr文件夹
        case_id: 病例ID

    返回:
        np.ndarray: 数据，形状为 [C, X, Y, Z]
        np.ndarray: 标签，形状为 [1, X, Y, Z] 
            (如果文件不存在则为None)
        dict: 属性
    """
    stack = load_npz_looped(os.path.join(folder, case_id) + ".npz",
                            keys=["data"], num_tries=3,
                            )["data"]
    data = stack[:-1]
    seg = stack[-1]

    with open(os.path.join(folder, case_id) + ".pkl", "rb") as f:
        props = pickle.load(f)
    assert data.shape[1:] == seg.shape, (f"Data and segmentation need to have same dim (except first). "
                                         f"Found data {data.shape} and "
                                         f"mask {seg.shape} for case {case_id}")
    return data.astype(np.float32), seg.astype(np.int32), props


@contextmanager
def npy_dataset(folder: str, processes: int,
                unpack: bool = True, delete_npy: bool = True,
                delete_npz: bool = False):
    """
    上下文管理器，用于处理一个包含.npz文件的数据集

    参数:
        folder: 数据集路径
        processes: 并行解包的进程数
        unpack: 是否解包数据集
        delete_npy: 是否在退出时删除.npy文件
        delete_npz: 是否在解包后删除.npz文件
    """
    if unpack:
        unpack_dataset(Path(folder), processes, delete_npz=delete_npz)
    try:
        yield True
    finally:
        if delete_npy:
            del_npy(Path(folder))


def unpack_dataset(folder: Pathlike,
                   processes: int,
                   delete_npz: bool = False):
    """
    解包数据集：将.npz文件转换为.npy文件

    参数:
        folder: 数据集路径
        processes: 并行解包的进程数
        delete_npz: 是否在解包后删除.npz文件
    """
    logger.info("Unpacking dataset")
    npz_files = subfiles(Path(folder), identifier="*.npz", join=True)
    if not npz_files:
        logger.warning(f'No paths found in {Path(folder)} matching *.npz')
        return
    with Pool(processes) as p:
        p.starmap(npz2npy, zip(npz_files, repeat(delete_npz)))


def pack_dataset(folder, processes: int, key: str):
    """
    打包数据集：将.npy文件转换为.npz文件

    参数:
        folder: 数据集路径
        processes: 并行打包的进程数
        key: 数据在npz文件中的键
    """
    logger.info("Packing dataset")
    npy_files = subfiles(Path(folder), identifier="*.npy", join=True)
    with Pool(processes) as p:
        p.starmap(npy2npz, zip(npy_files, repeat(key)))


def npz2npy(npz_file: str, delete_npz: bool = False):
    """
    将单个npz文件转换为多个npy文件

    参数:
        npz_file: npz文件路径
        delete_npz: 是否在转换后删除npz文件
    """
    if not os.path.isfile(npz_file[:-3] + "npy"):
        a = load_npz_looped(npz_file, keys=["data", "seg"], num_tries=3)
        if a is not None:
            np.save(npz_file[:-3] + "npy", a["data"])
            np.save(npz_file[:-4] + "_seg.npy", a["seg"])
    if delete_npz:
        os.remove(npz_file)


def npy2npz(npy_file: str, key: str):
    """
    将单个npy文件转换为npz文件

    参数:
        npy_file: npy文件路径
        key: 数据在npz文件中的键
    """
    d = np.load(npy_file)
    np.savez_compressed(npy_file[:-3] + "npz", **{key: d})


def del_npy(folder: Pathlike):
    """
    从文件夹中删除所有npy文件

    参数:
        folder: 包含npy文件的文件夹
    """
    npy_files = Path(folder).glob("*.npy")
    npy_files = [i for i in npy_files if os.path.isfile(i)]
    logger.info(f"Found {len(npy_files)} for removal")
    for n in npy_files:
        os.remove(n)


def load_json(path: Path, **kwargs) -> Any:
    """
    加载JSON文件

    参数:
        path: 文件路径
        **kwargs: 传递给json.load的额外参数

    返回:
        文件内容

    异常:
        FileNotFoundError: 如果文件不存在
    """
    if isinstance(path, str):
        path = Path(path)
    if not(".json" == path.suffix):
        path = str(path) + ".json"

    with open(path, "r") as f:
        data = json.load(f, **kwargs)
    return data


def save_json(data: Any, path: Pathlike, indent: int = 4, **kwargs):
    """
    保存数据为JSON文件

    参数:
        data: 要保存的数据
        path: 保存路径
        indent: 缩进量
        **kwargs: 传递给json.dump的额外参数
    """
    if isinstance(path, str):
        path = Path(path)
    if not(".json" == path.suffix):
        path = Path(str(path) + ".json")

    with open(path, "w") as f:
        json.dump(data, f, indent=indent, **kwargs)


def load_pickle(path: Path, **kwargs) -> Any:
    """
    加载Pickle文件

    参数:
        path: 文件路径
        **kwargs: 传递给pickle.load的额外参数

    返回:
        文件内容

    异常:
        FileNotFoundError: 如果文件不存在
    """
    if isinstance(path, str):
        path = Path(path)
    if not any([fix == path.suffix for fix in [".pickle", ".pkl"]]):
        path = Path(str(path) + ".pkl")

    with open(path, "rb") as f:
        data = pickle.load(f, **kwargs)
    return data


def save_pickle(data: Any, path: Pathlike, **kwargs):
    """
    保存数据为Pickle文件

    参数:
        data: 要保存的数据
        path: 保存路径
        **kwargs: 传递给pickle.dump的额外参数
    """
    if isinstance(path, str):
        path = Path(path)
    if not any([fix == path.suffix for fix in [".pickle", ".pkl"]]):
        path = str(path) + ".pkl"

    with open(str(path), "wb") as f:
        data = pickle.dump(data, f, **kwargs)
    return data


def save_yaml(data: Any, path: Path, **kwargs):
    """
    保存数据为YAML文件

    参数:
        data: 要保存的数据
        path: 保存路径
        **kwargs: 传递给yaml.dump的额外参数

    注意:
        如果使用的是旧版PyYAML，safe_dump参数不起作用，
        这可能导致安全问题。确保使用了最新版本的PyYAML。
    """
    if isinstance(path, str):
        path = Path(path)
    if not(".yaml" == path.suffix):
        path = str(path) + ".yaml"

    with open(path, "w") as f:
        yaml.dump(data, f, **kwargs)


def save_txt(data: str, path: Path, **kwargs):
    """
    保存字符串到文本文件

    参数:
        data: 要保存的字符串
        path: 保存路径
        **kwargs: 传递给open的额外参数
    """
    if isinstance(path, str):
        path = Path(path)
    if not(".txt" == path.suffix):
        path = str(path) + ".txt"

    with open(path, "a") as f:
        f.write(str(data))


def load_npz_looped(
        p: Pathlike,
        keys: Sequence[str],
        *args,
        num_tries: int = 3,
        **kwargs,
        ) -> Union[np.ndarray, dict]:
    """
    使用重试机制加载NPZ文件
    
    有时候，由于压缩的原因，加载npz文件会失败。
    此函数会多次尝试加载文件，减少失败概率。

    参数:
        p: 文件路径
        keys: 要加载的键
        *args: 传递给np.load的位置参数
        num_tries: 尝试次数
        **kwargs: 传递给np.load的关键字参数

    返回:
        如果keys长度为1，则返回单个数组，
        否则返回字典，键是给定的keys，值是对应的数组

    异常:
        ValueError: 如果文件无法加载
    """
    if num_tries <= 0:
        raise ValueError(f"Num tires needs to be larger than 0, found {num_tries} tries.")

    for i in range(num_tries):  # try reading the file 3 times
        try:
            _data = np.load(str(p), *args, **kwargs)
            data = {k: _data[k] for k in keys}
            break
        except Exception as e:
            if i == num_tries - 1:
                logger.error(f"Could not unpack {p}")
                return None
            time.sleep(5.)
    return data
