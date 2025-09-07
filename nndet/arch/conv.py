import torch
import torch.nn as nn
from typing import Union, Callable, Any, Optional, Tuple, Sequence, Type

from nndet.arch.initializer import InitWeights_He
from nndet.arch.layers.norm import GroupNorm


NdParam = Union[int, Tuple[int, int], Tuple[int, int, int]]


class Generator:
    def __init__(self, conv_cls, dim: int):
        """
        工厂助手类，保存卷积类和维度信息以生成对象
        
        参数:
            conv_cls (callable): 卷积类
            dim (int): 空间维度数量（通常为2或3）
        """
        self.dim = dim
        self.conv_cls = conv_cls

    def __call__(self, *args, **kwargs) -> Any:
        """
        创建对象
        
        参数:
            *args: 传递给对象的位置参数
            **kwargs: 传递给对象的关键字参数
            
        返回:
            Any: 创建的对象
        """
        return self.conv_cls(self.dim, *args, **kwargs)


class BaseConvNormAct(torch.nn.Sequential):
    def __init__(self,
                 dim: int,
                 in_channels: int,
                 out_channels: int,
                 norm: Optional[Union[Callable[..., Type[nn.Module]], str]],
                 act: Optional[Union[Callable[..., Type[nn.Module]], str]],
                 kernel_size: Union[int, tuple],
                 stride: Union[int, tuple] = 1,
                 padding: Union[int, tuple] = 0,
                 dilation: Union[int, tuple] = 1,
                 groups: int = 1,
                 bias: bool = None,
                 transposed: bool = False,
                 norm_kwargs: Optional[dict] = None,
                 act_inplace: Optional[bool] = None,
                 act_kwargs: Optional[dict] = None,
                 initializer: Callable[[nn.Module], None] = None,
                 ):
        """
        默认顺序的基类:
        卷积 -> 归一化 -> 激活
        
        参数：
            dim: 卷积应选择的维度数
            in_channels: 输入通道数
            out_channels: 输出通道数
            norm: 归一化类型。如果为None，则不应用归一化
            kernel_size: 卷积核大小
            act: 非线性类；如果为None则不使用激活函数
            stride: 卷积步长
            padding: 填充值（输入或输出填充取决于卷积是否转置）
            dilation: 卷积扩张率
            groups: 卷积组数
            bias: 是否包含偏置
                 如果为None，偏置将动态确定：如果有归一化层则为False，否则为True
            transposed: 卷积是否应为转置卷积
            norm_kwargs: 归一化层的关键字参数
            act_inplace: 是否执行原地激活
                         如果为None，将动态确定：如果有归一化层则为True，否则为False
            act_kwargs: 非线性层的关键字参数
            initializer: 初始化权重的函数
        """
        super().__init__()
        # 处理可选参数
        norm_kwargs = {} if norm_kwargs is None else norm_kwargs
        act_kwargs = {} if act_kwargs is None else act_kwargs

        if "inplace" in act_kwargs:
            raise ValueError("使用关键字参数来启用/禁用原地激活")
        if act_inplace is None:
            act_inplace = bool(norm is not None)
        act_kwargs["inplace"] = act_inplace

        # 处理动态值
        bias = bool(norm is None) if bias is None else bias

        conv = nd_conv(dim=dim,
                       in_channels=in_channels,
                       out_channels=out_channels,
                       kernel_size=kernel_size,
                       stride=stride,
                       padding=padding,
                       dilation=dilation,
                       groups=groups,
                       bias=bias,
                       transposed=transposed
                       )
        self.add_module("conv", conv)

        if norm is not None:
            if isinstance(norm, str):
                _norm = nd_norm(norm, dim, out_channels, **norm_kwargs)
            else:
                _norm = norm(dim, out_channels, **norm_kwargs)
            self.add_module("norm", _norm)

        if act is not None:
            if isinstance(act, str):
                _act = nd_act(act, dim, **act_kwargs)
            else:
                _act = act(**act_kwargs)
            self.add_module("act", _act)

        if initializer is not None:
            self.apply(initializer)


class ConvInstanceRelu(BaseConvNormAct):
    def __init__(self,
                 dim: int,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, tuple],
                 stride: Union[int, tuple] = 1,
                 padding: Union[int, tuple] = 0,
                 dilation: Union[int, tuple] = 1,
                 groups: int = 1,
                 bias: bool = None,
                 transposed: bool = False,
                 add_norm: bool = True,
                 add_act: bool = True,
                 act_inplace: Optional[bool] = None,
                 norm_eps: float = 1e-5,
                 norm_affine: bool = True,
                 initializer: Callable[[nn.Module], None] = None,
                 ):
        """
        实例归一化+ReLU的卷积块
        
        参数：
            dim: 卷积应选择的维度数
            in_channels: 输入通道数
            out_channels: 输出通道数
            kernel_size: 卷积核大小
            stride: 卷积步长
            padding: 填充值（输入或输出填充取决于卷积是否转置）
            dilation: 卷积扩张率
            groups: 卷积组数
            bias: 是否包含偏置
                 如果为None，偏置将动态确定：如果有归一化层则为False，否则为True
            transposed: 卷积是否应为转置卷积
            add_norm: 是否向卷积块添加归一化层
            add_act: 是否向卷积块添加激活层
            act_inplace: 是否执行原地激活
                         如果为None，将动态确定：如果有归一化层则为True，否则为False
            norm_eps: 实例归一化的eps参数（详见pytorch文档）
            norm_affine: 实例归一化的affine参数（详见pytorch文档）
            initializer: 初始化权重的函数
        """
        norm = "Instance" if add_norm else None
        act = "ReLU" if add_act else None
        
        super().__init__(
            dim=dim,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            transposed=transposed,
            norm=norm,
            act=act,
            norm_kwargs={
                "eps": norm_eps,
                "affine": norm_affine,
            },
            act_inplace=act_inplace,
            initializer=initializer,
        )


class ConvGroupRelu(BaseConvNormAct):
    def __init__(self,
                 dim: int,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, tuple],
                 stride: Union[int, tuple] = 1,
                 padding: Union[int, tuple] = 0,
                 dilation: Union[int, tuple] = 1,
                 groups: int = 1,
                 bias: bool = None,
                 transposed: bool = False,
                 add_norm: bool = True,
                 add_act: bool = True,
                 act_inplace: Optional[bool] = None,
                 norm_eps: float = 1e-5,
                 norm_affine: bool = True,
                 norm_channels_per_group: int = 16,
                 initializer: Callable[[nn.Module], None] = None,
                 ):
        """
        组归一化+ReLU的卷积块
        
        参数：
            dim: 卷积应选择的维度数
            in_channels: 输入通道数
            out_channels: 输出通道数
            kernel_size: 卷积核大小
            stride: 卷积步长
            padding: 填充值（输入或输出填充取决于卷积是否转置）
            dilation: 卷积扩张率
            groups: 卷积组数
            bias: 是否包含偏置
                 如果为None，偏置将动态确定：如果有归一化层则为False，否则为True
            transposed: 卷积是否应为转置卷积
            add_norm: 是否向卷积块添加归一化层
            add_act: 是否向卷积块添加激活层
            act_inplace: 是否执行原地激活
                         如果为None，将动态确定：如果有归一化层则为True，否则为False
            norm_eps: 组归一化的eps参数（详见pytorch文档）
            norm_affine: 组归一化的affine参数（详见pytorch文档）
            norm_channels_per_group: 组归一化中每组的通道数
            initializer: 初始化权重的函数
        """
        norm = "Group" if add_norm else None
        act = "ReLU" if add_act else None
        
        super().__init__(
            dim=dim,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            transposed=transposed,
            norm=norm,
            act=act,
            norm_kwargs={
                "eps": norm_eps,
                "affine": norm_affine,
                "channels_per_group": norm_channels_per_group,
            },
            act_inplace=act_inplace,
            initializer=initializer,
        )

# 
class ConvBatchLeaky(BaseConvNormAct):
    def __init__(self,
                 dim: int,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, tuple],
                 stride: Union[int, tuple] = 1,
                 padding: Union[int, tuple] = 0,
                 dilation: Union[int, tuple] = 1,
                 groups: int = 1,
                 bias: bool = None,
                 transposed: bool = False,
                 add_norm: bool = True,
                 add_act: bool = True,
                 act_inplace: Optional[bool] = None,
                 norm_eps: float = 1e-5,
                 norm_affine: bool = True,
                 initializer: Callable[[nn.Module], None] = None,
                 ):
        """
        批归一化+LeakyReLU的卷积块

        参数：
            dim: 卷积应选择的维度数
            in_channels: 输入通道数
            out_channels: 输出通道数
            kernel_size: 卷积核大小
            stride: 卷积步长
            padding: 填充值（输入或输出填充取决于卷积是否转置）
            dilation: 卷积扩张率
            groups: 卷积组数
            bias: 是否包含偏置
                 如果为None，偏置将动态确定：如果有归一化层则为False，否则为True
            transposed: 卷积是否应为转置卷积
            add_norm: 是否向卷积块添加归一化层
            add_act: 是否向卷积块添加激活层
            act_inplace: 是否执行原地激活
                         如果为None，将动态确定：如果有归一化层则为True，否则为False
            norm_eps: 批归一化的eps参数（详见pytorch文档）
            norm_affine: 批归一化的affine参数（详见pytorch文档）
            initializer: 初始化权重的函数
        """
        norm = "Batch" if add_norm else None
        super().__init__(
            dim=dim,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            transposed=transposed,
            norm=norm,
            act='LeakyReLU',
            norm_kwargs={
                "eps": norm_eps,
                "affine": norm_affine,
            },
            act_inplace=act_inplace,
            initializer=initializer,
        )
        

class ConvDilatedBatchRelu(BaseConvNormAct):
    def __init__(self,
                 dim: int,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, tuple] = 3,
                 stride: Union[int, tuple] = 1,
                 padding: Union[int, tuple] = None,
                 dilation: Union[int, tuple] = 1,
                 groups: int = 1,
                 bias: bool = None,
                 transposed: bool = False,
                 add_norm: bool = True,
                 add_act: bool = True,
                 act_inplace: Optional[bool] = None,
                 norm_eps: float = 1e-5,
                 norm_momentum: float = 0.1,
                 norm_affine: bool = True,
                 initializer: Callable[[nn.Module], None] = InitWeights_He(1e-2),
                 ):
        
        # 如果没有指定padding，则根据kernel_size和dilation自动计算，以保持特征图大小不变
        if padding is None:
            # 针对扩张卷积自动计算padding
            if isinstance(kernel_size, int) and isinstance(dilation, int):
                padding = ((kernel_size - 1) * dilation) // 2
            else:
                # 处理元组情况
                if isinstance(kernel_size, int):
                    kernel_size = (kernel_size,) * dim
                if isinstance(dilation, int):
                    dilation = (dilation,) * dim
                padding = tuple(((k - 1) * d) // 2 for k, d in zip(kernel_size, dilation))
        
        super().__init__(
            dim=dim,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            transposed=transposed,
            # 根据add_norm和add_act决定是否使用norm和act
            norm="Batch" if add_norm else None,
            act="ReLU" if add_act else None,
            act_inplace=act_inplace,
            norm_kwargs={"eps": norm_eps, "momentum": norm_momentum, "affine": norm_affine},
            initializer=initializer,
        )

def nd_conv(dim: int,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, tuple],
            stride: Union[int, tuple] = 1,
            padding: Union[int, tuple] = 0,
            dilation: Union[int, tuple] = 1,
            groups: int = 1,
            bias: bool = True,
            transposed: bool = False,
            **kwargs,
            ) -> torch.nn.Module:
    """
    卷积包装器，通过单个参数切换不同维度和转置卷积
    
    参数：
        dim (int): 卷积应选择的维度数
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        kernel_size (int或可迭代对象): 卷积核大小
        stride (int或可迭代对象): 卷积步长
        padding (int或可迭代对象): 填充值
            （输入或输出填充取决于卷积是否转置）
        dilation (int或可迭代对象): 卷积扩张率
        groups (int): 卷积组数
        bias (bool): 是否包含偏置
        transposed (bool): 卷积是否应为转置卷积
        
    返回:
        torch.nn.Module: 生成的模块
        
    另见:
        Torch卷积类:
            * :class:`torch.nn.Conv1d`
            * :class:`torch.nn.Conv2d`
            * :class:`torch.nn.Conv3d`
            * :class:`torch.nn.ConvTranspose1d`
            * :class:`torch.nn.ConvTranspose2d`
            * :class:`torch.nn.ConvTranspose3d`
    """
    if transposed:
        transposed_str = "Transpose"
    else:
        transposed_str = ""

    conv_cls = getattr(torch.nn, f"Conv{transposed_str}{dim}d")

    return conv_cls(in_channels=in_channels, out_channels=out_channels,
                    kernel_size=kernel_size, stride=stride, padding=padding,
                    dilation=dilation, groups=groups, bias=bias, **kwargs)


def nd_pool(pooling_type: str, dim: int, *args, **kwargs) -> torch.nn.Module:
    """
    通过单个参数在不同池化类型和卷积之间切换的包装器
    
    参数：
        pooling_type (str): 池化类型，区分大小写。
                支持的值有:
                * ``Max`` - 最大池化
                * ``Avg`` - 平均池化
                * ``AdaptiveAvg`` - 自适应平均池化
                * ``AdaptiveMax`` - 自适应最大池化
        dim (int): 维度数
        *args: 所选池化类的位置参数
        **kwargs: 所选池化类的关键字参数
        
    返回:
        torch.nn.Module: 生成的模块
        
    另见:
        Torch池化类:
            * :class:`torch.nn.MaxPool1d`
            * :class:`torch.nn.MaxPool2d`
            * :class:`torch.nn.MaxPool3d`
            * :class:`torch.nn.AvgPool1d`
            * :class:`torch.nn.AvgPool2d`
            * :class:`torch.nn.AvgPool3d`
            * :class:`torch.nn.AdaptiveMaxPool1d`
            * :class:`torch.nn.AdaptiveMaxPool2d`
            * :class:`torch.nn.AdaptiveMaxPool3d`
            * :class:`torch.nn.AdaptiveAvgPool1d`
            * :class:`torch.nn.AdaptiveAvgPool2d`
            * :class:`torch.nn.AdaptiveAvgPool3d`
    """
    pool_cls = getattr(torch.nn, f"{pooling_type}Pool{dim}d")
    return pool_cls(*args, **kwargs)


def nd_norm(norm_type: str, dim: int, *args, **kwargs) -> torch.nn.Module:
    """
    通过单个参数在不同类型的归一化和维度之间切换的包装器
    
    参数：
        norm_type (str): 归一化类型，区分大小写。
            支持的类型有:
                * ``Batch`` - 批归一化
                * ``Instance`` - 实例归一化
                * ``LocalResponse`` - 局部响应归一化
                * ``Group`` - 组归一化
                * ``Layer`` - 层归一化
        dim (int, None): 归一化输入的维度；如果归一化与维度无关
            (例如LayerNorm)，可以为None
        *args: 所选归一化类的位置参数
        **kwargs: 所选归一化类的关键字参数
        
    返回:
        torch.nn.Module: 生成的模块
        
    另见:
        Torch归一化类:
                * :class:`torch.nn.BatchNorm1d`
                * :class:`torch.nn.BatchNorm2d`
                * :class:`torch.nn.BatchNorm3d`
                * :class:`torch.nn.InstanceNorm1d`
                * :class:`torch.nn.InstanceNorm2d`
                * :class:`torch.nn.InstanceNorm3d`
                * :class:`torch.nn.LocalResponseNorm`
                * :class:`nndet.arch.layers.norm.GroupNorm`
    """
    if dim is None:
        dim_str = ""
    else:
        dim_str = str(dim)

    # 确保首字母大写
    if norm_type.lower() == "group":
        norm_cls = GroupNorm
    else:
        # 首字母大写，其余小写
        capitalized_norm_type = norm_type[0].upper() + norm_type[1:].lower()
        norm_cls = getattr(torch.nn, f"{capitalized_norm_type}Norm{dim_str}d")
    return norm_cls(*args, **kwargs)

def nd_act(act_type: str, dim: int, *args, **kwargs) -> torch.nn.Module:
    """
    通过字符串查找激活函数的助手
    dim参数被忽略。
    在torch.nn中查找激活。
    
    参数:
        act_type: 要查找的激活层的名称
        dim: 被忽略
        
    返回:
        torch.nn.Module: 激活模块
    """
    act_cls = getattr(torch.nn, f"{act_type}")
    return act_cls(*args, **kwargs)


def nd_dropout(dim: int, p: float = 0.5, inplace: bool = False, **kwargs) -> torch.nn.Module:
    """
    生成1,2,3维的dropout
    
    参数:
        dim (int): 维度数
        p (float): dropout概率
        inplace (bool): 是否原地应用操作
        **kwargs: 传递给dropout的其他参数
        
    返回:
        torch.nn.Module: 生成的模块
    """
    dropout_cls = getattr(torch.nn, "Dropout%dd" % dim)
    return dropout_cls(p=p, inplace=inplace, **kwargs)


def compute_padding_for_kernel(kernel_size: Union[int, Sequence[int]]) -> \
        Union[int, Tuple[int, int], Tuple[int, int, int]]:
    """
    计算填充值，使特征图在步长为1时保持其大小
    
    参数:
        kernel_size: 要计算填充的核大小
        
    返回:
        Union[int, Tuple[int, int], Tuple[int, int, int]]: 计算的填充值
    """
    if isinstance(kernel_size, Sequence):
        padding = tuple([(i - 1) // 2 for i in kernel_size])
    else:
        padding = (kernel_size - 1) // 2
    return padding


def conv_kwargs_helper(norm: bool, activation: bool):
    """
    帮助在默认有这些层的层中强制禁用归一化和激活的助手函数
    
    参数:
        norm: 启用/禁用归一化层
        activation: 启用/禁用激活层
        
    返回:
        dict: 传递给卷积生成器的关键字参数
    """
    kwargs = {
        "add_norm": norm,
        "add_act": activation,
    }
    return kwargs
