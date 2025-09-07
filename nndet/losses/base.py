import torch

__all__ = ["reduction_helper"]


def reduction_helper(data: torch.Tensor, reduction: str) -> torch.Tensor:
    """
    帮助以不同模式汇总数据的辅助函数

    参数:
        data: 需要汇总的数据
        reduction: 归约类型。可选 `mean`, `sum`, None

    返回:
        Tensor: 归约后的数据
    """
    if reduction == 'mean':
        return torch.mean(data)
    if reduction == 'none' or reduction is None:
        return data
    if reduction == 'sum':
        return torch.sum(data)
    raise AttributeError('Reduction参数未知。')
