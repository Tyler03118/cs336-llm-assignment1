import torch
import os
import typing

def save_checkpoint(
    model: torch.nn.Module, 
    optimizer: torch.optim.Optimizer, 
    iteration: int, 
    out: typing.Union[str, os.PathLike, typing.BinaryIO, typing.IO[bytes]]
) -> None:
    """
    保存模型权重、优化器状态和当前迭代步数到指定路径或文件对象。
    """
    # 1. 构造一个包含所有必要状态的大字典
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration
    }
    
    # 2. 使用 PyTorch 的序列化工具保存到硬盘
    torch.save(checkpoint, out)


def load_checkpoint(
    src: typing.Union[str, os.PathLike, typing.BinaryIO, typing.IO[bytes]], 
    model: torch.nn.Module, 
    optimizer: torch.optim.Optimizer
) -> int:
    """
    从指定路径加载状态，恢复模型和优化器，并返回保存时的迭代步数。
    """
    # 1. 加载字典
    # 💡 M1 Pro 避坑指南：为了防止在不同设备间移动权重时报错（比如在 CPU 上读取 GPU 存的权重），
    # 最好显式地告诉 PyTorch 把权重加载到当前模型所在的设备上。
    # 这里我们使用 map_location 参数，确保安全加载。
    device = next(model.parameters()).device
    checkpoint = torch.load(src, map_location=device)
    
    # 2. 恢复模型和优化器的状态
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # 3. 返回训练进度
    return checkpoint['iteration']