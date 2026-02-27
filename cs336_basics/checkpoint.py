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
    optimizer: torch.optim.Optimizer = None
) -> int:
    """
    从指定路径加载状态，恢复模型和优化器，并返回保存时的迭代步数。
    自动处理 torch.compile 产生的 _orig_mod. 前缀。
    """
    device = next(model.parameters()).device
    checkpoint = torch.load(src, map_location=device)
    
    # 1. 恢复模型状态，自动处理前缀
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace("_orig_mod.", "") # 去掉 compile 产生的装饰器前缀
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    
    # 2. 恢复优化器状态（如果提供）
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # 3. 返回训练进度
    return checkpoint.get('iteration', 0)
def find_latest_checkpoint(out_dir: str) -> str:
    """
    寻找指定目录下最新的 iteration checkpoint 文件。
    """
    import glob
    ckpts = sorted(glob.glob(os.path.join(out_dir, "ckpt_iter*.pt")), 
                   key=lambda x: int(os.path.basename(x).split("iter")[-1].split(".")[0]))
    if not ckpts:
        return None
    return ckpts[-1]
