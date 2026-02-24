import numpy as np
import torch

def get_batch(
    x: np.ndarray, 
    batch_size: int, 
    context_length: int, 
    device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    从一维 token 数组中随机采样 batch_size 个序列，并放在指定设备上。
    """
    # 1. 确定随机采样的上限
    # 为了保证能同时切出长度为 context_length 的 X 和向后偏移 1 位的 Y，
    # 最大的起始索引只能到 len(x) - context_length - 1
    max_start_idx = len(x) - context_length
    
    # 2. 生成 batch_size 个随机起始点
    # torch.randint 的上限是开区间，所以直接传入 max_start_idx
    ix = torch.randint(high=max_start_idx, size=(batch_size,))
    
    # 3. 提取数据切片
    # 从 numpy 数组中切出 batch，并强制转换为 int64 (PyTorch Embedding 层要求长整型)
    x_batch = np.stack([x[i : i + context_length] for i in ix]).astype(np.int64)
    y_batch = np.stack([x[i + 1 : i + 1 + context_length] for i in ix]).astype(np.int64)
    
    # 4. 转换为 Tensor 并移动到目标设备 (比如你的 'mps')
    X_tensor = torch.from_numpy(x_batch).to(device)
    Y_tensor = torch.from_numpy(y_batch).to(device)
    
    return X_tensor, Y_tensor