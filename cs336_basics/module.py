import math
import torch
import torch.nn as nn
from einops import einsum

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        # 必须调用的父类初始化
        super().__init__()
        
        # 1. 创建权重矩阵 W
        # 作业要求：W 的形状必须是 (out_features, in_features) 以符合行优先的内存排序
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=dtype)
        )
        
        # 2. 计算标准差 std = sqrt(2 / (d_in + d_out))
        std = math.sqrt(2.0 / (in_features + out_features))
        
        # 3. 截断正态分布初始化 (限制在 -3*std 到 3*std 之间)
        nn.init.trunc_normal_(
            self.weight, 
            mean=0.0, 
            std=std, 
            a=-3.0 * std, 
            b=3.0 * std
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x 形状: (..., in_features)
        weight 形状: (out_features, in_features)
        返回形状: (..., out_features)
        """
        # 使用 einsum 进行无脑且优雅的维度消消乐！
        # 把 in_features 这个维度乘掉，留下 out_features
        return einsum(x, self.weight, '... in_f, out_f in_f -> ... out_f')
        
        # 备选方案：如果你更喜欢原生的 PyTorch 写法，可以换成这句：
        # return x @ self.weight.T