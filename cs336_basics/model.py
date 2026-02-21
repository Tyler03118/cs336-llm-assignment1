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

class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        # 1. 创建嵌入矩阵。要求 d_model 在最后一个维度。
        # 形状为 (num_embeddings, embedding_dim)
        self.weight = nn.Parameter(
            torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        )
        
        # 2. 初始化：根据作业要求，使用均值为 0，方差为 1 的截断正态分布
        # 截断范围为 [-3, 3]
        nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=1.0,  # 因为方差 sigma^2 = 1，所以标准差 sigma 也是 1
            a=-3.0,
            b=3.0
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        token_ids 形状: (batch_size, sequence_length) 类型为 torch.LongTensor
        返回形状: (batch_size, sequence_length, embedding_dim)
        """
        # 在 PyTorch 中，对 Tensor 使用索引会自动触发“查表”操作
        # 这等价于 nn.functional.embedding
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        # 按照初始化规则，RMSNorm 的 gain 参数初始化为 1
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. 记录原始类型并转换为 float32 防止平方溢出
        in_dtype = x.dtype
        x_float = x.to(torch.float32)

        # 2. 计算均方根 (RMS)
        # x_float.pow(2).mean(-1, keepdim=True) 计算每个向量元素的平方平均值
        rms = torch.sqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps)

        # 3. 归一化并应用可学习的权重 g_i
        # 注意：self.weight 也会自动广播到 batch 维度
        result = (x_float / rms) * self.weight

        # 4. 转回原始类型
        return result.to(in_dtype)