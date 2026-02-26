import math
import torch
import torch.nn as nn
from einops import einsum
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor
from typing import Optional

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


class FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        # 1. 计算硬件对齐的 d_ff
        # 先算 8/3 * d_model
        # 向上取整到 64 的倍数
        if d_ff is None:
            d_ff_raw = (8 / 3) * d_model
            self.d_ff = int(64 * math.ceil(d_ff_raw / 64))
        else:
            self.d_ff = d_ff
        
        # 2. 定义三个线性层（不带 bias，符合主流 LLM 设计）
        self.w1 = Linear(d_model, self.d_ff, device=device, dtype=dtype)
        self.w3 = Linear(d_model, self.d_ff, device=device, dtype=dtype)
        self.w2 = Linear(self.d_ff, d_model, device=device, dtype=dtype)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 按照SwiGLU公式 (7): W2(SiLU(W1x) * W3x)
        
        # 计算 W1x 和 W3x
        x_w1 = self.w1(x) # (..., d_ff)
        x_w3 = self.w3(x) # (..., d_ff)
        
        # 实现 SiLU(W1x) = x * sigmoid(x)
        # 这里显式使用 torch.sigmoid 遵循作业的稳定性建议
        gate = x_w1 * torch.sigmoid(x_w1)
        
        # 逐元素相乘并映射回 d_model
        return self.w2(gate * x_w3)

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        # 1. 频率计算：确保使用 float64 提高预计算精度，然后再转回 float32
        # 公式: theta_i = theta ^ (-2i/d)
        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2).float() / d_k))
        
        # 2. 生成时间步/位置向量
        t = torch.arange(max_seq_len).float()
        
        # 3. 外积得到角度矩阵 (max_seq_len, d_k/2)
        freqs = torch.outer(t, inv_freq)
        
        # 4. 关键：交替重复构造 (max_seq_len, d_k)
        # 变成 [f1, f1, f2, f2, ...] 这种形式
        emb = torch.stack((freqs, freqs), dim=-1).flatten(1)
        
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # 获取切片后的 cos 和 sin, 定位旋转角度
        # 形状: (..., seq_len, d_k)
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]
        
        # 5. 构造交替旋转项 [-x2, x1, -x4, x3, ...]
        # 这种方式通常被称为 rotate_interleaved
        x1 = x[..., 0::2] # 取偶数索引
        x2 = x[..., 1::2] # 取奇数索引
        
        # 交替拼接：(-x2, x1)
        rotated_x = torch.stack((-x2, x1), dim=-1).flatten(-2)
        
        # 应用公式: x*cos + rotate(x)*sin
        return x * cos + rotated_x * sin

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    对输入张量的指定维度应用 Softmax 运算
    
    Args:
        x (torch.Tensor): 输入张量。
        dim (int): 需要进行 Softmax 的维度。
        
    Returns:
        torch.Tensor: 形状与 x 相同，但指定维度被归一化为概率分布
    """
    # 1. 寻找指定维度上的最大值
    # keepdim=True 是关键，它能保持维度数量不变，从而允许自动广播减法
    max_val = torch.max(x, dim=dim, keepdim=True)[0]
    
    # 2. 减去最大值（数值稳定性技巧：防止 exp(x) 变成 inf）
    x_stable = x - max_val
    
    # 3. 计算指数
    exp_x = torch.exp(x_stable)
    
    # 4. 归一化：除以该维度上所有指数值的总和
    sum_exp = torch.sum(exp_x, dim=dim, keepdim=True)
    
    return exp_x / sum_exp


def scaled_dot_product_attention(
    Q: torch.Tensor, 
    K: torch.Tensor, 
    V: torch.Tensor, 
    mask: torch.Tensor = None
) -> torch.Tensor:
    d_k = Q.shape[-1]
    
    # 1. 计算 Q @ K.T 并缩放
    # Q 形状: (..., n, d_k)
    # K 形状: (..., m, d_k)
    # 我们需要将 K 的最后两个维度转置，变成 (..., d_k, m) 才能进行矩阵乘法
    # 结果形状: (..., n, m)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    
    # 2. 应用掩码
    if mask is not None:
        # mask 为 False 的位置填充为负无穷
        scores = scores.masked_fill(~mask, float("-inf"))
    
    # 3. 归一化
    # 使用你手写的 Softmax
    weights = softmax(scores, dim=-1)
    
    # 4. 加权求和得到输出
    # weights 形状: (..., n, m)
    # V 形状: (..., m, d_v)
    # 结果形状: (..., n, d_v)
    return torch.matmul(weights, V)

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, theta: float, max_seq_len: int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        # 按照要求设置 dk = dv = dmodel / h
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        
        # 1. 定义 Q, K, V 的线性投影层
        self.W_q = Linear(d_model, num_heads * self.d_k, device=device, dtype=dtype)
        self.W_k = Linear(d_model, num_heads * self.d_k, device=device, dtype=dtype)
        self.W_v = Linear(d_model, num_heads * self.d_v, device=device, dtype=dtype)
        
        # 2. 定义输出投影层 W_O
        self.W_o = Linear(num_heads * self.d_v, d_model, device=device, dtype=dtype)
        
        # 3. 实例化旋转位置编码模块
        self.rope = RotaryPositionalEmbedding(theta, self.d_k, max_seq_len, device=device)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # x 形状: (batch, seq_len, d_model)
        batch_size, seq_len, _ = x.shape
        
        # A. 投影并切分为多头
        # 形状变化: (b, s, d_m) -> (b, s, h, d_k) -> (b, h, s, d_k)
        q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_v).transpose(1, 2)
        
        # B. 对 Q 和 K 应用 RoPE
        # 注意：RoPE 将头维度视为 Batch，因此位置向量需要扩展维度以对齐
        # token_positions 形状 (b, s) -> (b, 1, s) 从而广播到所有头
        q = self.rope(q, token_positions.unsqueeze(1))
        k = self.rope(k, token_positions.unsqueeze(1))
        
        # C. 构造因果掩码 (Causal Mask)
        # 形状 (seq_len, seq_len)，确保 j <= i 为 True
        # 构建下三角矩阵
        indices = torch.arange(seq_len, device=x.device)
        mask = indices.unsqueeze(0) <= indices.unsqueeze(1)
        
        # D. 调用缩放点积注意力
        # 输出形状: (batch, num_heads, seq_len, d_v)
        attn_out = scaled_dot_product_attention(q, k, v, mask=mask)
        
        # E. 合并多头并应用 W_O
        # (b, h, s, d_v) -> (b, s, h, d_v) -> (b, s, h * d_v)
        concat_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        return self.W_o(concat_out)

# Transformer Block
class TransformerBlock(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        num_heads: int, 
        d_ff: int, 
        theta: float, 
        max_seq_len: int, 
        device=None, 
        dtype=None
    ):
        super().__init__()
        
        # 1. 注意力子层的组件
        # Attention前面的RMSNorm
        self.norm1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.mha = CausalSelfAttention(
            d_model=d_model, 
            num_heads=num_heads, 
            theta=theta, 
            max_seq_len=max_seq_len, 
            device=device, 
            dtype=dtype
        )
        
        # 2. 前馈神经网络子层的组件
        # FFN前面的RMSNorm
        self.norm2 = RMSNorm(d_model, device=device, dtype=dtype)
        # 注意：这里调用的是你之前定义的 FFN 类
        self.ffn = FFN(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        x 形状: (batch_size, seq_len, d_model)
        token_positions 形状: (batch_size, seq_len)
        """
        # 子层 1: Multi-Head Attention + 残差连接
        # 公式: y = x + MultiHeadSelfAttention(RMSNorm(x))
        x_norm1 = self.norm1(x)
        attn_out = self.mha(x_norm1, token_positions)
        h = x + attn_out
        
        # 子层 2: Feed-Forward Network + 残差连接
        # 公式: z = h + FFN(RMSNorm(h))
        h_norm2 = self.norm2(h)
        ffn_out = self.ffn(h_norm2)
        out = h + ffn_out
        
        return out


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        theta: float,
        device=None,
        dtype=None
    ):
        super().__init__()
        
        # 1. 词嵌入层 (Token Embedding)
        # 将输入的 Token ID 转换为 d_model 维度的稠密向量
        self.token_embedding = Embedding(
            num_embeddings=vocab_size, 
            embedding_dim=d_model, 
            device=device, 
            dtype=dtype
        )
        
        # 2. Transformer 块列表 (N 层堆叠)
        # 使用 nn.ModuleList 来注册多层，确保 PyTorch 能正确追踪参数
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                theta=theta,
                max_seq_len=context_length, # context_length 对应 RoPE 的最大序列长度
                device=device,
                dtype=dtype
            ) for _ in range(num_layers)
        ])
        
        # 3. 最终层归一化 (Final Layer Norm)
        self.final_norm = RMSNorm(d_model, device=device, dtype=dtype)
        
        # 4. 语言模型头 (LM Head)
        # 将 d_model 维度的特征映射回 vocab_size，以输出每个词的概率预测
        self.lm_head = Linear(
            in_features=d_model, 
            out_features=vocab_size, 
            device=device, 
            dtype=dtype
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        token_ids 形状: (batch_size, seq_len)
        """
        batch_size, seq_len = token_ids.shape
        
        # 生成 token_positions，用于 RoPE 位置编码
        # 形状: (seq_len) -> (1, seq_len) -> (batch_size, seq_len)
        token_positions = torch.arange(seq_len, device=token_ids.device)
        token_positions = token_positions.unsqueeze(0).expand(batch_size, seq_len)
        
        # 1. 获取词嵌入向量: (batch_size, seq_len, d_model)
        x = self.token_embedding(token_ids)
        
        # 2. 依次穿过所有的 Transformer Block
        for layer in self.layers:
            x = layer(x, token_positions)
            
        # 3. 应用最终的 RMSNorm
        x = self.final_norm(x)
        
        # 4. 映射到词表空间得到 Logits: (batch_size, seq_len, vocab_size)
        logits = self.lm_head(x)
        
        return logits


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    计算交叉熵损失。
    
    参数:
        logits: 形状为 (..., vocab_size) 的预测分值
        targets: 形状为 (...) 的真实标签索引
    """
    # 1. 数值稳定性处理：减去每个位置的最大值
    # keepdim=True 保证减法时的广播正确
    max_logits = torch.max(logits, dim=-1, keepdim=True).values
    stable_logits = logits - max_logits
    
    # 2. 提取正确类别的 logit (即公式中的 oi[xi+1])
    # 使用 gather 将 target 对应的 logit 拿出来
    # 假设 targets 形状为 (B, S)，logits 形状为 (B, S, V)
    # 我们需要在最后一个维度进行 gather
    target_logits = torch.gather(stable_logits, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    
    # 3. 计算 log-sum-exp 部分
    # 公式: log(sum(exp(stable_logits)))
    log_sum_exp = torch.log(torch.sum(torch.exp(stable_logits), dim=-1))
    
    # 4. 计算负对数似然 (NLL)
    # ℓ = -(target_logit - log_sum_exp) = log_sum_exp - target_logit
    # 注意：这里我们已经减去了 max_logits，但它在分子分母中相互抵消了
    loss_per_token = log_sum_exp - target_logits
    
    # 5. 返回所有 batch 维度的平均值
    return torch.mean(loss_per_token)


@torch.no_grad()
def sample(model, prompt_tokens, max_new_tokens, temperature=1.0, top_p=1.0, eos_token_id=None):
    """
    根据给定的 prompt 生成后续文本。
    """
    model.eval()
    # 将输入转为 tensor 并增加 batch 维度: (1, T)
    curr_tokens = torch.tensor(prompt_tokens, dtype=torch.long, device=next(model.parameters()).device).unsqueeze(0)
    
    for _ in range(max_new_tokens):
        # 1. 截断输入以适应 context_length (如果模型不支持长序列)
        # 注意：这里假设你的模型在 forward 里处理了位置编码的上限
        idx_cond = curr_tokens[:, -model.context_length:] if curr_tokens.size(1) > model.context_length else curr_tokens
        
        # 2. 前向传播拿到最后一个位置的 logits: (1, vocab_size)
        logits = model(idx_cond)
        logits = logits[:, -1, :] 
        
        # 3. 应用温度缩放
        if temperature != 1.0:
            logits = logits / temperature
            
        # 4. Top-p (Nucleus) 采样过滤
        if top_p < 1.0:
            # 对 logits 进行降序排列
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            # 计算累加概率
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # 找到超过阈值 p 的位置
            # 我们保留累加和小于 p 的 token，以及第一个超过 p 的 token (核)
            sorted_indices_to_remove = cumulative_probs > top_p
            # 将第一个超过阈值的 token 的掩码设为 False (即保留它)
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False
            
            # 将需要剔除的 token 概率设为负无穷
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[0, indices_to_remove] = -float('Inf')
            
        # 5. 转化为概率分布并采样
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1) # (1, 1)
        
        # 6. 将新生成的 token 拼接到序列中
        curr_tokens = torch.cat((curr_tokens, next_token), dim=1)
        
        # 7. 如果撞到了结束符 <|endoftext|>，提前停止
        if eos_token_id is not None and next_token.item() == eos_token_id:
            break
            
    return curr_tokens.squeeze(0).tolist()