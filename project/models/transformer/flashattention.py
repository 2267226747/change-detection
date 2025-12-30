import torch.nn as nn
import torch.nn.functional as F
from models.transformer.RMSNorm import RMSNorm


class FlashAttention(nn.Module):
    """
    自定义 Attention 模块，支持:
    1. Flash Attention (via F.scaled_dot_product_attention)
    2. QK Normalization (可选)
    """

    def __init__(self, q_dim, kv_dim, num_heads, qk_norm=False, dropout=0.0):
        """
        Args:
            q_dim: Query 的输入维度 (也是模型的主干维度)
            kv_dim: Key/Value 的输入维度 (如果不填，默认等于 q_dim)
        """
        super().__init__()
        self.q_dim = q_dim
        self.num_heads = num_heads
        self.head_dim = q_dim // num_heads

        if kv_dim:
            self.kv_dim = kv_dim
        else:
            self.kv_dim = q_dim

        self.scale = self.head_dim ** -0.5
        self.dropout = dropout
        self.use_qk_norm = qk_norm

        assert q_dim % num_heads == 0, "embed_dim 必须能被 num_heads 整除"

        # Q, K, V 投影层
        self.q_proj = nn.Linear(self.q_dim, self.q_dim)
        self.k_proj = nn.Linear(self.kv_dim, self.q_dim)
        self.v_proj = nn.Linear(self.kv_dim, self.q_dim)

        # 输出投影层
        self.out_proj = nn.Linear(self.q_dim, self.q_dim)

        # QK Normalization 组件
        if self.use_qk_norm:
            # QK Norm 通常对每个 Head 的维度进行 RMSNorm
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

    def forward(self, query, key, value):
        """
        Args:
            query: [B, L_q, q_dim]
            key:   [B, L_k, kv_dim]
            value: [B, L_k, kv_dim]
        """
        B, L_q, _ = query.shape
        _, L_k, _ = key.shape

        # 1. 投影
        # 经过这里后，q, k, v 的最后一维都变成了 self.q_dim
        q = self.q_proj(query)  # [B, L_q, q_dim]
        k = self.k_proj(key)  # [B, L_k, q_dim]
        v = self.v_proj(value)  # [B, L_k, q_dim]

        # 2. Reshape 为 [Batch, SeqLen, NumHeads, HeadDim]
        # 并转置为 [Batch, NumHeads, SeqLen, HeadDim] 以适配 FlashAttention
        q = q.view(B, L_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L_k, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L_k, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. Apply QK Norm (如果开启)
        # QK Norm 作用在 HeadDim 维度上
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # 4. Flash Attention
        # PyTorch 2.0+ 自动选择最优内核 (FlashAttention v2, MemEfficient 等)
        # dropout_p 只有在训练模式下才生效
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False  # Cross Attn 和普通 Self Attn 通常是非因果的(能看到所有Context)
        )

        # 5. Reshape back
        out = out.transpose(1, 2).contiguous().view(B, L_q, self.q_dim)

        # 6. Output Projection
        out = self.out_proj(out)

        return out
