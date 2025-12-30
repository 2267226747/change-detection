import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
from models.transformer.RMSNorm import RMSNorm
from models.transformer.flashattention import FlashAttention


# =================================================================
# 主 Transformer Block
# =================================================================

class FusionTransformerBlock(nn.Module):
    def __init__(self, cfg, if_query=False):
        super().__init__()

        # 简化引用
        block_cfg = cfg.model.transformer_block

        # 1. 基础参数读取
        self.if_query = if_query
        self.q_dim = getattr(block_cfg, 'query_dim', 1024)
        self.vis_dim = getattr(block_cfg, 'vision_dim', 4096)
        self.num_heads = getattr(block_cfg, 'num_heads', 16)
        self.mlp_ratio = getattr(block_cfg, 'mlp_ratio', 4.0)
        self.dropout = getattr(block_cfg, 'dropout', 0.1)
        self.if_q_pos = getattr(block_cfg, 'if_q_position', False)
        self.if_k_pos = getattr(block_cfg, 'if_k_position', True)
        self.if_v_pos = getattr(block_cfg, 'if_v_position', False)
        self.use_qk_norm = getattr(block_cfg, 'use_qk_norm', False)

        # 2. 动态选择 Norm 类 (主干 Norm)
        norm_layer_name = getattr(block_cfg, 'norm_layer', "LayerNorm")
        norm_map = {
            "LayerNorm": nn.LayerNorm,
            "RMSNorm": RMSNorm  # 自定义的 RMSNorm
        }
        NormLayer = norm_map.get(norm_layer_name, nn.LayerNorm)

        # 3. 动态选择 Activation 类
        act_layer_name = getattr(block_cfg, 'act_layer', "GELU")
        act_map = {"GELU": nn.GELU, "ReLU": nn.ReLU, "SiLU": nn.SiLU}
        ActLayer = act_map.get(act_layer_name, nn.GELU)

        print(
            f"Build Block with: Norm={NormLayer.__name__}, "
            f"QKNorm={self.use_qk_norm}, "
            f"ActLayer={ActLayer.__name__}, "
            f"if_query={if_query}")

        # -----------------------------------------------------------
        # A. Cross-Attention 部分
        if self.if_query:
            self.norm_cross = NormLayer(self.q_dim)
            # 使用自定义的 EfficientAttention 替换 nn.MultiheadAttention
            self.cross_attn = FlashAttention(
                q_dim=self.q_dim,
                kv_dim=self.vis_dim,
                num_heads=self.num_heads,
                qk_norm=self.use_qk_norm,  # 传入配置
                dropout=self.dropout
            )

        # B. Self-Attention 部分
        # query只和自己attention
        self.norm_self = NormLayer(self.q_dim)
        self.self_attn = FlashAttention(
            q_dim=self.q_dim,
            kv_dim=self.q_dim,
            num_heads=self.num_heads,
            qk_norm=self.use_qk_norm,  # 传入配置
            dropout=self.dropout
        )

        # C. FFN 部分
        self.norm_ffn = NormLayer(self.q_dim)
        hidden_dim = int(self.q_dim * self.mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(self.q_dim, hidden_dim),
            ActLayer(),
            nn.Dropout(self.dropout),
            nn.Linear(hidden_dim, self.q_dim),
            nn.Dropout(self.dropout)
        )

        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, q_t, vision_1=None, vision_2=None, vision_pos1=None, vision_pos2=None):
        """
        q_t: [Batch, Seq_Q, q_dim]
        vision_t: [Batch, Seq_KV, kv_dim]  (如果 if_cross=False，可以是 None)
        """

        x = q_t

        # 1. Cross-Attention
        # 将所有视觉相关的预处理（拼接、检查）都移到 if_cross 内部，避免无用的计算
        if self.if_query:
            assert vision_1 is not None and vision_2 is not None, "if_cross=True but Vision features missing"

            # 延迟合并：只有需要 Cross Attention 时才合并 Tensor
            vision_t = torch.cat([vision_1, vision_2], dim=1)

            # 准备 Key 和 Value
            # 默认 Key/Value 就是 vision_t，根据 flag 叠加位置编码
            k, v = vision_t, vision_t

            # 统一处理位置编码逻辑
            if self.if_k_pos or self.if_v_pos:
                assert vision_pos1 is not None and vision_pos2 is not None, "Pos embedding enabled but input is None"
                vision_pos = torch.cat([vision_pos1, vision_pos2], dim=1)

                if self.if_k_pos:
                    k = vision_t + vision_pos
                if self.if_v_pos:
                    v = vision_t + vision_pos

            # 计算 Cross Attention (Pre-Norm + Residual)
            # x = x + Dropout(Attn(Norm(x), k, v))
            x = x + self.dropout_layer(
                self.cross_attn(query=self.norm_cross(x), key=k, value=v)
            )

        # 2. Self-Attention (Query=x, Key=x, Value=x)
        # 结构: x = x + Dropout(SelfAttn(Norm(x)))
        x_norm = self.norm_self(x)
        x = x + self.dropout_layer(
            self.self_attn(query=x_norm, key=x_norm, value=x_norm)
        )

        # 3. FFN
        # 结构: x = x + Dropout(FFN(Norm(x)))
        x = x + self.dropout_layer(
            self.ffn(self.norm_ffn(x))
        )

        return x


# =================================================================
# 3. 模拟配置与运行
# =================================================================
if __name__ == "__main__":
    from utils.config import Config

    # 1. 加载配置
    cfg_path = "../../configs/defaults.yaml"
    if os.path.exists(cfg_path):
        cfg = Config.from_yaml(cfg_path)
        cfg.model.transformer_block.query_dim = 128
        cfg.model.transformer_block.vision_dim = 256
        cfg.model.transformer_block.num_head = 4

        # 检查 CUDA 是否可用（Flash Attention 在 GPU 上才有明显加速）
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Running on device: {device}")

        # 初始化模型
        model = FusionTransformerBlock(cfg, if_query=True).to(device)

        # 创建输入数据
        # Batch=2, Seq_Q=10, Seq_Ctx=20, Dim=1024
        q_input = torch.randn(2, 10, 128).to(device)
        ctx_input = torch.randn(2, 20, 256).to(device)

        # 前向传播
        with torch.cuda.amp.autocast(enabled=(device == "cuda")):  # FlashAttn 推荐使用半精度
            output = model(q_input, ctx_input)

        print(f"Input shape: {q_input.shape}")
        print(f"Output shape: {output.shape}")

        # 验证是否包含 QK Norm 参数
        if cfg.model.transformer_block.use_qk_norm:
            print("Check: Model contains QK Norm parameters:", hasattr(model.self_attn, 'q_norm'))
