import torch
import torch.nn as nn
from models.transformer.RMSNorm import RMSNorm
from models.transformer.flashattention import FlashAttention


class FusionTransformerBlock2(nn.Module):
    def __init__(self, cfg, if_query=True):
        """
        Args:
            cfg: 全局配置对象 (cfg.model.transformer_block)
            if_query: (原if_cross) 是否进行 Query-Image 的交互 (Siamese Cross Attention)
            fusion_type: 融合方式，可选 ['concat_sub', 'cross_attn']
        """
        super().__init__()

        # 简化引用
        block_cfg = cfg.model.transformer_block

        # 1. 基础参数读取
        self.if_query = if_query
        self.fusion_type = getattr(block_cfg, 'fusion_type', "cross_attn")
        self.q_dim = getattr(block_cfg, 'query_dim', 1024)
        self.vis_dim = getattr(block_cfg, 'vision_dim', 4096)
        self.num_heads = getattr(block_cfg, 'num_heads', 16)
        self.mlp_ratio = getattr(block_cfg, 'mlp_ratio', 4.0)
        self.dropout = getattr(block_cfg, 'dropout', 0.1)
        self.use_qk_norm = getattr(block_cfg, 'use_qk_norm', False)
        self.if_q_pos = getattr(block_cfg, 'if_q_position', False)
        self.if_k_pos = getattr(block_cfg, 'if_k_position', True)
        self.if_v_pos = getattr(block_cfg, 'if_v_position', False)

        # 2. 动态选择组件
        norm_layer_name = getattr(block_cfg, 'norm_layer', "LayerNorm")
        norm_map = {"LayerNorm": nn.LayerNorm, "RMSNorm": RMSNorm}
        NormLayer = norm_map.get(norm_layer_name, nn.LayerNorm)

        act_layer_name = getattr(block_cfg, 'act_layer', "GELU")
        act_map = {"GELU": nn.GELU, "ReLU": nn.ReLU, "SiLU": nn.SiLU}
        ActLayer = act_map.get(act_layer_name, nn.GELU)

        print(
            f"Build Block with: Norm={NormLayer.__name__}, "
            f"QKNorm={self.use_qk_norm}, "
            f"ActLayer={ActLayer.__name__}, "
            f"if_query={if_query}, "
            f"fusion={self.fusion_type}")

        # -----------------------------------------------------------
        # A. Siamese Cross-Attention (Query <-> Vision)
        # -----------------------------------------------------------
        if self.if_query:
            self.norm_query = NormLayer(self.q_dim)
            # 共享权重的 Attention，用于分别提取 I1 和 I2
            self.siamese_cross_attn = FlashAttention(
                q_dim=self.q_dim,
                kv_dim=self.vis_dim,
                num_heads=self.num_heads,
                qk_norm=self.use_qk_norm,
                dropout=self.dropout
            )

        # -----------------------------------------------------------
        # B. Communication / Interaction (Q1 <-> Q2)
        # -----------------------------------------------------------
        else:
            if self.fusion_type == 'concat_sub':
                # 策略1: 拼接 (Q1, Q2, Q1-Q2) -> Linear
                # 输入维度是 3 * q_dim, 输出维度是 q_dim
                self.fusion_proj = nn.Linear(3 * self.q_dim, self.q_dim)
                self.norm_fusion = NormLayer(3 * self.q_dim)  # 可选：在拼接后Norm
            elif self.fusion_type == 'cross_attn':
                # 策略: Cross Attention (Q1 search info in Q2 & Q2 search info in Q1)
                # Query=Q1/2, Key=Q2/1, Value=Q2/1
                self.norm_fusion = NormLayer(self.q_dim)
                self.fusion_cross_attn = FlashAttention(
                    q_dim=self.q_dim,
                    kv_dim=self.q_dim,  # 这里 KV 来自 Q1/2，维度也是 q_dim
                    num_heads=self.num_heads,
                    qk_norm=self.use_qk_norm,
                    dropout=self.dropout
                )
            else:
                raise ValueError(f"Unknown fusion type: {self.fusion_type}")

        # -----------------------------------------------------------
        # C. Self-Attention
        #    共享权重，分别处理Q1和Q2
        # -----------------------------------------------------------
        self.norm_self = NormLayer(self.q_dim)
        self.self_attn = FlashAttention(
            q_dim=self.q_dim,
            kv_dim=self.q_dim,
            num_heads=self.num_heads,
            qk_norm=self.use_qk_norm,
            dropout=self.dropout
        )

        # -----------------------------------------------------------
        # D. FFN
        #    共享权重，分别处理Q1和Q2
        # -----------------------------------------------------------
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

    def forward(self, q_t1, q_t2, vision_1=None, vision_2=None, vision_pos1=None, vision_pos2=None):
        """
        分别输入两组 Query 和 Vision Features 及 位置编码
        q1, q2: [B, L, D]
        vis1, vis2: [B, L_v, D_v]
        Input:  q1, q2 (两个流)
        Output: q1, q2 (更新后的两个流)
        """

        # 1. Siamese Cross-Attention (分别交互)
        if self.if_query:
            assert vision_1 is not None and vision_2 is not None, "if_cross=True but Vision features missing"
            # 准备 Key 和 Value
            # 默认 Key/Value 就是 vision_t，根据 flag 叠加位置编码
            k1, v1 = vision_1, vision_1
            k2, v2 = vision_2, vision_2

            # 统一处理位置编码逻辑
            if self.if_k_pos or self.if_v_pos:
                assert vision_pos1 is not None and vision_pos2 is not None, "Pos embedding enabled but input is None"
                if self.if_k_pos:
                    k1 = vision_1 + vision_pos1
                    k2 = vision_2 + vision_pos2
                if self.if_v_pos:
                    v1 = vision_1 + vision_pos1
                    v2 = vision_2 + vision_pos2

            # Branch 1
            q1_norm = self.norm_query(q_t1)
            attn_out_1 = self.siamese_cross_attn(query=q1_norm, key=k1, value=v1)
            q_t1 = q_t1 + self.dropout_layer(attn_out_1)  # Residual

            # Branch 2 (共享权重，同样的逻辑)
            q2_norm = self.norm_query(q_t2)
            attn_out_2 = self.siamese_cross_attn(query=q2_norm, key=k2, value=v2)
            q_t2 = q_t2 + self.dropout_layer(attn_out_2)  # Residual

        # 2. Fusion (融合 F1 和 F2)
        # 如果该层 block 进行 query vision feature，则不进行query token之间的交互
        else:

            if self.fusion_type == 'cross_attn':
                # 使用 F1 去查询 F2 的信息 (F1 -> F2)
                # Query = F1, Key/Value = F2
                q1_norm = self.norm_fusion(q_t1)
                q2_norm = self.norm_fusion(q_t2)

                fusion_out1 = self.fusion_cross_attn(query=q1_norm, key=q2_norm, value=q2_norm)
                fusion_out2 = self.fusion_cross_attn(query=q2_norm, key=q1_norm, value=q1_norm)

                # Residual connection: 加在 F1 上 (以 F1 为主视角)
                q_t1 = q_t1 + self.dropout_layer(fusion_out1)
                q_t2 = q_t2 + self.dropout_layer(fusion_out2)

            elif self.fusion_type == 'concat_sub':
                # 2.2 显式差分更新
                # 对于 Q1: 重点是 Q1 和 (Q1-Q2)
                diff1 = q_t1 - q_t2
                cat1 = torch.cat([q_t1, q_t2, diff1], dim=-1)
                # 投影回 dim
                cat1 = self.norm_fusion(cat1)
                q_t1 = q_t1 + self.dropout_layer(self.fusion_proj(cat1))  # Residual
                # 对于 Q2: 重点是 Q2 和 (Q2-Q1)
                diff2 = q_t2 - q_t1  # 注意方向
                cat2 = torch.cat([q_t2, q_t1, diff2], dim=-1)
                cat2 = self.norm_fusion(cat2)
                q_t2 = q_t2 + self.dropout_layer(self.fusion_proj(cat2))  # Residual

        # --- Step 3: Siamese Self Attention (各自整理内部信息) ---
        # 处理 Q1
        q1_norm = self.norm_self(q_t1)
        out_1 = self.self_attn(query=q1_norm, key=q1_norm, value=q1_norm)
        q_t1 = q_t1 + self.dropout_layer(out_1)

        # 处理 Q2 (共享权重)
        q2_norm = self.norm_self(q_t2)
        out_2 = self.self_attn(query=q2_norm, key=q2_norm, value=q2_norm)
        q_t2 = q_t2 + self.dropout_layer(out_2)

        # --- Step 4: Siamese FFN ---
        # 处理 Q1
        q1_norm = self.norm_ffn(q_t1)
        q_t1 = q_t1 + self.dropout_layer(self.ffn(q1_norm))

        # 处理 Q2
        q2_norm = self.norm_ffn(q_t2)
        q_t2 = q_t2 + self.dropout_layer(self.ffn(q2_norm))

        return q_t1, q_t2



def print_model_statistics(model):
    print(f"\n{'=' * 20} Model Parameter Statistics {'=' * 20}")
    print(f"{'Module Name':<25} | {'Total Params':<15} | {'Trainable':<10}")
    print("-" * 60)

    total_params = 0
    total_trainable = 0

    # 遍历主模块的直接子模块 (vision_encoder, blocks, heads, etc.)
    for name, child in model.named_children():
        num_params = sum(p.numel() for p in child.parameters())
        num_trainable = sum(p.numel() for p in child.parameters() if p.requires_grad)

        total_params += num_params
        total_trainable += num_trainable

        print(f"{name:<25} | {num_params:<15,} | {str(num_trainable == num_params)}")

    print("-" * 60)
    print(f"{'Total':<25} | {total_params:<15,} | {total_trainable:<15,}")

    # 计算模型权重占用的显存大小 (FP32 = 4 bytes)
    model_size_mb = total_params * 4 / (1024 ** 2)
    print(f"Estimated Model Size (Weights only): {model_size_mb:.2f} MB")
    print(f"{'=' * 64}\n")


# --- 使用示例 ---
# 假设 model 已经实例化
# model = AssembledFusionModel(cfg)
# print_model_statistics(model)


# =================================================================
# 测试代码
# =================================================================
if __name__ == "__main__":
    from utils.config import Config  # 假设存在
    # 模拟 Config 类
    class MockConfig:
        class model:
            class transformer_block:
                query_dim = 1024
                vision_dim = 4096
                num_heads = 16
                mlp_ratio = 4.0
                dropout = 0.1
                norm_layer = "RMSNorm"
                act_layer = "GELU"
                use_qk_norm = True
                fusion_type = "cross_attn"
                if_q_position = False
                if_k_position = True
                if_v_position = True

        model = model()


    cfg = MockConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 初始化模型
    model_block = FusionTransformerBlock2(cfg, if_query=True).to(device)
    print_model_statistics(model_block)





    # # 创建数据
    # B, L, D_q, D_v = 2, 10, 128, 256
    # q1 = torch.randn(B, L, D_q).to(device)
    # q2 = torch.randn(B, L, D_q).to(device)  # 第二个时相的 query
    # vis1 = torch.randn(B, 20, D_v).to(device)
    # vis2 = torch.randn(B, 20, D_v).to(device)
    #
    # print("-" * 30)
    # print("Testing Fusion Type: cross_attn")
    # out1, out2 = model_block(q1, q2, vis1, vis2)
    # print(f"Output shape: {out2.shape}")  # 预期 [2, 10, 128]
