import torch
import torch.nn as nn


class SubTaskHead(nn.Module):
    def __init__(self, in_dim, mid_dim, dropout=0.1):
        """
        标准的分类头结构
        in_dim: 输入维度 (通常是 hidden_dim * 2)
        mid_dim: 中间层维度 (可以设为 hidden_dim // 2 或者 hidden_dim)
        """
        super().__init__()
        self.net = nn.Sequential(
            # 第一层：特征提取与解耦
            nn.Linear(in_dim, mid_dim),
            nn.LayerNorm(mid_dim),  # 稳定分布，加速收敛
            nn.GELU(),  # 引入非线性
            nn.Dropout(dropout),  # 防止过拟合
            # 第二层：最终分类
            nn.Linear(mid_dim, 1)  # 输出 Logits
        )

    def forward(self, x):
        return self.net(x)


class MultitaskClassifier(nn.Module):
    def __init__(self, cfg, i, logger):
        """
        Args:
            cfg: 全局配置对象 (cfg.model.class_head)
        """
        super().__init__()

        # 1. 获取配置节点
        head_cfg = cfg.model.class_head

        # 2. 从配置中解析参数
        self.in_dim = getattr(head_cfg, 'query_dim', 1024)
        self.hidden_dim = getattr(head_cfg, 'hidden_dim', 1024)
        self.mid_hidden_dim = getattr(head_cfg, 'mid_hidden_dim', 256)
        self.dropout = getattr(head_cfg, 'dropout', 0.1)

        # 3. 解析任务字典列表
        # 结构示例: [{road: 8}, {building: 8}, ...]
        self.group_names = []
        self.sub_counts = []

        for item in head_cfg.task_dicts:
            # item 是一个只有一个键值对的字典，例如 {'road': 8}
            # 为了兼容性，我们获取第一个键和值
            if isinstance(item, dict):
                key = list(item.keys())[0]
                val = list(item.values())[0]
            else:
                # 假如 cfg 库把这个转成了非 dict 对象 (如 omegaconf ListConfig)
                # 这里假设它依然像字典一样可迭代，或者根据具体库调整
                key = list(item.keys())[0]
                val = item[key]

            if i == 0:
                logger.info(f"Task group name {key}, Sub task nums: {val}")
            self.group_names.append(key)
            self.sub_counts.append(val)

        self.num_groups = len(self.group_names)  # e.g., 4

        # 4. 局部投影 (Local Projection / MLP)
        # 输入维度: 4 * in_dim (因为 forward 中做了 concat: t1, t2, diff, prod)
        fusion_dim = self.in_dim * 4

        self.group_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(fusion_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.GELU(),
                nn.Dropout(self.dropout)
            ) for _ in range(self.num_groups)
        ])

        # 5. 细分类头 (Sub-class Heads)
        # Heads[Group_ID][Sub_Class_ID]
        self.sub_heads = nn.ModuleList()

        for count in self.sub_counts:
            # 对于每个大类，创建 count 个小分类头
            # 输入维度: hidden_dim * 2 (因为后面做了 Max+Avg 聚合)
            group_heads = nn.ModuleList([
                # 这里替换了简单的 nn.Linear
                SubTaskHead(
                    in_dim=self.hidden_dim * 2,
                    mid_dim=self.mid_hidden_dim,
                    dropout=self.dropout
                )
                for _ in range(count)
            ])
            self.sub_heads.append(group_heads)
        logger.info(f"Build MultitaskClassifier {i // 2 + 1}")

    def forward(self, q_t1, q_t2):
        """
        Input: [B, N, D] (例如 [B, 1024, 1024])
        Output: Dict { "building": [Logits_Sub1, ...], ... }
        """
        B, N, D = q_t1.shape

        # 确保 token 数量可以被组数整除 (1024 // 4 = 256)
        if N % self.num_groups != 0:
            raise ValueError(f"Input tokens ({N}) cannot be evenly divided by groups ({self.num_groups})")

        tokens_per_group = N // self.num_groups

        # --- Step 1: 特征构造 ---
        diff = torch.abs(q_t1 - q_t2)
        prod = q_t1 * q_t2
        # [B, N, 4D]
        raw_feat = torch.cat([q_t1, q_t2, diff, prod], dim=-1)

        # Reshape 分组: [B, 4, 256, 4D]
        feat_grouped = raw_feat.view(B, self.num_groups, tokens_per_group, -1)

        results = {}

        # 遍历每个 Group
        for g_idx in range(self.num_groups):
            group_name = self.group_names[g_idx]

            # --- Step 2: 局部投影 ---
            # [B, 256, 4D]
            g_input = feat_grouped[:, g_idx, :, :]

            # [B, 256, Hidden]
            g_semantic_feat = self.group_mlps[g_idx](g_input)

            # --- Step 3: 全局聚合 ---
            # [重要] 聚合操作涉及累加，必须转为 float32 以防止 BF16 精度丢失
            g_semantic_feat_fp32 = g_semantic_feat.float()
            # [B, Hidden]
            g_max = torch.max(g_semantic_feat_fp32, dim=1)[0]
            g_avg = torch.mean(g_semantic_feat_fp32, dim=1)
            # [B, Hidden*2]
            g_summary = torch.cat([g_max, g_avg], dim=-1)

            # --- Step 4: 细分任务分类 ---
            sub_results = []
            current_group_heads = self.sub_heads[g_idx]

            for head in current_group_heads:
                # [B, 1]
                logits = head(g_summary)
                sub_results.append(logits.float())

            results[group_name] = sub_results

        return results


# ==========================================
# 测试代码示例
# ==========================================
if __name__ == "__main__":
    from types import SimpleNamespace

    # 模拟 Config 结构
    # 通常使用 OmegaConf 或 Hydra，这里用 SimpleNamespace 模拟
    cfg = SimpleNamespace(
        model=SimpleNamespace(
            class_head=SimpleNamespace(
                task_dicts=[
                    {'road': 8},
                    {'building': 8},
                    {'greenery': 7},
                    {'infrastructure': 8}
                ],
                query_dim=1024,
                hidden_dim=512,
                mid_hidden_dim=256,
                drop_out=0.1
            )
        )
    )

    # 实例化模型
    model = MultitaskClassifier(cfg)

    # 打印模型结构验证
    print(f"Groups: {model.group_names}")
    print(f"Sub-tasks counts: {model.sub_counts}")
    print(f"Number of groups: {model.num_groups}")

    # 模拟输入 [Batch, Tokens, Dim]
    # Tokens 必须是 4 的倍数 (如 1024)
    dummy_q1 = torch.randn(2, 1024, 1024)
    dummy_q2 = torch.randn(2, 1024, 1024)

    # Forward
    output = model(dummy_q1, dummy_q2)

    # 验证输出
    print("\nOutput Check:")
    for k, v in output.items():
        print(f"Group: {k}, Num Tasks: {len(v)}, Logits Shape: {v[0].shape}")
