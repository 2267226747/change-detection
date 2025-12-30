import torch
import torch.nn as nn


class QueryGenerator(nn.Module):
    def __init__(self, cfg):
        """
        Args:
            cfg: 全局配置对象 (cfg.model.query_token)
        """
        super().__init__()

        cfg = cfg.model.query_token
        self.num_tasks = getattr(cfg, 'task_nums', 4)
        self.tokens_per_task = getattr(cfg, 'tokens_per_task', 256)
        self.dim = getattr(cfg, 'dim', 1024)
        self.batch_size = getattr(cfg, 'batch_size', 16)

        # --- 组件 1: 语义属性 Embedding (Group Attribute) ---
        # 形状: [4, D]
        # 作用: 决定大方向 (建筑/绿化...)
        # 策略: 正交初始化 (Orthogonal) - 确保不同组之间差异最大化
        self.group_embed = nn.Parameter(torch.empty(self.num_tasks, self.dim))
        nn.init.orthogonal_(self.group_embed)

        # --- 组件 2: 个体 Embedding (Individual Variance) ---
        # 形状: [4, 256, D]
        # 作用: 决定个体差异 (关注纹理? 关注边缘? 关注左上角?)
        # 策略: 随机初始化 (Random Normal) - 初始值要小，依附于 Group
        self.token_embed = nn.Parameter(torch.randn(self.num_tasks, self.tokens_per_group, self.dim) * 0.02)

    def get_queries(self):
        """
        组合 Group 和 Token Embedding 生成最终 Query
        """
        # 1. 扩展 Group Embed: [4, D] -> [4, 1, D] -> [4, 256, D]
        # 利用 broadcasting 让组内共享同一个语义中心
        group_semantic = self.group_embed.unsqueeze(1).expand(-1, self.tokens_per_task, -1)

        # 2. 叠加个体差异
        # Q = Base + Delta
        # [4, 256, D]
        final_query = group_semantic + self.token_embed

        # 3. 展平为 transformer 需要的形状
        # [4, 256, D] -> [1024, D]
        final_query = final_query.reshape(-1, self.dim)

        # 4. 扩展 Batch 维度
        # [B, token_nums, D]
        return final_query.unsqueeze(0).repeat(self.batch_size, 1, 1)