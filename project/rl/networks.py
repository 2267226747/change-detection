import torch
import torch.nn as nn
import numpy as np


class MultiModalFeatureExtractor(nn.Module):
    def __init__(self, config, env_shapes):
        """
        特征提取器：负责将 Env 返回的 Dict Obs 编码为统一的 Hidden State
        """
        super().__init__()

        self.num_groups = env_shapes['num_groups']
        self.tokens_per_group = env_shapes['tokens_per_group']
        hidden_dim = config.hidden_dim

        # 1. Query Encoder (处理主要状态)
        # 输入: [B, N, 2D] -> 聚合为 [B, G, 2D] -> MLP -> [B, G, H]
        query_input_dim = env_shapes['query_dim'] * 2
        self.query_mlp = nn.Sequential(
            nn.Linear(query_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        # 2. Vision Encoder (处理全局上下文)
        # 输入: [B, D_vis] -> MLP -> [B, H]
        self.vision_mlp = nn.Sequential(
            nn.Linear(env_shapes['vision_dim'], hidden_dim),
            nn.LayerNorm(hidden_dim),  # 可选
            nn.GELU()
        )

        # 3. Context Encoder (处理 Entropy 和 Time)
        # 输入维度: Total_Subtasks (Entropy) + Total_Subtasks (Probs) + 1 (Time)
        context_input_dim = env_shapes['total_subtasks'] * 2 + 1
        self.context_mlp = nn.Sequential(
            nn.Linear(context_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        # 4. Fusion Layer (融合多模态特征)
        # 输入: Query(H) + Vision(H) + Context(H) = 3H
        self.fusion_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

    def forward(self, obs):
        """
        Args:
            obs: Dict form env._get_observation()
        Returns:
            fused_feat: [B, Groups, Hidden] 保留 Group 维度用于 Correction 动作
            flat_feat:  [B, Groups * Hidden] 展平特征用于 Stop 动作和 Value
        """
        # A. Query 处理: [B, N, 2D] -> [B, G, TPG, 2D] -> Mean -> [B, G, 2D]
        B = obs['query_state'].shape[0]
        q_grouped = obs['query_state'].view(B, self.num_groups, self.tokens_per_group, -1)
        q_pooled = q_grouped.mean(dim=2)
        q_embed = self.query_mlp(q_pooled)  # [B, G, H]

        # B. Vision 处理: [B, D] -> [B, H] -> Broadcast -> [B, G, H]
        v_embed = self.vision_mlp(obs['vision_context'])
        v_embed = v_embed.unsqueeze(1).expand(-1, self.num_groups, -1)

        # C. Context 处理: [B, Total] cat [B, 1] -> [B, H] -> Broadcast -> [B, G, H]
        ctx_data = torch.cat([obs['entropy'], obs['probs'], obs['time']], dim=-1)
        c_embed = self.context_mlp(ctx_data)
        c_embed = c_embed.unsqueeze(1).expand(-1, self.num_groups, -1)

        # D. Fusion
        # cat dim=-1: [B, G, 3H]
        combined = torch.cat([q_embed, v_embed, c_embed], dim=-1)
        fused_feat = self.fusion_mlp(combined)  # [B, G, H]

        # Flatten for global tasks
        flat_feat = fused_feat.reshape(B, -1)  # [B, G*H]

        return fused_feat, flat_feat


class ActorCriticNetwork(nn.Module):
    def __init__(self, config, env_shapes):
        super().__init__()
        self.feature_extractor = MultiModalFeatureExtractor(config, env_shapes)
        hidden_dim = config.hidden_dim

        # --- Actor Heads ---

        # 1. Correction Head (Group-wise Continuous)
        # 输入: [B, G, H] -> 输出: Mean [B, G, D]
        # 注意：这里输出 D 而不是 2D，因为我们是对 Query 的 Embedding 进行修正
        self.correction_mean = nn.Sequential(
            nn.Linear(hidden_dim, env_shapes['query_dim']),
            nn.Tanh()  # Action Scaling: 限制输出在 [-1, 1]
        )
        # LogStd 是可学习参数，初始值设为 0 (std=1) 或 -0.5, []
        # 优化：Per-Group LogStd
        self.correction_logstd = nn.Parameter(
            torch.ones(1, env_shapes['num_groups'], env_shapes['query_dim']) * -0.5
        )

        # 2. Stop Head (Global Discrete)
        # 输入: [B, G*H] -> MLP -> 输出: Logits [B, Total_Subtasks]
        # 修改 Stop Head 输入维度：Hidden特征 + 显式的强信号
        stop_input_dim = (hidden_dim * env_shapes['num_groups']) + (env_shapes['total_subtasks'] * 2)

        self.stop_head = nn.Sequential(
            nn.Linear(stop_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, env_shapes['total_subtasks'])
        )

        # --- Critic Head ---

        # Value Head (Global Scalar)
        # 输入: [B, G*H] -> MLP -> 输出: Value [B, 1]
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim * env_shapes['num_groups'], hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

        # 权重初始化 (对于 RL 非常重要)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # Orthogonal 初始化通常对 PPO 收敛有帮助
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0)

    def forward(self, obs):
        """
        一次性返回所有分布参数，供 Agent 采样使用
        """
        # 提取特征
        # group_feat: [B, G, H], flat_feat: [B, G*H]
        group_feat, flat_feat = self.feature_extractor(obs)

        # 1. Correction 分布参数
        corr_mean = self.correction_mean(group_feat)  # [B, G, D]
        # 注意: 这里 corr_mean 已经在 [-1, 1] 之间
        # 真正的 scaling (e.g. * 0.1) 将在 Agent.get_action 中执行

        corr_logstd = self.correction_logstd.expand_as(corr_mean)  # [B, G, D]

        # 2. Stop 分布参数
        # --- 优化：拼接原始 Context 信息 ---
        # explicit_signals: [B, Total*2]
        explicit_signals = torch.cat([obs['entropy'], obs['probs']], dim=-1)
        # stop_input: [B, G*H + Total*2]
        stop_input = torch.cat([flat_feat, explicit_signals], dim=-1)
        stop_logits = self.stop_head(stop_input)

        # 3. State Value
        value = self.value_head(flat_feat)  # [B, 1]

        return corr_mean, corr_logstd, stop_logits, value
