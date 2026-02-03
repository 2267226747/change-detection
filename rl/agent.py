# PPO/DQN 算法逻辑 (select_action, update)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Bernoulli
from utils.loss import RLLoss


class PPOAgent(nn.Module):
    def __init__(self, network, classifier_heads, classifier_params, config, logger=None):
        """
        PPO Agent: 封装网络、分布构建和 PPO 更新逻辑
        Args:
            network: 实例化好的 ActorCriticNetwork (from network.py)
            classifier_heads: nn.ModuleList (来自 env.model.class_heads)，用于在 update 时重算 logits
            classifier_params: list (来自 env.get_classifier_parameters())，用于优化器
            config: 包含 lr, gamma, clip_param, entropy_coef, etc.
        """
        super().__init__()
        self.network = network
        self.classifier_heads = classifier_heads
        self.rl_config = config.rl

        # 参数初始化
        self.clip_param = getattr(self.rl_config, 'clip_param', 0.2)
        self.use_value_clip = getattr(self.rl_config, 'use_value_clip', True)
        self.value_loss_coef = getattr(self.rl_config, 'value_loss_coef', 0.5)
        self.entropy_coef = getattr(self.rl_config, 'entropy_coef', 0.01)
        self.cls_loss_coef = getattr(self.rl_config, 'cls_loss_coef', 0.5)
        self.max_grad_norm = getattr(self.rl_config, 'max_grad_norm', 0.5)

        # 获取是否冻结分类器的配置
        self.freeze_classifier = getattr(self.rl_config, 'freeze_classifier', False)

        # 1. 构建优化器参数列表
        # 如果冻结分类器，只优化 Policy/Value 网络 (network)
        # 如果不冻结，同时优化 Policy/Value 网络 和 分类头 (classifier_params)
        if self.freeze_classifier:
            logger.info("PPOAgent: Classifier is FROZEN. Only optimizing RL Policy/Value network.")
            all_params = list(self.network.parameters())
        else:
            logger.info("PPOAgent: Classifier is TRAINABLE. Optimizing Jointly.")
            all_params = list(self.network.parameters()) + classifier_params
            self.cls_criterion = RLLoss(config, logger=logger).to(self.device)

        # 优化器
        self.optimizer = torch.optim.AdamW(
            all_params,
            lr=self.rl_config.lr,
            eps=1e-5,
            weight_decay=self.rl_config.weight_decay
        )

        self.action_scale = self.rl_config.action_scale

        # [NEW] 初始化混合精度 Scaler
        # 注意：默认开启 enable=True，如果后续想关掉可以在这里控制
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.rl_config.use_amp)

    def get_action(self, obs, deterministic=False):
        """
        Rollout 阶段使用：根据观测采样动作
        """
        # 1. 前向传播
        # corr_mean: [B, G, D] (Range: -1~1 via Tanh)
        # corr_logstd: [B, G, D]
        # stop_logits: [B, Total]
        # value: [B, 1]
        corr_mean, corr_logstd, stop_logits, value = self.network(obs)

        # 2. 构建分布

        # --- A. Correction (Continuous) ---
        corr_std = corr_logstd.exp()
        dist_corr = Normal(corr_mean, corr_std)

        if deterministic:
            correction_sample = corr_mean
        else:
            correction_sample = dist_corr.sample()

        # [Action Scaling]
        # 网络输出在 [-1, 1]，我们需要将其映射到实际的 Query 修正幅度
        # 例如 action_scale = 0.1，则最大修正量为 0.1
        scaled_correction = correction_sample * self.action_scale

        # --- B. Stop (Discrete) ---
        dist_stop = Bernoulli(logits=stop_logits)

        if deterministic:
            # 阈值 0.5 (logits > 0)
            stop_sample = (stop_logits > 0).float()
        else:
            stop_sample = dist_stop.sample()

        # 3. 计算 Log Probability (用于 Buffer 存储)
        # 假设动作各维度独立，LogProb 求和
        # [B]
        log_prob_corr = dist_corr.log_prob(correction_sample).sum(dim=(1, 2))
        log_prob_stop = dist_stop.log_prob(stop_sample).sum(dim=1)

        total_log_prob = log_prob_corr + log_prob_stop

        # 4. 组装动作字典 (Env 需要的格式)
        action = {
            'correction': scaled_correction,  # 注意：传给 Env 的是缩放后的
            'stop': stop_sample
        }

        # 返回: 动作(Env用), 原始动作(Buffer用, 未缩放), LogProb, Value
        # 注意: Buffer 中最好存 unscaled correction，以便 update 时分布对齐
        return action, correction_sample, stop_sample, total_log_prob, value

    def evaluate_actions(self, obs, raw_correction, stop_decision):
        """
        Update 阶段使用：评估给定的状态和动作，计算新策略下的 LogProb, Entropy, Value
        Args:
            raw_correction: 未缩放的 correction (直接来自 Normal 采样)
        """
        # 1. 前向传播
        corr_mean, corr_logstd, stop_logits, value = self.network(obs)

        # 2. 构建分布
        corr_std = corr_logstd.exp()
        dist_corr = Normal(corr_mean, corr_std)
        dist_stop = Bernoulli(logits=stop_logits)

        # 3. 计算 Log Prob
        # 使用旧的动作 (raw_correction, stop_decision) 在新分布下求概率
        log_prob_corr = dist_corr.log_prob(raw_correction).sum(dim=(1, 2))
        log_prob_stop = dist_stop.log_prob(stop_decision).sum(dim=1)
        total_log_prob = log_prob_corr + log_prob_stop

        # 4. 计算 Entropy (用于正则化，防止过早收敛)
        # Correction 熵
        entropy_corr = dist_corr.entropy().sum(dim=(1, 2))
        # Stop 熵
        entropy_stop = dist_stop.entropy().sum(dim=1)
        total_entropy = entropy_corr + entropy_stop

        return total_log_prob, total_entropy, value

    def update(self, rollouts):
        """
        PPO 核心更新逻辑
        Args:
            rollouts: 一个 Batch 的数据 (来自 Buffer)
        """
        # 解包数据
        obs_batch = rollouts['obs']
        actions_corr_batch = rollouts['actions_corr']  # Unscaled
        actions_stop_batch = rollouts['actions_stop']
        old_log_probs = rollouts['log_probs']
        advantages = rollouts['advantages']
        returns = rollouts['returns']  # Target Value
        labels_batch = rollouts['labels']  # [新增] Buffer 必须存储 Label

        # 2. 使用 autocast 上下文开启混合精度
        # 注意: 这里的 'cuda' 应根据你的设备动态调整
        device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        with torch.amp.autocast(device_type=device_type):
            # 1. 评估当前策略
            new_log_probs, entropy, values = self.evaluate_actions(
                obs_batch, actions_corr_batch, actions_stop_batch
            )

            # 2. 计算 Ratio
            # ratio = exp(new - old)
            ratio = torch.exp(new_log_probs - old_log_probs)

            # 3. 计算 Surrogate Loss (CLIP)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # 4. 计算 Value Loss (MSE)
            # 可选：Value Clip
            if self.use_value_clip:
                old_values = rollouts['values']
                value_pred_clipped = old_values + (values.squeeze(-1) - old_values).clamp(
                    -self.clip_param, self.clip_param
                )
                value_loss_1 = (values.squeeze(-1) - returns).pow(2)
                value_loss_2 = (value_pred_clipped - returns).pow(2)
                value_loss = 0.5 * torch.max(value_loss_1, value_loss_2).mean()
            else:
                value_loss = 0.5 * F.mse_loss(values.squeeze(-1), returns)

            # 5. 计算 Entropy Loss (Bonus)
            entropy_loss = -entropy.mean()  # 我们希望 Entropy 大，所以 Loss 取负

            # ====================================================
            # 2. 计算 Classification Loss (Supervised)
            # ====================================================
            cls_loss = torch.tensor(0.0, device=self.device)
            # [统计容器 1] 用于计算梯度: 加权的总 Loss 和 加权的总元素数
            weighted_cls_loss_sum = torch.tensor(0.0, device=self.device)
            total_weighted_elements = 0.0
            # [统计容器 2] 用于 Info 日志: 加权的子任务 Loss (区分任务)
            per_subtask_weighted_loss_sum = None
            total_weighted_samples = 0.0  # 分母：加权的样本数 (对于所有子任务是共享的)
            # 只有在未冻结分类器 且 有分类头 的情况下才计算
            if (not self.freeze_classifier) and (len(self.classifier_heads) > 0):
                # 我们需要重算 Logits 以获取梯度。
                # 问题：Buffer 中的 Obs 混合了不同 Step 的数据。
                # 解决：利用 obs['time'] 反推 Step Index，分批处理。

                # obs['time'] 是归一化的 [0, 1]，我们需要反解出整数 step_idx
                # 假设 max_steps 已知，或者直接在 Buffer 存 step_idx 可能更方便
                # 这里演示利用 time 反推: step ≈ time * max_steps
                # 为了稳健，建议在 buffer 中增加 'step_idx' 字段

                step_indices = rollouts['step_indices']  # [B]

                # 遍历 Batch 中涉及到的所有 Step
                unique_steps = torch.unique(step_indices)

                for step_idx in unique_steps:
                    idx = int(step_idx.item())
                    if idx >= len(self.classifier_heads): continue

                    # A. 获取当前层权重
                    # 确保 layer_weights 存在且 idx 不越界，否则默认为 1.0
                    if hasattr(self.cls_criterion, 'layer_weights') and idx < len(self.cls_criterion.layer_weights):
                        layer_weight = self.cls_criterion.layer_weights[idx]
                    else:
                        layer_weight = (idx + 1) / len(self.classifier_heads)

                    # B. 数据准备
                    # 找出属于当前 Step 的样本掩码
                    mask = (step_indices == step_idx)

                    # 提取对应的 Query (Query State 包含 t1 和 t2)
                    # obs['query_state']: [B, N, 2D]
                    q_state_masked = obs_batch['query_state'][mask]
                    D = q_state_masked.shape[-1] // 2
                    q_t1 = q_state_masked[:, :, :D]
                    q_t2 = q_state_masked[:, :, D:]

                    # 提取 Label
                    labels_masked = labels_batch[mask]

                    # C. 重新执行分类头 Forward (带梯度)
                    # 注意：输入的 Query 是从 Buffer 取出的 (Detached)，
                    # 所以我们只更新 Classifier 的权重，不更新 Transformer Backbone (符合预期)
                    current_head = self.classifier_heads[idx]
                    logits_dict = current_head(q_t1, q_t2)

                    # # Flatten Logits (需复用 env 中的 flatten 逻辑或在 agent 里重写)
                    # # 这里假设你把 _flatten_logits 变成了一个通用的工具函数或 Agent 的方法
                    # logits_flat = self._flatten_logits(logits_dict)

                    # D. 计算 Loss Matrix [B_subset, N_Tasks]
                    loss_matrix_subset = self.cls_criterion(logits_dict, labels_masked.float())

                    # --- 分支 1: 累加到加权 Loss (用于 Backward) ---
                    # 逻辑: Sum(Loss_i * Weight_step) / Sum(1 * Weight_step)
                    # 使用 sum() 而不是 mean() 避免 Step 间样本不平衡导致的偏差
                    weighted_cls_loss_sum += loss_matrix_subset.sum() * layer_weight
                    total_weighted_elements += loss_matrix_subset.numel() * layer_weight

                    # --- 分支 2: 累加到原始子任务 Loss (用于 Info) ---
                    if per_subtask_weighted_loss_sum is None:
                        # [N_Tasks]
                        per_subtask_weighted_loss_sum = torch.zeros(loss_matrix_subset.shape[1], device=self.device)

                    #这里也乘以 layer_weight
                    # 分子：[N_Tasks] 向量，每个元素是该任务在该 Step 的总 Loss * 权重
                    per_subtask_weighted_loss_sum += loss_matrix_subset.sum(dim=0) * layer_weight
                    # 分母：该 Step 的样本数 * 权重
                    # (所有子任务共享同一个分母)
                    total_weighted_samples += loss_matrix_subset.shape[0] * layer_weight

                # E. 计算最终平均 Cls Loss
                if total_weighted_elements > 0:
                    cls_loss = weighted_cls_loss_sum / total_weighted_elements
                else:
                    cls_loss = torch.tensor(0.0, device=self.device)

            # 6. 总 Loss
            # L = L_policy + c1 * L_value + c2 * L_entropy
            total_loss = (
                    policy_loss
                    + self.value_loss_coef * value_loss
                    + self.entropy_coef * entropy_loss
                    + self.cls_loss_coef * cls_loss.mean()
            )

        # 7. 反向传播
        self.optimizer.zero_grad()
        # 使用 scaler 缩放 loss 并反向传播
        self.scaler.scale(total_loss).backward()

        # 梯度裁剪 (防止爆炸)
        # 在 clip 之前必须先 unscale
        self.scaler.unscale_(self.optimizer)
        nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        
        # 5. 更新参数
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # batch avg
        info = {
            'update/total_loss': total_loss.item(),
            'update/policy_loss': policy_loss.item(),
            'update/value_loss': value_loss.item(),
            'update/org_entropy': entropy.mean().item(),  # 记录正的 entropy
            'update/cls_loss': cls_loss.item(),
            'update/meta/approx_kl': (old_log_probs - new_log_probs).mean().item()  # 监控 KL 散度
        }

        # 计算并记录每个子任务组的【加权】平均 Loss
        if (per_subtask_weighted_loss_sum is not None) and (total_weighted_samples > 0):
            # 计算加权平均值: 加权总和 / 加权样本总数
            # 结果形状: [Total_Subtasks]
            avg_per_subtask_loss = per_subtask_weighted_loss_sum / total_weighted_samples

            # 转为 List
            per_task_loss_list = avg_per_subtask_loss.detach().cpu().tolist()

            # 解析任务组
            for task_name, meta in self.cls_criterion.tasks.items():
                start = meta['start']
                end = meta['end']
                group_losses = per_task_loss_list[start:end]

                if len(group_losses) > 0:
                    for idx, loss in enumerate(group_losses):
                        info[f'update/cls_{task_name}/{idx}'] = loss

        return info


