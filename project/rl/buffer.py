import torch
import numpy as np
from collections import defaultdict


class RolloutBuffer:
    def __init__(self, config, device):
        """
        PPO Rollout Buffer，支持 Dict Observation 和 Dict Action
        """
        self.buffer_size = config.num_steps  # 每次更新前收集的步数
        self.batch_size = config.batch_size  # PPO 更新时的 mini-batch 大小
        self.gamma = config.gamma
        self.gae_lambda = config.gae_lambda
        self.device = device

        # 存储容器
        # 使用 List 存储每一步的数据，最后 stack 起来
        # 这种方式比预先分配 Tensor 更灵活，特别是对于 Dict 结构
        self.reset()

    def reset(self):
        """清空 Buffer"""
        self.obs_storage = defaultdict(list)
        self.action_corr_storage = []  # Correction 动作
        self.action_stop_storage = []  # Stop 动作
        self.logprobs_storage = []
        self.rewards_storage = []
        self.dones_storage = []
        self.values_storage = []
        self.cls_query_storage = []
        self.labels_storage = []
        self.step_indices_storage = []

        self.ptr = 0  # 当前存储指针

    def add(self, obs, action, log_prob, reward, done, value, cls_input_q, labels, step_indices):
        """
        存储一个 Step 的数据
        假设输入的数据已经是 Batched 的 [B, ...]
        """
        # 1. 处理 Dict Observation
        for k, v in obs.items():
            # 这里的 v 是 Tensor， detach 并转到 CPU 以节省显存
            self.obs_storage[k].append(v.detach().cpu())

        # 2. 处理 Dict Action
        self.action_corr_storage.append(action['correction'].detach().cpu())
        self.action_stop_storage.append(action['stop'].detach().cpu())

        # 3. 其他标量/Tensor
        self.logprobs_storage.append(log_prob.detach().cpu())
        self.rewards_storage.append(reward.detach().cpu())
        self.dones_storage.append(done.detach().cpu())  # done 是 bool 或 float
        self.values_storage.append(value.detach().cpu())

        # 专门存一个用于分类训练的 Query
        self.cls_query_storage.append(cls_input_q.detach().cpu())
        self.labels_storage.append(labels.detach().cpu())
        self.step_indices_storage.append(step_indices.detach().cpu())

        self.ptr += 1

    def compute_returns_and_advantage(self, last_value, done):
        """
        GAE (Generalized Advantage Estimation) 计算核心
        Args:
            last_value: Rollout 结束后下一个状态的 Value (用于 Bootstrap)
            done: 最后一个状态是否结束
        """
        # 将 list 转为 tensor: [Time, Batch, ...]
        rewards = torch.stack(self.rewards_storage).to(self.device)
        values = torch.stack(self.values_storage).to(self.device)
        dones = torch.stack(self.dones_storage).to(self.device)

        last_value = last_value.to(self.device)

        # 容器
        self.advantages = torch.zeros_like(rewards).to(self.device)
        self.returns = torch.zeros_like(rewards).to(self.device)

        last_gae_lam = 0

        # 倒序遍历时间步
        num_steps = len(self.rewards_storage)

        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                next_non_terminal = 1.0 - done.float().to(self.device)
                next_value = last_value.squeeze(-1)  # 确保维度匹配
            else:
                next_non_terminal = 1.0 - dones[t + 1]
                next_value = values[t + 1].squeeze(-1)

            # Delta = r + gamma * V(s') * mask - V(s)
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t].squeeze(-1)

            # A_t = Delta + gamma * lambda * mask * A_{t+1}
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam

            self.advantages[t] = last_gae_lam

        # Returns = Advantage + Value
        self.returns = self.advantages + values.squeeze(-1)

    def get_generator(self):
        """
        生成器：生成打乱的 Mini-batch 用于 Agent Update
        """
        # 1. 整理所有数据为 Flatten Tensor [T * B, ...]
        # Obs (Dict)
        flat_obs = {}
        for k, v in self.obs_storage.items():
            # stack: [T, B, ...] -> view: [T*B, ...]
            stacked = torch.stack(v).to(self.device)
            flat_obs[k] = stacked.view(-1, *stacked.shape[2:])

        # Actions
        flat_act_corr = torch.stack(self.action_corr_storage).to(self.device)
        flat_act_corr = flat_act_corr.view(-1, *flat_act_corr.shape[2:])

        flat_act_stop = torch.stack(self.action_stop_storage).to(self.device)
        flat_act_stop = flat_act_stop.view(-1, *flat_act_stop.shape[2:])

        # LogProbs & Values & Returns & Advantages
        flat_logprobs = torch.stack(self.logprobs_storage).view(-1).to(self.device)
        flat_values = torch.stack(self.values_storage).view(-1).to(self.device)
        flat_returns = self.returns.view(-1)
        flat_advantages = self.advantages.view(-1)

        # 2. 优势标准化 (Advantage Normalization) - PPO 标准操作
        flat_advantages = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1e-8)

        # Cls Query
        flat_cls_query = torch.stack(self.cls_query_storage).to(self.device)
        # view(-1, *shape[2:]) 可以自动处理除 Time 和 Batch 以外的剩余维度
        flat_cls_query = flat_cls_query.view(-1, *flat_cls_query.shape[2:])

        # Labels
        flat_labels = torch.stack(self.labels_storage).to(self.device)
        flat_labels = flat_labels.view(-1, *flat_labels.shape[2:])

        # Step Indices
        flat_step_indices = torch.stack(self.step_indices_storage).to(self.device)
        flat_step_indices = flat_step_indices.view(-1, *flat_step_indices.shape[2:])

        # 3. 生成 Mini-batch 索引
        total_samples = flat_returns.shape[0]
        indices = np.arange(total_samples)
        np.random.shuffle(indices)

        for start in range(0, total_samples, self.batch_size):
            end = start + self.batch_size
            mb_indices = indices[start:end]

            # 构建 Obs Mini-batch
            mb_obs = {k: v[mb_indices] for k, v in flat_obs.items()}

            yield {
                'obs': mb_obs,
                'actions_corr': flat_act_corr[mb_indices],
                'actions_stop': flat_act_stop[mb_indices],
                'log_probs': flat_logprobs[mb_indices],
                'values': flat_values[mb_indices],
                'returns': flat_returns[mb_indices],
                'advantages': flat_advantages[mb_indices],
                'cls_query': flat_cls_query[mb_indices],
                'labels': flat_labels[mb_indices],
                'step_indices': flat_step_indices[mb_indices]
            }

    def __len__(self):
        return len(self.rewards_storage)
