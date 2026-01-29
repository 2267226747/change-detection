# Step 2: 继承 BaseTrainer，冻结 DL，训练 RL Agent
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from collections import defaultdict, deque
import os
import time

# 假设使用 tensorboard 记录
from torch.utils.tensorboard import SummaryWriter


class PPOTrainer:
    def __init__(self, config, agent, env, buffer, reward_calculator, train_loader, val_loader=None):
        """
        PPO 训练管理器
        Args:
            config: 全局配置
            agent: PPOAgent 实例
            env: TransformerRLEnv 实例
            buffer: RolloutBuffer 实例
            reward_calculator: RewardCalculator 实例
            train_loader: 训练集 DataLoader (需支持无限迭代或自动重置)
            val_loader: 验证集 DataLoader
        """
        self.config = config
        self.agent = agent
        self.env = env
        self.buffer = buffer
        self.reward_calc = reward_calculator
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.device = config.device
        self.writer = SummaryWriter(log_dir=config.log_dir)

        # 将 DataLoader 转为迭代器，以便在 RL 循环中按需获取
        self.train_iter = iter(self.train_loader)

        # 统计指标容器
        self.ep_rewards = deque(maxlen=100)
        self.ep_lengths = deque(maxlen=100)
        self.ep_accuracies = deque(maxlen=100)

    def _get_batch_data(self):
        """从 DataLoader 获取下一个 Batch，如果耗尽则重置"""
        try:
            batch = next(self.train_iter)
        except StopIteration:
            self.train_iter = iter(self.train_loader)
            batch = next(self.train_iter)
        return batch

    def collect_rollouts(self):
        """
        收集经验 (Rollout Phase)
        运行 env.reset() -> 跑 num_steps 步 -> 存入 Buffer
        """
        self.buffer.reset()
        self.agent.eval()  # 收集数据时不更新 BatchNorm

        # 1. 获取新的一批图像数据并 Reset 环境
        batch_data = self._get_batch_data()

        # 将 Labels 移到 Device (Reward计算需要)
        # 假设 batch_data['labels'] 是 [B, Total]
        batch_labels = batch_data['labels'].to(self.device)

        # Reset Env (会执行 Pre-rollout)
        obs = self.env.reset(batch_data)

        # 统计当前 Rollout 的总奖励
        current_ep_reward = torch.zeros(self.env.batch_size, device=self.device)

        for step in range(self.config.num_steps):
            with torch.no_grad():
                # 1. Agent 决策
                # return: dict_action, raw_corr, stop, log_prob, value
                action, raw_corr, stop, log_prob, value = self.agent.get_action(obs)

                # 2. Env 执行
                next_obs, _, dones, info = self.env.step(action)

                # 3. 计算 Reward (使用 info 中的 pre_action_mask)
                # 注意：传入 Env 中锁定的 final_logits
                rewards, r_info = self.reward_calc.compute_reward(
                    logits=self.env.final_logits.detach(),  # 显式 detach 增强安全性,
                    labels=batch_labels,
                    stop_decision=stop,
                    pre_action_mask=info['pre_action_mask'],  # [Key] 使用旧 Mask
                    done_mask=dones
                )

            # 记录统计
            current_ep_reward += rewards

            # 4. 存入 Buffer
            # 注意：buffer 需要存 raw_correction 用于后续计算分布
            buffer_action = {'correction': raw_corr, 'stop': stop}
            # 修正后
            # 确保 step_indices 是一个形状为 [Batch] 的 Tensor
            # 假设当前 step 对所有 batch 样本都是一样的
            step_indices_tensor = torch.full((self.env.batch_size,), info['step'], dtype=torch.long, device='cpu')

            self.buffer.add(
                obs=obs,
                action=buffer_action,
                log_prob=log_prob,
                reward=rewards,
                done=dones,
                value=value,
                cls_input_q=info['cls_input_q'],
                labels=batch_labels,
                step_indices=step_indices_tensor
            )

            # 更新 Obs
            obs = next_obs

            # 如果所有样本都 Done 了，可以提前跳出当前 Rollout 循环 (可选)
            # 或者让 Env 内部处理 Dummy Step (通常 VectorEnv 会自动 Reset，但这里是单 Batch Env)
            # 鉴于我们的 Env 是处理固定步数 (max_steps)，这里循环通常会跑满 config.num_steps
            # 除非 max_steps < config.num_steps，这里假设 config.num_steps == env.max_steps

        # Rollout 结束，计算 GAE
        # 需要最后一个状态的 Value 来做 Bootstrap
        with torch.no_grad():
            _, _, _, last_value = self.agent.network(next_obs)  # 只取 Value

        self.buffer.compute_returns_and_advantage(last_value, dones)

        # 记录 Episode 统计数据
        self.ep_rewards.append(current_ep_reward.mean().item())
        if 'reward/settled_acc' in r_info:
            self.ep_accuracies.append(r_info['reward/settled_acc'])

    def update(self):
        """
        PPO 更新 (Update Phase + Joint Cls Training)
        """
        self.agent.train()  # 开启 Dropout 等

        loss_metrics = defaultdict(list)

        # PPO Epochs (同一个 Batch 数据更新多次)
        for _ in range(self.config.ppo_epochs):
            data_generator = self.buffer.get_generator()

            for batch in data_generator:
                # Agent 内部执行 Forward -> Ratio -> Clip Loss -> Backward
                metrics = self.agent.update(batch)

                for k, v in metrics.items():
                    loss_metrics[k].append(v)

        # 平均 Loss
        avg_metrics = {k: np.mean(v) for k, v in loss_metrics.items()}
        return avg_metrics

    def train(self):
        """主训练循环"""
        print(f"Start Training on {self.device}...")

        global_step = 0

        for epoch in tqdm(range(self.config.total_epochs)):
            # 1. 收集数据
            self.collect_rollouts()

            # 2. 更新模型
            train_metrics = self.update()

            # 3. 日志记录
            global_step += self.config.num_steps * self.env.batch_size

            # 记录 Loss
            for k, v in train_metrics.items():
                self.writer.add_scalar(k, v, epoch)

            # 记录 Rewards & Metrics
            if len(self.ep_rewards) > 0:
                self.writer.add_scalar('perf/ep_reward', np.mean(self.ep_rewards), epoch)
                self.writer.add_scalar('perf/accuracy', np.mean(self.ep_accuracies), epoch)

            # 4. 保存模型
            if (epoch + 1) % self.config.save_interval == 0:
                self.save_checkpoint(epoch)

            # 5. (可选) 验证集评估
            if (epoch + 1) % self.config.eval_interval == 0 and self.val_loader:
                self.evaluate(epoch)

    def save_checkpoint(self, epoch):
        path = os.path.join(self.config.ckpt_dir, f"checkpoint_ep{epoch}.pt")
        torch.save({
            'epoch': epoch,
            'agent_state_dict': self.agent.state_dict(),
            # 如果 classifier 是单独微调的，它的参数变化反映在 model.class_heads 中
            # 这里保存一份引用方便查看，实际恢复时通常加载整个 agent 或原始 model
            'classifier_state_dict': self.env.model.class_heads.state_dict(),
            'optimizer_state_dict': self.agent.optimizer.state_dict(),
            'config': self.config,
        }, path)
        print(f"Saved checkpoint to {path}")

    def evaluate(self, epoch):
        """验证逻辑 (简略版)"""
        # 类似于 collect_rollouts，但不更新 buffer，不训练，只记录准确率
        pass