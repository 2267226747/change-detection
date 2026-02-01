# Step 2: 继承 BaseTrainer，冻结 DL，训练 RL Agent
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from collections import defaultdict, deque
import os
import time
from utils.logger import setup_logger

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

        self.logger = setup_logger(config.log_dir)

        self.device = config.device
        self.writer = SummaryWriter(log_dir=config.writer_dir)

        # 将 DataLoader 转为迭代器，以便在 RL 循环中按需获取
        self.train_iter = iter(self.train_loader)

        # 统计指标容器
        self.train_perf_metrics = defaultdict(lambda: deque(maxlen=100))

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
            if dones.all():
                break

        # Rollout 结束，计算 GAE
        # 需要最后一个状态的 Value 来做 Bootstrap
        with torch.no_grad():
            _, _, _, last_value = self.agent.network(next_obs)  # 只取 Value

        # 记录 Reward (取 Batch 平均)
        self.train_perf_metrics['reward'].append(current_ep_reward.mean().item())

        # 解析 r_info 中的指标
        # 假设 r_info 包含: {'reward/settled_acc': 0.8, 'reward/settled_f1': 0.7, ...}
        for k, v in r_info.items():
            if 'finalall_' in k:
                # 提取名称: 'reward/settled_f1' -> 'f1'
                metric_name = k.split('finalall_')[-1]
                self.train_perf_metrics[metric_name].append(v)


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
        self.logger.info(f"Start Training on {self.device}...")

        global_step = 0

        for epoch in tqdm(range(self.config.total_epochs), desc="Training"):
            # 1. 收集数据
            self.collect_rollouts()

            # 2. 更新模型
            train_metrics = self.update()

            # 3. 日志记录
            global_step += self.config.num_steps * self.env.batch_size

            # 构建日志信息
            loss_msg_parts = [f"Epoch {epoch}"]
            # 记录 Loss
            for k, v in train_metrics.items():
                self.writer.add_scalar(k, v, epoch)
                loss_msg_parts.append(f"{k}: {v:.4f}")

            # 记录 Rewards & Metrics
            perf_msg_parts = []
            # 遍历 perf_metrics 中的所有 key (reward, acc, f1, precision...)
            for k, v_deque in self.train_perf_metrics.items():
                if len(v_deque) > 0:
                    # 使用滑动平均写入 TensorBoard
                    mean_val = np.mean(v_deque)
                    self.writer.add_scalar(f'perf/{k}', mean_val, epoch)
                    perf_msg_parts.append(f"{k}: {mean_val:.4f}")

            # 统一输出 Training Loop 日志
            self.logger.info(" | ".join(loss_msg_parts))
            if perf_msg_parts:
                self.logger.info(f"Train Metrics: {' | '.join(perf_msg_parts)}")

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
        self.logger.info(f"Saved checkpoint to {path}")

    def evaluate(self, epoch):
        """验证逻辑"""
        if not self.val_loader:
            return

        self.logger.info(f"Evaluating at epoch {epoch}...")
        self.agent.eval()
        # 如果 env 中有这就切换，没有就不管 (Transformer backbone 通常一直 eval)
        if hasattr(self.env, 'eval'): self.env.eval()

        # 使用 defaultdict 存储所有出现的指标
        # Key 例如: 'reward', 'steps', 'acc', 'f1', 'precision'...
        val_metrics = defaultdict(list)


        # 遍历验证集
        with torch.no_grad():
            # 使用 tqdm 显示进度
            pbar = tqdm(self.val_loader, desc=f"Eval Ep{epoch}", leave=False)
            for batch_data in pbar:
                # 数据准备
                # 假设 batch_data['labels'] 是 [B, Total]
                batch_labels = batch_data['labels'].to(self.device)

                # Reset Env
                obs = self.env.reset(batch_data)

                # 统计容器
                batch_rewards = torch.zeros(self.env.batch_size, device=self.device)
                steps_taken = torch.zeros(self.env.batch_size, device=self.device)  # 记录每个样本跑了多少步
                active_mask = torch.ones(self.env.batch_size, dtype=torch.bool, device=self.device)  # 记录样本是否还在跑

                # Rollout loop
                for step in range(self.config.num_steps):
                    # Agent 决策
                    action, _, stop, _, _ = self.agent.get_action(obs, deterministic=True)

                    # Env 执行
                    next_obs, _, dones, info = self.env.step(action)

                    # 3. 记录步数 (只要还没 done，步数就+1)
                    # 注意：dones 是 [B]，表示该样本所有任务是否都结束
                    # 如果样本还在跑 (active)，这一步算作有效消耗
                    steps_taken[active_mask] += 1
                    active_mask = ~dones  # 更新活跃状态

                    # 4. 计算 Reward (仅作记录)
                    # 注意：这里我们信任 RewardCalculator 的逻辑，但 Accuracy 我们自己算更准
                    # 计算 Reward (用于统计指标)
                    rewards, r_info = self.reward_calc.compute_reward(
                        logits=self.env.final_logits.detach(),
                        labels=batch_labels,
                        stop_decision=stop,
                        pre_action_mask=info['pre_action_mask'],
                        done_mask=dones
                    )

                    batch_rewards += rewards
                    obs = next_obs

                    # 5. [优化] 提前退出：如果所有样本都结束了，不需要空跑
                    if dones.all():
                        break

                # --- Batch 结算 ---

                # 1. 记录基础指标
                val_metrics['reward'].append(batch_rewards.mean().item())
                val_metrics['avg_steps'].append(steps_taken.mean().item())

                # 2. 从 final_step_info 中提取高阶指标 (Acc, F1, Rec...)
                # 假设 r_info key 格式为 "reward/settled_acc" 或 "settled_acc"
                current_acc = 0.0  # 用于进度条显示

                for k, v in r_info.items():
                    # 过滤掉不需要记录的中间变量，只保留 settled 指标
                    if 'finalall_' in k:
                        # 清洗 key 名称: 'reward/finalall_acc' -> 'acc'
                        metric_name = k.split('finalall_')[-1]
                        val_metrics[metric_name].append(v)

                        # 顺便获取 acc 用于显示
                        if 'acc' in metric_name:
                            current_acc = v
                # 更新进度条
                pbar.set_postfix({
                    'acc': f"{current_acc:.3f}",
                    'rew': f"{batch_rewards.mean().item():.2f}"
                })

            # --- 汇总与日志 ---

            # 计算所有指标的平均值
            avg_metrics = {k: np.mean(v) for k, v in val_metrics.items()}

            # 打印关键信息 (确保 print 不会报错，使用 get 设置默认值)
            self.logger.info(f"Eval Ep {epoch}: "
                  f"Reward={avg_metrics.get('reward', 0):.4f}, "
                  f"Acc={avg_metrics.get('acc', 0):.4f}, "
                  f"F1={avg_metrics.get('f1', 0):.4f}, "
                  f"Steps={avg_metrics.get('avg_steps', 0):.2f}")

            # 写入 TensorBoard
            for k, v in avg_metrics.items():
                self.writer.add_scalar(f'eval/{k}', v, epoch)

            # 恢复训练模式
            self.agent.train()

            return avg_metrics
