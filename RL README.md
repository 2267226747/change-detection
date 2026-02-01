# 基于预训练模型的 PPO 强化学习（Change Detection）——项目说明

日期: 2026-01-29

## 项目概述
本项目使用基于预训练深度模型的策略网络，通过 PPO（Proximal Policy Optimization）在序列/步长场景中训练强化学习 Agent，以完成 change-detection / correction + stop 决策任务。系统以“预训练特征提取器 + RL 策略头 + 分类/校正头”的组合形式实现联合训练与优化。

## 代码组织（关键文件）
- rl/
  - agent.py        — PPOAgent 的实现（action 策略、update 接口等）
  - buffer.py       — RolloutBuffer（经验收集、GAE、mini-batch 生成）
  - env.py          — TransformerRLEnv（环境接口：reset, step, final_logits 等）
  - networks.py     — 网络模块（backbone / policy / value / class heads）
  - rewards.py      — RewardCalculator（按策略计算奖励）
- trainer/
  - rl_trainer.py   — PPO 训练控制器（收集 Rollout、调用 buffer、更新 Agent、日志、保存）

## 模型整体结构与数据流
1. 训练入口由 `rl_trainer.PPOTrainer.train()` 控制：
   - 每个 epoch:
     - 调用 `collect_rollouts()`：
       - 从 train_loader 取 batch（包含图像、labels 等），调用 `env.reset(batch_data)`。
       - 在 config.num_steps 步内循环：
         - Agent 根据当前 obs 计算 action（返回 action dict、raw_corr、stop、log_prob、value）。
         - Env 执行 action，返回 next_obs、dones、info（含 pre_action_mask、cls_input_q、step 等）。
         - RewardCalculator 基于 env.final_logits、labels、stop、pre_action_mask 计算 reward。
         - 将 obs/action/logprob/reward/done/value/cls_input_q/labels/step_indices 存入 RolloutBuffer。
       - 用 agent.network(next_obs) 获取最后一步 value，调用 buffer.compute_returns_and_advantage(last_value, dones) 计算 GAE/returns。
     - 调用 `update()`：
       - 从 buffer.get_generator() 获取打乱的 mini-batch（多次 epoch），并交给 agent.update(batch) 做优化。
2. 训练过程中记录 TensorBoard（loss、episode reward、accuracy 等），并按 interval 保存 checkpoint。

## 关键组件详解

### RolloutBuffer（rl/buffer.py）
- 功能
  - 支持 Dict 类型 observation（obs 为 dict）和 Dict/action（action 包含 'correction' 与 'stop' 两分支）。
  - 存储每个时间步的 Batched 张量（假设每步输入为 [B, ...]）。
  - 使用 GAE（Generalized Advantage Estimation）计算 advantages 与 returns。
  - 将时间维和 batch 维合并为扁平化样本 [T*B, ...]，生成随机打乱的 mini-batch 供 PPO 更新。
  - 优势标准化： (adv - mean) / (std + 1e-8)。

- 存储与设备管理
  - add() 中对输入 tensor 调用 `.detach().cpu()` 存储到 list（以节省显存并避免梯度）。
  - get_generator() 中在需要时将堆叠后的 tensor `.to(self.device)`（包括 obs、actions、cls_query、labels、step_indices）。
  - compute_returns_and_advantage() 在 device 上做 GAE 计算，倒序累加 A_t。

- 主要字段（按 add() 传入）
  - obs_storage: dict of list，键对应 obs 中各项。
  - action_corr_storage: list（raw correction）。
  - action_stop_storage: list（stop 分支）。
  - logprobs_storage / rewards_storage / dones_storage / values_storage。
  - cls_query_storage / labels_storage / step_indices_storage（用于联合分类训练或记录 step 信息）。

- GAE 实现要点
  - 对倒序 t:
    - next_non_terminal = 1 - done_{t+1}（最后一步用外部 last done）
    - delta = r_t + gamma * V_{t+1} * next_non_terminal - V_t
    - last_gae_lam = delta + gamma * lambda * next_non_terminal * last_gae_lam
    - advantages[t] = last_gae_lam
  - returns = advantages + values

- mini-batch 生成（get_generator）
  - 将所有 list 通过 torch.stack -> [T, B, ...] -> view(-1, ...) 转为 [T*B, ...]。
  - 随机打乱 indices，按 batch_size 切分并 yield dict：
    - obs: dict（每键 [batch_size, ...]）
    - actions_corr / actions_stop / log_probs / values / returns / advantages
    - cls_query / labels / step_indices

- 注意点
  - buffer 将数据先放 CPU，再在生成 batch 时移动至 device（节省显存、兼容多种 obs 大小）。
  - compute_returns_and_advantage 需要外部提供最后一步 value 及最后 done mask。

### Agent（rl/agent.py）
- 典型接口（在 trainer 中使用）
  - get_action(obs) -> (action_dict, raw_corr, stop, log_prob, value)
    - action_dict 用于 env.step（通常包含分布参数或采样结果）。
    - raw_corr 被 buffer 存储用于后续概率/分布计算。
  - network(next_obs) -> (_, _, _, value) 用于 bootstrap（仅获取 value）。
  - update(batch) 接收 buffer.get_generator() 提供的 mini-batch，执行前向、ratio、clip、损失计算及反向传播并返回 metrics（字典）。

- 优化细节（PPO）
  - 使用 ratio = exp(new_logprob - old_logprob) 与 clip，计算 policy loss。
  - 价值损失与熵正则（常见的 PPO 组件）应在 agent.update 中实现。
  - 可能包含联合分类头的监督损失（cls_query / labels）。

### Environment（rl/env.py）
- 接口约定（在 trainer 中的调用方式）
  - env.reset(batch_data) -> obs（接受 batch 数据以支持批量环境）
  - env.step(action) -> next_obs, reward(s), dones, info
  - env.batch_size 属性用于 batch 处理
  - env.final_logits：用于 reward 计算的模型输出（trainer 将其 detach 传入 reward 计算器）
  - info 通常包含:
    - pre_action_mask: 用于 reward 计算的 mask
    - cls_input_q: 用于分类的 query（buffer 存储）
    - step: 当前时间步标识（用于 step_indices）

### RewardCalculator（rl/rewards.py）
- compute_reward(logits, labels, stop_decision, pre_action_mask, done_mask) -> (rewards, info_dict)
  - 奖励逻辑基于 final_logits 与 ground-truth labels，以及 stop 决策与 pre_action_mask。
  - 返回的 info may 包含额外指标，如 settled_acc（若有）。

### Networks（rl/networks.py）
- 包含 backbone（预训练模型）、policy head、value head、class heads 等。
- Agent 使用网络输出 policy logits、value、class logits 等（具体实现请查看文件）。

## 配置项（config 常见字段）
- num_steps: 每次 rollout 的时间步长度（Rollout 长度 T）
- batch_size: PPO 升级时的 mini-batch 大小（用于 flat 后的样本数）
- ppo_epochs: 每次更新在同一经验上的迭代次数
- gamma: 折扣因子（GAE、returns）
- gae_lambda: GAE 的 lambda
- device: "cuda" / "cpu"
- save_interval / eval_interval / total_epochs / log_dir / ckpt_dir 等

## 输入输出张量形状（约定）
- 单步 obs（每项）传入 add 时形状: [B, ...]
- rewards: [B] 或 [B, 1]（代码在 stack 后与 values squeeze(-1) 对齐）
- values: 常为 [B, 1]，compute 中使用 .squeeze(-1)
- log_probs: [B]（或 [B, 1] 后 view(-1)）
- cls_query: [B, N, D]（具体维度视 networks 而定）
- labels: [B, ...]（classification 标签）
- step_indices: [B]（当前 step index，为 long，存为 CPU）

最终 get_generator 中 yield 的每个字段均为 [mini_batch_size, ...]（除 obs 为 dict）

## 运行步骤（示例）
1. 准备配置 config（num_steps, batch_size, ppo_epochs, gamma, gae_lambda, device 等）。
2. 构建 DataLoader（train_loader）并确保每个 batch 包含至少 'labels'、图像等 env.reset 所需字段。
3. 初始化 env、agent、buffer、reward_calculator：
   - buffer = RolloutBuffer(config, device)
   - trainer = PPOTrainer(config, agent, env, buffer, reward_calc, train_loader, val_loader)
4. 调用 trainer.train() 开始训练。

示例（伪代码）：
```python
from rl.buffer import RolloutBuffer
from trainer.rl_trainer import PPOTrainer

buffer = RolloutBuffer(config, device)
trainer = PPOTrainer(config, agent, env, buffer, reward_calc, train_loader, val_loader)
trainer.train()
```

## 排错与注意事项
- 确保 DataLoader 返回的 labels 在传入 buffer 时已在 CPU（或在 trainer.add 时统一移动到 device/CPU），trainer 中示例把 labels .to(self.device) 后再传入 reward 计算与 buffer.add（buffer 内部会 detach().cpu()）。
- buffer 中的数据会先存至 CPU，再在 get_generator 时转到 device；若系统内存不足，需减小 num_steps 或 batch_size。
- compute_returns_and_advantage 需要最后一步的 value（agent.network(next_obs) 返回）和最后 done mask（trainer 使用 loop 中的 dones）。
- env.reset 接受 batch_data 并返回 obs，应与 agent.get_action 的 obs 输入格式一致（dict）。
- env 或网络输出形状不匹配时，重点检查 value 的形状（是否含最后一维 1），以及 log_probs 的形状。

## 建议改进点（可选）
- buffer 支持预分配 tensor 以加速（当前实现便于 Dict 但频繁 stack/CPU<->GPU 转移）。
- 在 buffer 中显式记录和校验 shapes（便于捕捉不一致 bug）。
- 增加 deterministic evaluation 流程（trainer.evaluate）以记录固定 checkpoint 的性能。
- env_shapes（若在构造函数中存在）可作为成员保存或从签名中移除，避免未使用字段引起困惑。

## 常见问题（FAQ）
- Q: 为什么 buffer 先放 CPU？
  - A: 减少 GPU 显存占用，便于存储不同形状的 dict obs。训练前在 get_generator 再搬回 device。
- Q: GAE 的最后一步如何 bootstrap？
  - A: 使用 agent 对 next_obs 的 value（传入 compute_returns_and_advantage 的 last_value）及最后一条 done mask 判断是否为 terminal。
- Q: 如何联合训练分类 head？
  - A: buffer 提供 cls_query 与 labels；agent.update 应在计算 policy/value loss 的同时加入分类损失（cross-entropy 等）。

---

如需把本 README 写入仓库或将某些说明细化为代码示例（比如 agent.update 的接口或具体 config 样例），可提供相应位置与期望的样式。  