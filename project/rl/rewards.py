# 专门定义 Reward 计算函数 (基于 Loss 或 评价指标)
import torch
import torch.nn as nn


class RewardCalculator:
    def __init__(self, config, env_group_names):
        """
        支持细粒度类别权重的奖励计算器

        Args:
            config: 包含 tasks 配置的对象 (即你提供的 YAML 结构)
            env_group_names: List[str], Env 中定义的 Group 顺序 (e.g., ['road', 'building', ...])
                             必须传入此参数以确保权重顺序与 Logits 顺序一致！
        """
        # 基础标量配置
        self.neg_weight = getattr(config, 'reward_neg_weight', 1.0)  # TN 奖励
        self.wrong_penalty = getattr(config, 'reward_wrong_penalty', -1.0)  # 错误惩罚
        self.time_penalty = getattr(config, 'time_penalty', 0.02)  # 时间惩罚

        # ====================================================
        # 解析并展平 pos_weight
        # ====================================================
        # 目标: 构建一个 [Total_Subtasks] 的 Tensor
        pos_weight_list = []

        # 必须按照 Env 定义的 Group 顺序遍历
        for group_name in env_group_names:
            # 获取该 Group 的配置 (兼容 dict 或属性访问)
            if isinstance(config.tasks, dict):
                group_cfg = config.tasks.get(group_name)
            else:
                group_cfg = getattr(config.tasks, group_name, None)

            if group_cfg is None:
                raise ValueError(f"Group '{group_name}' not found in reward config tasks!")

            # 获取 pos_weight 列表
            # 注意: 这里直接使用 BCE 的 pos_weight 作为 RL 的 TP 奖励
            # 如果想利用 focal_alpha，也可以在这里进行转换逻辑
            weights = group_cfg.pos_weight  # List[float]

            if len(weights) != group_cfg.num_classes:
                raise ValueError(
                    f"Length mismatch in {group_name}: num_classes={group_cfg.num_classes}, len(pos_weight)={len(weights)}")

            pos_weight_list.extend(weights)

        # 转为 Tensor，暂时放在 CPU，计算时再挪到 Device
        self.pos_weight_tensor = torch.tensor(pos_weight_list, dtype=torch.float32)
        self.total_subtasks = len(pos_weight_list)

    def compute_reward(self, logits, labels, stop_decision, pre_action_mask, done_mask):
        """
        计算单步奖励

        Args:
            logits: [B, Total]
            labels: [B, Total] (0 or 1)
            stop_decision: [B, Total] (1=Stop)
            pre_action_mask: [B, Total] (Step开始时的状态)
            done_mask: [B]
        """
        device = logits.device
        # 确保权重 Tensor 在正确的设备上
        if self.pos_weight_tensor.device != device:
            self.pos_weight_tensor = self.pos_weight_tensor.to(device)

        # 1. 确定结算状态 (Settling)
        # 逻辑：原本是 Active (True) 且 Agent 喊停 (Stop=1)
        # 注意：done_expanded 也需要结合 pre_action_mask，防止对原本已经 inactive 的任务重复结算
        done_expanded = done_mask.unsqueeze(1).expand_as(pre_action_mask)
        # Settling: 原本活着，现在被杀死了 (Stop 或 Done)
        is_settling = pre_action_mask & ((stop_decision == 1) | done_expanded)

        # 2. 确定运行状态 (Running)
        # 逻辑：原本是 Active (True) 且 Agent 决定继续 (Stop=0)
        # [修正] 只有决定继续跑的任务，才扣除时间惩罚。
        # 如果 Agent 决定 Stop，因为 Action-First 机制，这层 Transformer 没跑，所以不扣分。
        is_running = pre_action_mask & (stop_decision == 0)

        # 3. 预测结果
        preds = (logits > 0).float()
        is_correct = (preds == labels)
        is_positive_label = (labels == 1)

        # ====================================================
        # A. 构建基础奖励矩阵 (根据细粒度权重)
        # ====================================================
        # 逻辑:
        #   如果是 TP -> 使用 pos_weight_tensor 对应的值
        #   如果是 TN -> 使用 neg_weight (1.0)
        #   如果是 Wrong -> 使用 wrong_penalty (-1.0)

        # 1. 成功奖励矩阵 (假设全都预测对了)
        # [B, Total] = [Total] * [B, Total] (广播) + [Scalar] * [B, Total]
        success_rewards = (self.pos_weight_tensor * is_positive_label.float()) + (
                    self.neg_weight * (~is_positive_label).float())

        # 2. 最终分类奖励矩阵
        # 如果预测正确 -> 取 success_rewards
        # 如果预测错误 -> 取 wrong_penalty
        clf_rewards = torch.where(
            is_correct,
            success_rewards,
            torch.tensor(self.wrong_penalty, device=device)
        )

        # 3. Masking: 只有结算时刻才给分类奖励，否则为 0
        # [B, Total] * [B, Total]
        final_clf_rewards = clf_rewards * is_settling.float()

        # ====================================================
        # B. 时间惩罚
        # ====================================================
        # 只要任务还在跑，就扣分
        time_rewards = torch.zeros_like(logits)
        time_rewards[is_running] = -self.time_penalty

        # ====================================================
        # C. 汇总
        # ====================================================
        total_matrix = final_clf_rewards + time_rewards
        step_rewards = total_matrix.sum(dim=1)  # [B]

        # Logging Info
        info = {
            'reward/step_mean': step_rewards.mean().item(),
            'reward/settled_acc': 0.0
        }
        if is_settling.any():
            acc = (preds[is_settling] == labels[is_settling]).float().mean().item()
            info['reward/settled_acc'] = acc

        return step_rewards, info
