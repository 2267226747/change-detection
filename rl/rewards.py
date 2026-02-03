# 专门定义 Reward 计算函数 (基于 Loss 或 评价指标)
import torch
import torch.nn as nn
from utils.loss import RLLoss
from sklearn.metrics import average_precision_score  # 需要安装 scikit-learn


class RewardCalculator:
    def __init__(self, config, env_group_names, logger=None):
        """
        支持细粒度类别权重的奖励计算器

        Args:
            rl_config: 包含 tasks 配置的对象 (即你提供的 YAML 结构)
            env_group_names: List[str], Env 中定义的 Group 顺序 (e.g., ['road', 'building', ...])
                             必须传入此参数以确保权重顺序与 Logits 顺序一致！
        """
        rl_config = config.rl
        # 基础标量配置
        self.neg_weight = getattr(rl_config, 'reward_neg_weight', 1.0)  # TN 奖励/ FP 惩罚（*-1）
        self.time_penalty = getattr(rl_config, 'time_penalty', -0.2)  # 时间惩罚
        self.tp_scale = getattr(rl_config, 'TP_scale', 10)  # 使用focal_alpha作为 TP奖励的系数

        # ====[ 新增配置: 奖励整形 ]====
        self.use_reward_shaping = getattr(rl_config, 'use_reward_shaping', True)
        self.shaping_coef = getattr(rl_config, 'shaping_coef', 20)

        # ====[ 新增配置: 奖励缩放 ]====
        self.use_reward_scaling = getattr(rl_config, 'use_reward_scaling', True)

        # ====[ 新增配置: 混合精度 (AMP) ]====
        # 默认跟随系统 cuda 情况，或者读取 config
        self.use_amp = getattr(rl_config, 'use_amp', True)
        # 获取设备类型 ('cuda' or 'cpu') 用于 autocast
        self.device_type = 'cuda' if torch.cuda.is_available() else 'cpu'

        logger.info(f"[Rewards] "
                    f"TN reward: {self.neg_weight}, "
                    f"FP penalty: {-self.neg_weight}, "
                    f"Time penalty: {self.time_penalty}, "
                    f"Use reward shaping: {self.use_reward_shaping}, "
                    f"Shaping coef: {self.shaping_coef}, "
                    f"Use reward scaling: {self.use_reward_scaling}."
                    f"AMP Enabled: {self.use_amp}")

        # ====================================================
        # 解析并展平 pos_weight
        # ====================================================
        # 目标: 构建一个 [Total_Subtasks] 的 Tensor
        pos_weight_list = []

        # 必须按照 Env 定义的 Group 顺序遍历
        for group_name in env_group_names:
            # 获取该 Group 的配置 (兼容 dict 或属性访问)
            if isinstance(config.loss.tasks, dict):
                group_cfg = config.loss.tasks.get(group_name)
            else:
                group_cfg = getattr(config.loss.tasks, group_name, None)

            if group_cfg is None:
                raise ValueError(f"Group '{group_name}' not found in reward config tasks!")

            # 获取 pos_weight 列表
            # 注意:使用focal_alpha作为 RL 的 TP 奖励权重
            weights = [((w-0.5) * self.tp_scale + 1) for w in group_cfg.focal_alpha]  # List[float]

            if len(weights) != group_cfg.num_classes:
                raise ValueError(
                    f"Length mismatch in {group_name}: num_classes={group_cfg.num_classes}, len(pos_weight)={len(weights)}")

            pos_weight_list.extend(weights)
            logger.info(f"{group_name} TP reward(FN -penalty) for each subtask: {weights}")

        # 转为 Tensor，暂时放在 CPU，计算时再挪到 Device
        self.pos_weight_tensor = torch.tensor(pos_weight_list, dtype=torch.float32)
        self.total_subtasks = len(pos_weight_list)

        # --- 奖励整形所需的状态 ---
        # 使用 reduction='none' 获取每个样本每个任务的 loss
        self.cls_criterion = RLLoss(config, logger=logger)
        self.last_cls_loss = None  # 用于存储上一步的损失

    def reset(self):
        """
        在每个 episode 开始时重置状态。
        必须在 env.reset() 后调用此方法。
        """
        self.last_cls_loss = None

    def _calc_batch_metrics(self, preds, labels, probs, mask=None):
        """
        辅助函数：计算一批数据的 Acc, P, R, F1, AP
        Args:
            preds: [N] or [B, T] 0/1 预测
            labels: [N] or [B, T] 0/1 标签
            probs: [N] or [B, T] Sigmoid 概率 (用于 AP)
            mask: [B, T] Bool Mask (可选，如果提供则只计算 Mask 为 True 的部分)
        Returns:
            dict: 包含各指标
        """
        # 如果提供了 Mask，先筛选数据
        if mask is not None:
            # 如果 Mask 为空，返回默认值 0
            if not mask.any():
                return {k: 0.0 for k in ['acc', 'precision', 'recall', 'f1', 'ap']}

            flat_preds = preds[mask]
            flat_labels = labels[mask]
            flat_probs = probs[mask]
        else:
            flat_preds = preds.flatten()
            flat_labels = labels.flatten()
            flat_probs = probs.flatten()

        # --- Torch 高效计算基础指标 (避免频繁 CPU 同步) ---
        # TP, FP, FN, TN
        tp = (flat_preds * flat_labels).sum().float()
        tn = ((1 - flat_preds) * (1 - flat_labels)).sum().float()
        fp = (flat_preds * (1 - flat_labels)).sum().float()
        fn = ((1 - flat_preds) * flat_labels).sum().float()

        epsilon = 1e-7
        acc = (tp + tn) / (tp + tn + fp + fn + epsilon)
        precision = tp / (tp + fp + epsilon)
        recall = tp / (tp + fn + epsilon)
        f1 = 2 * (precision * recall) / (precision + recall + epsilon)

        # --- Sklearn 计算 AP (需要 CPU) ---
        # AP 必须基于概率排序，Torch 实现较麻烦，这里转 CPU 计算
        # 注意：如果是极大的 Batch，这一步可能会有微小耗时，但在 RL 训练中通常可接受
        try:
            # 只有当存在正样本或负样本时 AP 才有意义
            if len(flat_labels) > 0:
                y_true = flat_labels.cpu().numpy()
                y_score = flat_probs.detach().cpu().numpy()
                # average_precision_score 处理全0或全1标签时比较健壮，通常返回 0 或 1
                ap = average_precision_score(y_true, y_score)
            else:
                ap = 0.0
        except Exception:
            ap = 0.0

        return {
            'acc': acc.item(),
            'precision': precision.item(),
            'recall': recall.item(),
            'f1': f1.item(),
            'ap': ap
        }

    def compute_reward(self, logits, labels, stop_decision, pre_action_mask, step, done_mask):
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
        logits = logits.float()
        # 确保权重 Tensor 在正确的设备上
        if self.pos_weight_tensor.device != device:
            self.pos_weight_tensor = self.pos_weight_tensor.to(device)
        if self.cls_criterion.device != device:
            self.cls_criterion = self.cls_criterion.to(device)

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
        probs = torch.sigmoid(logits)
        preds = (logits > 0).float()
        is_correct = (preds == labels)
        is_positive_label = (labels == 1)

        # ====================================================
        # A. 构建基础奖励矩阵 (根据细粒度权重)
        # ====================================================

        # 1. 成功奖励矩阵 (假设全都预测对了)
        #   如果是 TP -> 使用 pos_weight_tensor 对应的值
        #   如果是 TN -> 使用 neg_weight (1.0)
        # [B, Total] = [Total] * [B, Total] (广播) + [Scalar] * [B, Total]
        success_rewards = (self.pos_weight_tensor * is_positive_label.float()) + (
                self.neg_weight * (~is_positive_label).float())

        # 2. 错误惩罚矩阵
        # 如果 Label=1 (FN风险): 惩罚 = -pos_weight (例如 -5)
        # 如果 Label=0 (FP风险): 惩罚 = -neg_weight (例如 -1)
        wrong_penalty = (-self.pos_weight_tensor * is_positive_label.float()) + (
                    -self.neg_weight * (~is_positive_label).float())

        # 3. 最终分类奖励矩阵
        # 如果预测正确 -> 取 success_rewards
        # 如果预测错误 -> 取 wrong_penalty
        clf_rewards = torch.where(
            is_correct,
            success_rewards,
            wrong_penalty
        )
        # 4. Masking: 只有结算时刻才给分类奖励，否则为 0
        # [B, Total] * [B, Total]
        final_clf_rewards = clf_rewards * is_settling.float()

        # ====================================================
        # B. 时间惩罚
        # ====================================================
        # 只要任务还在跑，就扣分
        time_rewards = torch.zeros_like(logits)
        time_rewards[is_running] = self.time_penalty

        # ====================================================
        # C. [新增] 奖励整形 (Potential-Based Reward Shaping)
        # ====================================================
        shaping_rewards = torch.zeros_like(logits)
        if self.use_reward_shaping:
            # 只在活跃的任务上计算 loss "潜力"
            # 注意：这里的 loss 是基于当前 logits，即 Agent 采取 action *之后* 的状态
            with torch.amp.autocast(device_type=self.device_type, enabled=self.use_amp):
                current_loss = self.cls_criterion(logits, labels.float()).float()
            # 用 mask 将非活跃任务的 loss 清零
            masked_current_loss = current_loss * pre_action_mask.float()
            # 如果不是第一步
            if self.last_cls_loss is not None:
                # 潜力变化 = -(新Loss - 旧Loss) = 旧Loss - 新Loss
                potential_diff = self.last_cls_loss - masked_current_loss
                shaping_rewards = self.shaping_coef * potential_diff
            # 更新状态，为下一步做准备
            self.last_cls_loss = masked_current_loss.detach()  # detach很重要，防止梯度穿越episodes
        else:
            # 只有在不使用 shaping 时才需要一个占位符
            # 如果后续汇总总是 .sum(dim=1)，甚至可以用一个标量0
            shaping_rewards = torch.tensor(0.0, device=device)

        # ====================================================
        # D. 汇总与缩放 (新代码，根据你的建议优化)
        # ====================================================

        # 分项进行缩放
        if self.use_reward_scaling:
            # 获取每批中结算/活跃的任务数，用于归一化，避免除以0
            num_settling = is_settling.sum(dim=1).float()
            num_active_pre = pre_action_mask.sum(dim=1).float()  # shaping 基于 pre_action_mask
            num_is_running = is_running.sum(dim=1).float()

            # 如果有任务结算，则对分类奖励进行缩放；否则分类奖励本身就是0
            final_clf_rewards = final_clf_rewards.sum(dim=1) / (num_settling + 1e-6)

            # 如果有任务活跃，则对整形奖励进行缩放；否则整形奖励也是0
            shaping_rewards = shaping_rewards.sum(dim=1) / (num_active_pre + 1e-6)

            time_rewards = time_rewards.sum(dim=1) / (num_is_running + 1e-6)

            # 时间惩罚已经是 per-task 的，直接加即可，它天然地反映了代价
            step_rewards = final_clf_rewards + shaping_rewards + time_rewards
        else:
            step_rewards = final_clf_rewards.sum(dim=1) + time_rewards.sum(dim=1) + shaping_rewards.sum(dim=1)

        # Logging Info 返回 batch 的平均值
        info = {
            'reward/step_mean': step_rewards.mean().item(),
            'reward/clf_reward_mean': final_clf_rewards.mean().item(),
            'reward/shaping_reward_mean': shaping_rewards.mean().item(),
            'reward/time_penalty_mean': time_rewards.mean().item()
        }

        # 1. 计算 "Settled" 指标 (仅针对本步结束的任务)
        # 如果本步没有任务结束，这些值为 0
        # settled_metrics = self._calc_batch_metrics(preds, labels, probs, mask=is_settling)
        # for k, v in settled_metrics.items():
        #     info[f'reward/settled_{k}'] = v

        # 2. 计算 "Final/All" 指标 (针对 Batch 内所有任务的当前状态)
        # 即使任务还在跑，也计算当前的预测性能；如果任务已停，Logits 已锁定。
        # 这代表了 "如果现在立刻结算整个 Batch" 的性能
        if done_mask.all():
            finalall_metrics = self._calc_batch_metrics(preds, labels, probs, mask=None)  # mask=None 表示取全部
            for k, v in finalall_metrics.items():
                info[f'reward/finalall_{k}'] = v

        return step_rewards.mean(), info
