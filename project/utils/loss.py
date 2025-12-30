# 定义分类 Loss 和 RL Loss
import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha: 平衡因子 (0 < alpha < 1)，用于调整正负样本权重。
            gamma: 聚焦参数 (gamma >= 0)，用于挖掘难分样本。
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        # BCEWithLogitsLoss = Sigmoid + BCELoss
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        # 计算 p_t (预测概率)
        p = torch.sigmoid(logits)
        # 如果 target=1, p_t = p; 如果 target=0, p_t = 1-p
        p_t = p * targets + (1 - p) * (1 - targets)

        # 计算 alpha_t
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Focal Loss 公式
        focal_loss = alpha_t * (1 - p_t) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class MultiLayerLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # --- 1. 获取 Loss 类型及参数 ---
        # 默认为 BCE
        self.loss_type = getattr(cfg.loss, 'loss_type', 'BCE')

        if self.loss_type == 'BCE':
            # 获取 pos_weight (解决正负样本不平衡)
            # 在 yaml 中通常配置为列表，如 [1.0, 5.0, 1.0]
            pos_weight_list = getattr(cfg.loss, 'pos_weight', None)
            if pos_weight_list is not None:
                # 注册为 buffer，确保模型保存时包含它，但不会被优化器更新
                self.register_buffer('pos_weight', torch.tensor(pos_weight_list))
            else:
                self.pos_weight = None

            self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
            print(f"[Loss] Using BCEWithLogitsLoss (pos_weight={pos_weight_list})")

        elif self.loss_type == 'Focal':
            # 获取 Focal Loss 参数
            alpha = getattr(cfg.loss, 'focal_alpha', 0.25)
            gamma = getattr(cfg.loss, 'focal_gamma', 2.0)

            self.criterion = BinaryFocalLoss(alpha=alpha, gamma=gamma)
            print(f"[Loss] Using BinaryFocalLoss (alpha={alpha}, gamma={gamma})")

        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")

        # --- 2. 定义层级权重 (Deep Supervision Weights) ---
        # 假设 output 有 N 层，列表长度应为 N
        self.layer_weights = getattr(cfg.loss, 'layer_weights', None)

        # --- 3. 定义任务映射 ---
        # Dict: {'group_name': [col_idx1, col_idx2]}
        self.task_mapping = getattr(cfg.loss, 'task_indices', None)

    def forward(self, model_outputs, targets):
        """
        Args:
            model_outputs: Dict {'Classifier_1_results': task_dict, 'Classifier_2_results': task_dict, ...}
            targets: Tensor [Batch, Num_Classes]

        Returns:
            total_loss: scalar (带梯度的总 Loss)
            loss_dict: dict (无梯度的数值，用于日志记录)
        """
        total_loss = 0.0
        loss_dict = {}

        # 确保 BCE 的 pos_weight 在正确的 device 上 (防止多卡训练/不同设备时的报错)
        if self.loss_type == 'BCE' and self.pos_weight is not None:
            if self.criterion.pos_weight.device != targets.device:
                self.criterion.pos_weight = self.pos_weight.to(targets.device)

        # 获取并排序层级 Keys (Classifier_1, Classifier_2, ...)
        layer_keys = sorted(model_outputs.keys(), key=lambda x: int(x.split('_')[1]))
        num_layers = len(layer_keys)

        # 动态生成层权重 (如果 cfg 未定义)
        if self.layer_weights is None:
            # 默认策略: 最后一层权重 1.0，其余层 0.3
            weights = [0.3] * (num_layers - 1) + [1.0]
        else:
            weights = self.layer_weights
            assert len(weights) == num_layers, \
                f"Config layer_weights length ({len(weights)}) does not match model outputs ({num_layers})"

        # --- 循环计算每一层 Loss ---
        for i, layer_key in enumerate(layer_keys):
            layer_output = model_outputs[layer_key]
            layer_weight = weights[i]

            current_layer_loss = 0.0

            # Case A: 多任务 (Classifier 输出 Dict)
            if isinstance(layer_output, dict):
                for task_name, logits in layer_output.items():
                    # 确定对应的 Target
                    if self.task_mapping:
                        indices = self.task_mapping[task_name]
                        task_target = targets[:, indices]
                    else:
                        # 无映射则假设对应所有
                        task_target = targets

                    # 计算 Loss
                    task_loss = self.criterion(logits, task_target)
                    current_layer_loss += task_loss

                    # [记录] 单个任务头的 Loss
                    loss_dict[f"{layer_key}/{task_name}"] = task_loss.item()

            # Case B: 单任务 (Classifier 输出 Tensor)
            else:
                logits = layer_output
                task_loss = self.criterion(logits, targets)
                current_layer_loss += task_loss
                loss_dict[f"{layer_key}/loss"] = task_loss.item()

            # 累加到总 Loss (带权重)
            total_loss += layer_weight * current_layer_loss

            # [记录] 该层的总 Loss (未加权的纯 Loss)
            loss_dict[f"{layer_key}_sum"] = current_layer_loss.item()

        # [记录] 最终加权 Loss
        loss_dict['total_loss'] = total_loss.item()

        return total_loss, loss_dict