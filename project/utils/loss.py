# 定义分类 Loss 和 RL Loss
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd


class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=None, reduction=None):
        """
        Args:
            alpha: (Tensor) 长度等于 num_classes，每个子任务独立的 alpha
            gamma: (float) 聚焦参数
        """
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

        # 注册 alpha 为 buffer (不作为参数更新，但随模型保存)
        if alpha is not None:
            self.register_buffer('alpha', alpha)
        else:
            self.alpha = None

        # 注册 gamma 为 buffer (不作为参数更新，但随模型保存)
        if gamma is not None:
            self.register_buffer('gamma', gamma)
        else:
            self.gamma = None

    def forward(self, logits, targets):
        """
        [AMP优化]
        无论输入是什么精度，这里强制转为 float32 计算。
        因为 Sigmoid 和 Pow 在 BF16 下极易损失精度。
        """
        logits = logits.float()
        targets = targets.float()

        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)

        # 动态获取 alpha
        if self.alpha is not None:
            # 确保 alpha 维度匹配: [Batch, N] -> alpha 需要是 [N] 或广播
            # self.alpha 应该是 [num_classes]
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        else:
            alpha_t = 0.25 * targets + 0.75 * (1 - targets)  # 默认值

        # 如果 gamma 为空，默认取 2.0
        if self.gamma.numel() > 0:
            gamma_val = self.gamma.to(dtype=torch.float32)
        else:
            gamma_val = 2.0

        focal_loss = alpha_t * (1 - p_t) ** gamma_val * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss  # 返回 [Batch, Num_Classes]


class MultiTaskLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.loss_type = getattr(cfg.loss, 'type', 'BCE')
        # [优化] 将 weights 转为 tensor buffer，方便 device 管理
        layer_weights = getattr(cfg.loss, 'layer_weights', [1.0])
        self.register_buffer('layer_weights', torch.tensor(layer_weights, dtype=torch.float32))

        # --- 核心：解析任务配置与构建 Loss ---
        # cfg.loss.tasks 是一个字典结构 (DictConfig)
        task_configs = cfg.loss.tasks
        self.tasks = {}  # 存储每个任务的元数据
        self.criteria = nn.ModuleDict()  # 存储每个任务对应的 Loss 函数
        current_idx = 0

        # 遍历 YAML 中的 tasks
        for task_name, task_cfg in task_configs.items():
            num_classes = task_cfg.num_classes

            # 1. 记录切片索引
            self.tasks[task_name] = {
                'start': current_idx,
                'end': current_idx + num_classes
            }
            current_idx += num_classes

            # 2. 构建该任务专属的 Loss Function
            if self.loss_type == 'BCE':
                # 获取该任务专属的 pos_weight
                pos_w = getattr(task_cfg, 'pos_weight', [1.0] * num_classes)
                pos_w_tensor = torch.tensor(pos_w, dtype=torch.float32)

                # 创建带权重的 BCE
                criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w_tensor, reduction='none')

            elif self.loss_type == 'Focal':
                # 获取该任务专属的 alpha
                alpha = getattr(task_cfg, 'focal_alpha', [0.25] * num_classes)
                gamma = getattr(cfg.loss, 'focal_gamma', [2.0] * num_classes)  # gamma 通常全局统一

                alpha_tensor = torch.tensor(alpha, dtype=torch.float32)
                gamma_tensor = torch.tensor(gamma, dtype=torch.float32)
                criterion = BinaryFocalLoss(alpha=alpha_tensor, gamma=gamma_tensor)

            else:
                raise ValueError("Unknown Loss Type")

            self.criteria[task_name] = criterion

        print(f"[Loss] Initialized {self.loss_type} for tasks: {list(self.tasks.keys())}")

    def forward(self, model_outputs, targets):
        """
        Args:
            model_outputs: Dict {'layer_X': {'road': [B, 8], ...}}
            targets: Tensor [Batch, Total_Cols] (所有任务标签拼接的大宽表)
        Returns:
            total_loss: scalar (Tensor, 带梯度)
            raw_stats: dict (包含所有细节数值，用于累积计算 Epoch 平均值)
        [AMP优化]
        确保所有 Loss 计算都在 FP32 下进行。
        """

        # 1. Targets 统一转 FP32
        targets = targets.float()
        total_loss = 0.0

        # 确保 BCE pos_weight 的 device 正确
        if self.loss_type == 'BCE':
            for crit in self.criteria.values():
                if crit.pos_weight is not None and crit.pos_weight.device != targets.device:
                    crit.pos_weight = crit.pos_weight.to(targets.device)

        # 获取层级 keys
        layer_keys = sorted(model_outputs.keys(), key=lambda x: int(x.split('_')[1]))

        # 动态权重处理
        if len(self.layer_weights) != len(layer_keys):
            # fallback
            weights = [1.0] * len(layer_keys)
        else:
            weights = self.layer_weights

        # 用于存储原始数值的字典，结构扁平化以便快速累加
        # Key: "Layer/Group/SubID" -> Value: float
        raw_stats = {}
        # 辅助存储，用于计算 AVG_ALL
        # Key: "Group/SubID" -> List[float]
        subclass_history = {}


        # --- 双重循环：Layer -> Task ---
        for i, layer_key in enumerate(layer_keys):
            layer_output_dict = model_outputs[layer_key]  # {'road': ..., 'building': ...}
            layer_weight = weights[i]
            layer_total_loss = 0.0  # 该层所有任务的总 Loss

            # 遍历每个任务头 (road, building...)
            for task_name, logits in layer_output_dict.items():
                if task_name not in self.tasks:
                    continue  # 忽略未在 loss config 中定义的任务

                # 1. 获取对应的 Label 切片
                start = self.tasks[task_name]['start']
                end = self.tasks[task_name]['end']
                # [Batch, Num_Subtasks]
                task_target = targets[:, start:end]

                # [AMP优化] 再次强制 Logits 转 FP32 (双重保险)
                # --- 新增的处理逻辑 ---
                if isinstance(logits, list):
                    # 将列表中的多个 Tensor 沿着维度 1 (列) 拼接起来
                    # 结果形状会变成 (Batch_Size, 8)
                    logits = torch.cat(logits, dim=1)
                # --------------------
                logits = logits.float()

                # 2. 获取对应的 Loss 函数
                criterion = self.criteria[task_name]
                # logits [B, Num_Subtasks]
                raw_loss = criterion(logits, task_target)

                # 3. 在 Batch 维度求平均 -> [Num_Subclasses]
                # 这就是该层、该任务中、每个子类的 Loss
                per_subclass_loss = raw_loss.mean(dim=0)

                # 4. 记录每个子类的 Loss
                for sub_idx, sub_loss in enumerate(per_subclass_loss):
                    val = sub_loss.item()

                    # 1. 记录层级子类 Loss
                    raw_stats[f"{layer_key}/{task_name}/{sub_idx}"] = val

                    # 2. 存入历史以计算全局平均
                    hist_key = f"{task_name}/{sub_idx}"
                    if hist_key not in subclass_history:
                        subclass_history[hist_key] = []
                    subclass_history[hist_key].append(val)

                # 5.
                # 计算该任务在本层的平均 (MEAN)
                task_mean = per_subclass_loss.mean().item()
                raw_stats[f"{layer_key}/{task_name}/MEAN"] = task_mean
                # 计算该任务在本层的总 Loss (通常是所有子类求和)
                layer_total_loss += per_subclass_loss.sum()


            # 累加层级 Loss
            total_loss += layer_weight * layer_total_loss
            raw_stats[f"{layer_key}/ALL/SUM"] = layer_total_loss.item()

        # 计算 AVG_ALL (跨层平均)
        for hist_key, vals in subclass_history.items():
            task_name, sub_idx = hist_key.split('/')
            avg_val = sum(vals) / len(vals)
            raw_stats[f"AVG_ALL/{task_name}/{sub_idx}"] = avg_val

            # 同时累加用于计算 AVG_ALL 的 MEAN
            if f"AVG_ALL/{task_name}/MEAN_list" not in raw_stats:
                raw_stats[f"AVG_ALL/{task_name}/MEAN_list"] = []
            raw_stats[f"AVG_ALL/{task_name}/MEAN_list"].append(avg_val)

        # 计算 AVG_ALL 的 Group Mean
        for key in list(raw_stats.keys()):
            if "MEAN_list" in key:
                vals = raw_stats[key]
                raw_stats[key.replace("MEAN_list", "MEAN")] = sum(vals) / len(vals)
                del raw_stats[key]  # 清理临时 list

        raw_stats['TOTAL/ALL/SUM'] = total_loss.item()

        return total_loss, raw_stats


    @staticmethod
    def format_loss_to_df(loss_stats_list, epoch=None):
        """
        将累积的 stats 列表转换为 DataFrame 表格
        Args:
            loss_stats_list: List[dict]，包含了每个 batch 的 raw_stats
            epoch: int
        Returns:
            pd.DataFrame
        """
        if not loss_stats_list:
            return pd.DataFrame()

        # 1. 对所有 Batch 的 stats 求平均
        # 先把 list of dicts 转为 dict of lists
        agg_dict = {}
        for stats in loss_stats_list:
            for k, v in stats.items():
                if k not in agg_dict:
                    agg_dict[k] = []
                agg_dict[k].append(v)

        # 计算平均值
        mean_stats = {k: np.mean(v) for k, v in agg_dict.items()}

        # 2. 解析 Key 构建表格行
        # Key 格式: Layer/Group/SubID
        rows = []
        for key, val in mean_stats.items():
            parts = key.split('/')
            if len(parts) == 3:
                layer, group, sub_id = parts
                rows.append({
                    'Layer': layer,
                    'Group': group,
                    'SubID': sub_id,
                    'Loss': round(val, 5)
                })

        # 3. 创建 DataFrame
        df = pd.DataFrame(rows)

        # 4. 排序美化
        # 简单排序逻辑
        df = df.sort_values(by=['Layer', 'Group', 'SubID'])

        if epoch is not None:
            df.insert(0, 'Epoch', epoch)

        return df


class RLLoss(MultiTaskLoss):
    def __init__(self, cfg):
        """
        RL 专用的 Loss 计算器。
        继承自 MultiTaskLoss 以复用配置解析和 Loss 准则构建逻辑。
        """
        super().__init__(cfg)

        # 预先计算每个 Group 在展平 Logits 中的切片索引
        # 假设 Logits 是按照 self.tasks 中定义的顺序拼接的
        # self.tasks 的结构: {'road': {'start': 0, 'end': 8}, ...}
        # 这个结构在父类 __init__ 中已经构建好了，直接用即可。
        pass

    def forward(self, logits, targets):
        """
        Args:
            logits:
                - 情况A (展平): [Batch, Total_Subtasks] Tensor
                - 情况B (字典): {'road': [B, 8], 'building': [B, 4], ...}
            targets: [Batch, Total_Subtasks] 始终假设标签是展平的 Tensor (来自 Buffer)

        Returns:
            loss_matrix: [Batch, Total_Subtasks] 每个样本每个子任务的 Loss (无梯度)
        """
        # [AMP优化] 强制 FP32
        targets = targets.float()

        batch_size = targets.shape[0]

        # 用于收集按顺序计算的各组 loss，最后拼接
        loss_parts = []

        # 遍历 self.tasks (Python 3.7+ 字典保持插入顺序，这很重要)
        # 顺序必须与 YAML 配置及 targets 的列顺序一致
        for task_name, meta in self.tasks.items():
            start = meta['start']
            end = meta['end']

            # --- A. 获取 Targets (始终切片) ---
            group_targets = targets[:, start:end]

            # --- B. 获取 Logits (兼容 Dict 和 Tensor) ---
            if isinstance(logits, dict):
                # 情况 1: 输入是 Dict
                if task_name not in logits:
                    raise ValueError(f"Task '{task_name}' missing in logits dict.")
                group_logits = logits[task_name]

                # 兼容性处理：如果字典里存的是 list (MultiTaskLoss 的历史遗留)
                if isinstance(group_logits, list):
                    group_logits = torch.cat(group_logits, dim=1)

            else:
                # 情况 2: 输入是展平 Tensor
                group_logits = logits[:, start:end]

            # 确保类型匹配 (防御性编程)
            group_logits = group_logits.float()

            # --- C. 计算 Loss ---
            criterion = self.criteria[task_name]

            # 确保 criterion 的 reduction 属性是 'none' 或 None
            # BCEWithLogitsLoss 的 reduction 属性是字符串
            # 自定义 BinaryFocalLoss 的 reduction 属性也是字符串
            if hasattr(criterion, 'reduction') and criterion.reduction != 'none':
                # 如果父类初始化时没设为 none，这里临时警告或强制修改（通常父类初始化已设为 none）
                criterion.reduction = None

            # 返回形状: [Batch, Group_Num_Classes]
            group_loss = criterion(group_logits, group_targets)

            loss_parts.append(group_loss)

        # --- D. 拼接回大矩阵 ---
        # 结果形状: [Batch, Total_Subtasks]
        loss_matrix = torch.cat(loss_parts, dim=1)

        return loss_matrix