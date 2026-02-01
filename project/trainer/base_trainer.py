import os
import argparse
import random
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

# --- 导入自定义模块 ---
from dataset.dataloader import build_dataloader
from models.model import AssembledFusionModel
from utils.loss import MultiTaskLoss
from utils.evaluator import Evaluator
from utils.logger import setup_logger
from utils.checkpoint import CheckpointManager


class Trainer:
    def __init__(self, cfg):
        """
        初始化 Trainer：构建所有必要的组件
        """
        self.cfg = cfg

        # 1. 基础设置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.epochs = getattr(self.cfg.train, 'epochs', 50)
        # 设置保存目录
        self.save_dir = getattr(self.cfg.train, 'save_dir', './results/')
        os.makedirs(self.save_dir, exist_ok=True)
        # 初始化日志
        self.logger = setup_logger(self.save_dir, filename="log/DL_training.log")
        self.logger.info(f"Using Device: {self.device}")
        self.logger.info(f"Save Directory: {self.save_dir}")
        self.writer = SummaryWriter(log_dir=os.path.join(self.save_dir, 'writer'))

        # 2. 构建数据
        self.logger.info("Building DataLoaders...")
        self.train_loader = build_dataloader(self.cfg, self.logger, split='train')
        self.val_loader = build_dataloader(self.cfg, self.logger, split='val')

        # 3. 构建模型
        self.logger.info("Building Model...")
        self.model = AssembledFusionModel(self.cfg, self.logger).to(self.device)
        print(f"加载模型后显存: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
        # 冻结指定层 (必须在构建优化器之前执行)
        self._freeze_layers()
        self._log_model_params()

        # 4. 损失函数与评估器
        self.criterion = MultiTaskLoss(self.cfg, self.logger).to(self.device)
        self.train_evaluator = Evaluator(self.cfg)
        self.val_evaluator = Evaluator(self.cfg)

        # 5. 优化器与调度器
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()

        # 6. Checkpoint 管理
        self.ckpt_manager = CheckpointManager(self.save_dir, self.logger)

        # 7. trainer配置
        seed = getattr(self.cfg.train, 'seed', 42)
        self._set_seed(seed)
        # 状态变量
        self.start_epoch = 0
        self.best_f1 = 0.0
        self.val_interval = getattr(self.cfg.train, 'val_interval', 1)

        # 用于存储历史记录的列表
        self.history_metric_dfs = []
        self.history_loss_dfs = []

        # [AMP] 自动判断是否使用 BF16
        # 如果 GPU 支持 BF16 (如 Ampere 架构)，则启用
        self.use_amp = getattr(self.cfg.train, 'use_amp', True)
        self.amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        # [AMP] Scaler
        # BF16 不需要 scaler，FP16 需要。这里为了通用性可以创建一个，如果是 BF16 设 enabled=False
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.amp_dtype == torch.float16))

        self.logger.info(f"Seed: {seed}, Epoch: {self.epochs}, Val_interval: {self.val_interval}")
        self.logger.info(f"AMP Enabled: {self.use_amp}, Dtype: {self.amp_dtype}")

        # 尝试自动恢复训练
        self.resume_training = getattr(self.cfg.train, 'if_resume_training', False)
        if self.resume_training:
            self._resume_training()
        else:
            self.logger.info(f"Starting from scratch.")

    def _set_seed(self, seed):
        """设置随机种子"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _freeze_layers(self):
        """
        [新增功能] 根据 cfg.train.freeze_patterns 冻结指定层
        """
        freeze_patterns = getattr(self.cfg.train, 'freeze_patterns', [])
        if not freeze_patterns:
            return

        self.logger.info(f"Attempting to freeze layers matching: {freeze_patterns}")

        frozen_count = 0
        total_count = 0

        for name, param in self.model.named_parameters():
            total_count += 1
            # 检查参数名是否包含任一冻结模式
            for pattern in freeze_patterns:
                if pattern in name:
                    param.requires_grad = False
                    frozen_count += 1
                    break

        self.logger.info(f"Successfully frozen {frozen_count}/{total_count} parameters.")

        # [进阶] 如果冻结了 vision_encoder，建议将其强制设为 eval 模式
        # 这将在 _enforce_eval_mode_for_frozen 方法中处理
        if "vision_encoder" in freeze_patterns:
            self.logger.info(
                "Note: 'vision_encoder' will be forced to eval mode during training (BN/Dropout disabled).")

    def _log_model_params(self):
        """打印模型参数统计"""
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        frozen = total - trainable
        self.logger.info(
            f"Model Params: Total {total / 1e6:.2f}M, Trainable {trainable / 1e6:.2f}M, Frozen {frozen / 1e6:.2f}M")

    def _build_optimizer(self):
        """构建优化器，自动过滤不需要梯度的参数"""
        lr = getattr(self.cfg.train, 'lr', 1e-4)
        weight_decay = getattr(self.cfg.train, 'weight_decay', 1e-4)
        self.logger.info(f"Learning rate: {lr}, Weight_decay: {weight_decay}")

        # [关键] 仅传入 requires_grad=True 的参数
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        if len(trainable_params) == 0:
            self.logger.warning("No trainable parameters found! Check your freeze_patterns.")

        return optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

    def _build_scheduler(self):
        """构建学习率调度器"""
        return optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=1e-6)

    def _resume_training(self):
        """断点续训"""
        start_epoch, best_f1 = self.ckpt_manager.auto_resume(
            self.model, self.optimizer, self.scheduler, device=self.device
        )
        self.start_epoch = start_epoch
        self.best_f1 = best_f1
        if start_epoch > 0:
            self.logger.info(f"Resumed from Epoch {start_epoch}. Current Best F1: {best_f1:.4f}")

    def _enforce_eval_mode_for_frozen(self):
        """
        [关键] 即使在 model.train() 之后，也强制将冻结的模块切回 eval 模式。
        这是为了防止 BatchNorm 的 Running Mean/Var 被更新，以及禁用 Dropout。
        """
        freeze_patterns = getattr(self.cfg.train, 'freeze_patterns', [])

        # 针对最常见的 vision_encoder 做特殊处理
        # 如果你冻结了 backbone，通常希望 BN 统计量也冻结
        if "vision_encoder" in freeze_patterns:
            self.model.vision_encoder.eval()

    def _get_mean_metrics(self, df_report, layer_name='AVG_ALL'):
        """
        [修改] 从 df_report 中提取特定 Layer 的 MEAN 行指标
        Args:
            layer_name: 'AVG_ALL' (所有层平均) 或 'layer_11_results' (具体某层)
        """
        if df_report.empty:
            return {k: 0.0 for k in ['Accuracy', 'Precision', 'Recall', 'F1', 'AP']}

        # 筛选特定 Layer 且 SubID 为 'MEAN' 的行
        # df_report 中包含了 Layer 列
        subset = df_report[
            (df_report['Layer'] == layer_name) &
            (df_report['SubID'] == 'MEAN')
            ]

        # 计算所有 Group 的平均 (Macro Average)
        if subset.empty:
            # Fallback: 如果找不到 AVG_ALL (比如只有一层)，取第一层
            return {k: 0.0 for k in ['Accuracy', 'Precision', 'Recall', 'F1', 'AP']}

        metrics_summary = subset[['Accuracy', 'Precision', 'Recall', 'F1', 'AP']].mean().to_dict()
        return metrics_summary

    def _get_last_layer_name(self, df_report):
        """辅助函数：找到最深的一层名称"""
        layers = [x for x in df_report['Layer'].unique() if x != 'AVG_ALL']
        if not layers: return None
        # 根据数字排序: layer_1, layer_3 -> 找 layer_3
        return sorted(layers, key=lambda x: int(x.split('_')[1]))[-1]

    def _log_to_tensorboard(self, df_loss, df_metrics, epoch, split='Train'):
        """
        [修改] 记录到 TensorBoard，支持 Group 列
        Tag 格式: Split/Metric/Layer/Group/SubID
        """
        # 1. 确定 Metrics 的最后一层 (用于筛选 Metrics，通常这是我们在乎的层)
        last_layer_metric = None
        if not df_metrics.empty:
            last_layer_metric = self._get_last_layer_name(df_metrics)

        # 2. 确定 Loss 的目标层
        # 逻辑：优先找和 Metrics 一样的最后一层；如果 Loss 表里没这层（比如只有 AVG_ALL），则退而求其次用 AVG_ALL
        target_layer_loss = None
        if not df_loss.empty:
            loss_layers = df_loss['Layer'].unique()
            if last_layer_metric in loss_layers:
                target_layer_loss = last_layer_metric
            elif 'AVG_ALL' in loss_layers:
                target_layer_loss = 'AVG_ALL'
            else:
                # 如果既没有匹配层也没AVG_ALL，尝试找 Loss 表里最深的一层
                target_layer_loss = self._get_last_layer_name(df_loss)

        # 3. 记录 Loss
        if target_layer_loss and not df_loss.empty:
            # 筛选目标层
            subset_loss = df_loss[df_loss['Layer'] == target_layer_loss]

            for _, row in subset_loss.iterrows():
                # 提取字段
                group = row['Group'] if 'Group' in row else 'Default'
                sub_id = row['SubID']
                loss_val = row['Loss']

                # Tag 格式: Train/Loss/AVG_ALL/building/0
                # 将 Group 加入路径，方便在 TensorBoard 左侧筛选
                tag = f"{split}/Loss/{target_layer_loss}/{group}/{sub_id}"
                self.writer.add_scalar(tag, loss_val, epoch)

        # 4. 记录 Metrics
        if last_layer_metric and not df_metrics.empty:
            # 筛选最后一层
            subset_metric = df_metrics[df_metrics['Layer'] == last_layer_metric]

            # 自动识别存在的指标列
            metric_cols = [c for c in ['Accuracy', 'Precision', 'Recall', 'F1', 'AP'] if c in subset_metric.columns]

            for _, row in subset_metric.iterrows():
                group = row['Group'] if 'Group' in row else 'Default'
                sub_id = row['SubID']

                for m in metric_cols:
                    val = row[m]
                    # Tag 格式: Train/F1/ClassifyLayer_1/building/0
                    tag = f"{split}/{m}/{last_layer_metric}/{group}/{sub_id}"
                    self.writer.add_scalar(tag, val, epoch)

    def _save_loss_history_csv(self):
        if not self.history_loss_dfs: return
        full_df = pd.concat(self.history_loss_dfs, ignore_index=True)
        # 调整列顺序
        cols = ['Epoch', 'Split', 'Layer', 'Group', 'SubID', 'Loss']
        # 确保列存在再索引，防止报错
        existing_cols = [c for c in cols if c in full_df.columns]
        full_df = full_df[existing_cols]

        save_path = os.path.join(self.save_dir, "acc_loss/training_loss_history.csv")
        full_df.to_csv(save_path, index=False)
        self.logger.info(f"Updated loss history: {save_path}")

    def _save_metric_history_csv(self):
        """
        将历史数据合并并保存为 CSV
        """
        if not self.history_metric_dfs:
            return

        # 1. 合并所有 DataFrame
        full_df = pd.concat(self.history_metric_dfs, ignore_index=True)

        # 2. 调整列顺序，把 Epoch, Split 放在最前面
        cols = full_df.columns.tolist()
        # 假设 Epoch 已经在第一列了(由 evaluator 处理)，我们把 Split 挪到第二列
        if 'Split' in cols:
            cols.insert(1, cols.pop(cols.index('Split')))
        full_df = full_df[cols]

        # 3. 保存
        save_path = os.path.join(self.save_dir, "acc_loss/training_metrics_history.csv")
        full_df.to_csv(save_path, index=False)
        self.logger.info(f"Updated metrics history: {save_path}")

    def train_epoch(self, epoch):
        """
        训练单个 Epoch
        """
        self.model.train()
        # [关键] 在调用 train() 后，再次强制冻结层进入 eval 模式
        self._enforce_eval_mode_for_frozen()

        self.train_evaluator.reset()

        total_loss = 0.0
        num_batches = len(self.train_loader)

        # 用于收集每个 Batch 的 Loss 详情
        batch_loss_stats = []

        pbar = tqdm(self.train_loader, desc=f"Train Epoch {epoch}", leave=False)

        for step, batch in enumerate(pbar):
            # 1. 数据搬运
            t1 = batch['pixel_values_t1'].to(self.device)
            t2 = batch['pixel_values_t2'].to(self.device)
            targets = batch['labels'].to(self.device)
            oids = batch['oid']  # oid列表通常不用上GPU

            # 2. 梯度清零 (set_to_none=True 更省显存)
            self.optimizer.zero_grad(set_to_none=True)

            # 3. 前向传播 + Loss (AMP)
            with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
                outputs = self.model(t1, t2)
                # Loss 计算会自动处理精度 (基于之前的优化)
                loss, raw_stats = self.criterion(outputs, targets)

            # print(f"前向传播后显存: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

            # 4. 反向传播 + 优化 (Scaler)
            self.scaler.scale(loss).backward()

            # 梯度裁剪
            self.scaler.unscale_(self.optimizer)
            total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            if step == 0:
                self.logger.info(f"Epoch {epoch} step 0 train/Grad_Norm: {total_norm.item()}")

            # 6. 更新参数
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # 记录 Loss
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            # 收集 raw_stats (注意：raw_stats 里的值已经是 float，不需要 detach)
            batch_loss_stats.append(raw_stats)

            # 7. 更新评估器 (Detach!)
            # 更新评估器 (Detach + 传入 OID)
            # [优化] 使用字典推导式快速 detach
            # 定义一个递归函数，不管字典套列表，还是列表套字典，都能处理
            def recursive_detach(data):
                if isinstance(data, dict):
                    return {k: recursive_detach(v) for k, v in data.items()}
                elif isinstance(data, list):
                    return [recursive_detach(v) for v in data]
                elif isinstance(data, tuple):
                    return tuple(recursive_detach(v) for v in data)
                elif hasattr(data, 'detach'):
                    return data.detach()
                else:
                    # 如果是 int, str, float 等基本类型，原样返回
                    return data

            # 一行代码解决所有层级问题
            detached_outputs = recursive_detach(outputs)

            # [关键] 传入 oids
            self.train_evaluator.update(detached_outputs, targets, oids)

            # # 每步后强制清空缓存
            # if step % 10 == 0:
            #     torch.cuda.empty_cache()

        avg_loss = total_loss / num_batches

        # 计算 Loss DataFrame
        df_loss_report = MultiTaskLoss.format_loss_to_df(batch_loss_stats, epoch=epoch)
        # 计算 Metric DataFrame
        metrics, df_metric_report = self.train_evaluator.compute(epoch=epoch)

        return avg_loss, df_loss_report, df_metric_report

    @torch.no_grad()
    def evaluate(self, epoch):
        """
        验证逻辑
        """
        self.model.eval()
        self.val_evaluator.reset()

        total_loss = 0.0
        batch_loss_stats = []
        pbar = tqdm(self.val_loader, desc=f"Val Epoch {epoch}", leave=False)

        for batch in pbar:
            t1 = batch['pixel_values_t1'].to(self.device)
            t2 = batch['pixel_values_t2'].to(self.device)
            targets = batch['labels'].to(self.device)
            oids = batch['oid']  # oid列表通常不用上GPU

            # 推理也建议开启 autocast 以节省显存并匹配训练时的行为
            with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
                outputs = self.model(t1, t2)
                loss, raw_stats = self.criterion(outputs, targets)

            batch_loss_stats.append(raw_stats)
            total_loss += loss.item()
            self.val_evaluator.update(outputs, targets, oids)

        avg_loss = total_loss / len(self.val_loader)
        # 生成 Loss 和 metric 表格
        df_loss_report = MultiTaskLoss.format_loss_to_df(batch_loss_stats, epoch=epoch)
        metrics, df_metric_report = self.val_evaluator.compute(epoch=epoch)

        return avg_loss, df_loss_report, df_metric_report

    def run(self):
        """
        主训练循环
        """
        self.logger.info("Starting Training Loop...")

        for epoch in range(self.start_epoch, self.epochs):
            self.logger.info(f"{'=' * 15} Epoch {epoch + 1}/{self.epochs} {'=' * 15}")

            # 1. Train
            train_loss, df_train_loss, df_train_metrics = self.train_epoch(epoch)

            # 保存 Loss 历史
            df_train_loss['Split'] = 'Train'
            self.history_loss_dfs.append(df_train_loss)
            # 标记数据集类型并记录
            if not df_train_metrics.empty:
                df_train_metrics['Split'] = 'Train'  # 添加一列标识
                self.history_metric_dfs.append(df_train_metrics)

            # 获取训练集平均 F1
            last_layer = self._get_last_layer_name(df_train_metrics)
            # train_means = self._get_mean_metrics(df_train_metrics, layer_name=last_layer)

            self.logger.info(f"LR: {self.optimizer.param_groups[0]['lr']:.2e}")

            # Log 打印 (只打印 AVG_ALL 部分，防止刷屏)
            # 筛选 Layer=last_layer
            loss_summary = df_train_loss[
                (df_train_loss['Layer'] == last_layer)
            ]
            self.logger.info(f"[Train Loss Summary(last_layer)]\n{loss_summary.to_string(index=False)}")
            metrics_summary = df_train_metrics[
                (df_train_metrics['Layer'] == last_layer)
            ]
            self.logger.info(f"[Train Metrics Summary(last_layer)]\n{metrics_summary.to_string(index=False)}")

            # [新增] TensorBoard 记录 Train
            self._log_to_tensorboard(df_train_loss, df_train_metrics, epoch, split='Train')

            # 更新学习率
            self.scheduler.step()

            # 2. Validate
            if (epoch + 1) % self.val_interval == 0:
                val_loss, df_val_loss, df_val_metrics = self.evaluate(epoch)

                # 保存 Loss 历史
                df_val_loss['Split'] = 'Val'
                self.history_loss_dfs.append(df_val_loss)
                # 标记数据集类型并记录
                if not df_val_metrics.empty:
                    df_val_metrics['Split'] = 'Val'  # 添加一列标识
                    self.history_metric_dfs.append(df_val_metrics)

                # 1. Log 最后一层表现 (通常代表最终性能)
                last_layer = self._get_last_layer_name(df_val_metrics)
                val_last_means = self._get_mean_metrics(df_val_metrics, layer_name=last_layer)

                # 2. Log 平均层表现 (代表 Deep Supervision 的整体稳定性)
                val_avg_means = self._get_mean_metrics(df_val_metrics, layer_name='AVG_ALL')
                self.logger.info(f"[Val] Avg Loss: {val_loss:.4f}")
                # self.logger.info(
                # f"   >> Last Layer ({last_layer}) F1: {val_last_means['F1']:.4f} | AP: {val_last_means['AP']:.4f}")
                self.logger.info(
                    f"   >> Avg  Layer (Ensemble)      | Acc: {val_avg_means['Accuracy']:.4f} | Pre: {val_avg_means['Precision']:.4f} | Recall: {val_avg_means['Recall']:.4f} | F1: {val_avg_means['F1']:.4f} | AP: {val_avg_means['AP']:.4f}")

                # 打印表格时，只打印 Last Layer，防止刷屏
                # Log
                loss_summary = df_val_loss[
                    (df_val_loss['Layer'] == last_layer)
                ]
                self.logger.info(f"[Val Loss Summary(last_layer)]\n{loss_summary.to_string(index=False)}")
                metrics_summary = df_val_metrics[
                    (df_val_metrics['Layer'] == last_layer)
                ]
                self.logger.info(f"[Val Metrics Summary(last_layer)]\n{metrics_summary.to_string(index=False)}")

                self._log_to_tensorboard(df_val_loss, df_val_metrics, epoch, split='Val')
                # 保存预测结果 (包含所有层)
                # pred_save_path = os.path.join(self.save_dir, "predictions", f"val_preds_epoch_{epoch}.csv")
                # self.val_evaluator.save_predictions(pred_save_path)

                # === Checkpoint ===
                # 使用 Last Layer 的 F1 作为保存最佳模型的依据
                current_f1 = val_last_means['F1']
                is_best = current_f1 > self.best_f1

                if is_best:
                    self.best_f1 = current_f1
                    self.logger.info(f"New Best F1: {self.best_f1:.4f} !!!")

                # 保存时，建议把 val_last_means (字典) 传进去
                self.ckpt_manager.save(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    metric=self.best_f1,
                    is_best=is_best,
                    filename='checkpoints/DL_latest.pth'
                )
            else:
                # 即使不验证，也保存最新的训练状态以防中断
                self.ckpt_manager.save(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    metric=self.best_f1,
                    is_best=False,
                    filename='checkpoints/DL_latest.pth'
                )

            # 每个 Epoch 结束都保存一次总表 (防止意外中断导致数据丢失)
            self._save_loss_history_csv()
            self._save_metric_history_csv()

        self.logger.info("Training Finished.")

# if __name__ == '__main__':
#     # 外部调用示例
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--config', type=str, default='config.yaml')
#     args = parser.parse_args()
#
#     if not os.path.exists(args.config):
#         raise FileNotFoundError(f"Config not found: {args.config}")
#
#     cfg = OmegaConf.load(args.config)
#
#     # 实例化并运行
#     trainer = SVITrainer(cfg)
#     trainer.run()
