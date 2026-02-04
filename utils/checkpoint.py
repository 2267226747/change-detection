# 模型加载与保存帮助函数
import os
import shutil
import torch
import logging


class CheckpointManager:
    def __init__(self, save_dir, logger=None):
        """
        检查点管理器
        Args:
            save_dir (str): 保存权重的目录
            logger (logging.Logger, optional): 日志记录器
        """
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        self.logger = logger if logger else logging.getLogger(__name__)

    def save(self, model, optimizer, epoch, metric, scheduler=None, is_best=False, filename='checkpoints/DL_latest.pth'):
        """
        保存检查点
        Args:
            model: 模型对象
            optimizer: 优化器
            epoch (int): 当前轮数
            metric (float): 当前评估指标 (如 F1-Score)
            scheduler: 学习率调度器 (可选)
            is_best (bool): 是否为最佳模型
            filename (str): 文件名
        """
        # 1. 处理 DataParallel/DDP 的情况
        # 如果模型被 DataParallel 包裹，保存 model.module 的权重
        # 这样加载时即使不用多卡也能加载
        if hasattr(model, 'module'):
            model_state = model.module.state_dict()
        else:
            model_state = model.state_dict()

        # 2. 构建保存字典
        state = {
            'epoch': epoch,
            'state_dict': model_state,
            'optimizer': optimizer.state_dict(),
            'best_metric': metric,
        }

        if scheduler is not None:
            state['scheduler'] = scheduler.state_dict()

        # 3. 保存 latest 文件
        filepath = os.path.join(self.save_dir, filename)
        torch.save(state, filepath)

        self.logger.info(f"Saved checkpoint: {filepath}")

        # 4. 如果是最佳模型，拷贝一份为 model_best.pth
        if is_best:
            best_path = os.path.join(self.save_dir, 'checkpoints/DL_model_best.pth')
            # 不再 copyfile，而是直接保存 model_state
            # 这样保存的文件不包含 optimizer, epoch 等信息，只有权重张量
            torch.save(model_state, best_path)
            self.logger.info(f"Saved Best Model Weights (Score: {metric:.4f}) to {best_path} [Lite Version]")

    def load(self, checkpoint_path, model, optimizer=None, scheduler=None, device='cpu'):
        """
        加载检查点
        Args:
            checkpoint_path (str): 权重文件路径
            model: 模型
            optimizer: 优化器 (用于恢复训练)
            scheduler: 调度器 (用于恢复训练)
            device: 加载设备
        Returns:
            start_epoch (int): 恢复的起始轮数
            best_metric (float): 之前的最佳指标
        """
        if not os.path.isfile(checkpoint_path):
            self.logger.error(f"No checkpoint found at '{checkpoint_path}'")
            raise FileNotFoundError(f"No checkpoint found at '{checkpoint_path}'")

        self.logger.info(f"Loading checkpoint from '{checkpoint_path}'")

        # 加载到指定设备
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # 1. 加载模型权重
        # --- 兼容性判断 ---
        # 如果加载的是 full checkpoint，权重在 ['state_dict'] 里
        # 如果加载的是 best model (lite)，checkpoint 本身就是 state_dict
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            start_epoch = checkpoint.get('epoch', 0) + 1
            best_metric = checkpoint.get('best_metric', 0.0)
            is_full_checkpoint = True
        else:
            # 假设这是纯权重文件
            state_dict = checkpoint
            start_epoch = 0
            best_metric = 0.0
            is_full_checkpoint = False
            self.logger.info("Detected inference-only weight file.")

        # 简单处理 DataParallel 的前缀问题
        if list(state_dict.keys())[0].startswith('module.') and not hasattr(model, 'module'):
            # 如果权重有 module. 但当前模型没有，去掉前缀
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        # strict=True 保证完全匹配，生产环境建议 True
        # strict=False 允许加载部分权重 (如做迁移学习时)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        if missing_keys:
            self.logger.warning(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            self.logger.warning(f"Unexpected keys: {unexpected_keys}")

        # 2. 加载优化器状态 (仅在恢复训练时需要)
        # 仅当是完整 Checkpoint 且 传入了 optimizer 时才加载优化器
        if is_full_checkpoint:
            if optimizer is not None and 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
                self.logger.info("Optimizer state loaded.")

            # 3. 加载调度器状态
            if scheduler is not None and 'scheduler' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler'])
                self.logger.info("Scheduler state loaded.")

        # 4. 获取元数据
        start_epoch = checkpoint.get('epoch', 0) + 1  # 从下一轮开始
        best_metric = checkpoint.get('best_metric', 0.0)

        self.logger.info(f"Loaded checkpoint '{checkpoint_path}' (epoch {checkpoint.get('epoch')})")

        return start_epoch, best_metric

    def auto_resume(self, model, optimizer=None, scheduler=None, device='cpu'):
        """
        自动查找 latest.pth 并恢复
        Returns:
            start_epoch, best_metric
        """
        latest_path = os.path.join(self.save_dir, 'checkpoints/DL_latest.pth')
        if os.path.exists(latest_path):
            self.logger.info(f"Found latest checkpoint, resuming...")
            return self.load(latest_path, model, optimizer, scheduler, device)
        else:
            self.logger.info(f"No latest checkpoint found in {latest_path}. Starting from scratch.")
            return 0, 0.0