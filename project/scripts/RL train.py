import torch
import argparse
import os
from types import SimpleNamespace
import yaml
import sys
from utils.logger import setup_logger

# 引入你的 RL 模块
from rl.env import RLEnv
from rl.networks import ActorCriticNetwork
from rl.agent import PPOAgent
from rl.buffer import RolloutBuffer
from rl.rewards import RewardCalculator
from trainer.rl_trainer import PPOTrainer

# 获取项目根目录路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
from utils.config import Config

# 预训练模型定义
from models.model import AssembledFusionModel
from dataset.dataloader import build_dataloader


def get_config_path(config_filename):
    """根据项目根目录获取配置文件的完整路径"""
    return os.path.join(project_root, 'configs', config_filename)


def get_config():
    """定义超参数"""
    config = SimpleNamespace()

    # 路径配置
    config.log_dir = "./logs/rl_finetune"
    config.ckpt_dir = "./checkpoints/rl_finetune"

    # RL 训练参数
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    config.total_epochs = 1000  # RL Iterations
    config.num_steps = 6  # Rollout steps per episode (对应 6 个分类头)
    config.batch_size = 64  # PPO Update Mini-batch size
    config.ppo_epochs = 4  # PPO Update epochs per rollout

    # PPO 超参数
    config.lr = 3e-5  # 学习率 (通常比预训练小)
    config.weight_decay = 1e-4
    config.gamma = 0.99  # 折扣因子
    config.gae_lambda = 0.95  # GAE 平滑系数
    config.clip_param = 0.2  # PPO Clip
    config.max_grad_norm = 0.5  # 梯度裁剪
    config.hidden_dim = 512  # RL Network 隐藏层维度

    # 损失权重
    config.value_loss_coef = 0.5
    config.entropy_coef = 0.01
    config.cls_loss_coef = 1.0  # 联合训练分类 Loss 的权重
    config.use_value_clip = True

    # 动作配置
    config.action_scale = 0.1  # Query 修正幅度缩放

    # 奖励配置
    config.reward_pos_weight = 2.0
    config.reward_neg_weight = 1.0
    config.reward_wrong_penalty = -1.0
    config.time_penalty = 0.05

    # 训练策略
    config.freeze_classifier = False  # 是否同时微调分类头 (建议 Warmup 后设为 False)
    config.save_interval = 50
    config.eval_interval = 100

    return config


def main():
    # 1. 加载配置
    cfg = Config.from_yaml(get_config_path("defaults.yaml"))
    print("Configuration loaded.")

    # 加载logger
    logger = setup_logger(getattr(cfg.train, 'save_dir', './results/'))

    # 2. 准备数据
    logger.info("Loading data...")
    train_loader = build_dataloader(cfg, logger, split='train')
    val_loader = build_dataloader(cfg, logger, split='val')

    # 3. 加载预训练模型 (Mockup)
    logger.info("Loading pretrained model...")
    pretrained_model = AssembledFusionModel(cfg, logger)
    pretrained_model.load_state_dict(torch.load(cfg.rl.pre_model_path))
    pretrained_model.to(cfg.rl.device)

    # 4. 实例化环境
    logger.info("Initializing environment...")
    env = RLEnv(
        pretrained_model=pretrained_model,
        config=cfg,
        device=cfg.rl.device,
        logger=logger
    )

    # 5. 定义网络形状 (用于构建 RL Network)
    # 根据 Env 实际解析出的结构
    env_shapes = {
        'vision_dim': getattr(cfg.model.vision, 'vision_dim', 1024),
        'query_dim': getattr(cfg.model.query_token, 'query_dim', 1024),  # 需与预训练模型一致
        'num_groups': getattr(cfg.model.query_token, 'task_nums', 4),
        'tokens_per_group': getattr(cfg.model.query_token, 'tokens_per_task', 128),
        'total_subtasks': env.total_subtasks
    }
    logger.info("Initializing Actor-Critic Network...")
    logger.info(f"Vision token dim: {env_shapes['vision_dim']}, "
                f"Query token dim: {env_shapes['query_dim']}, "
                f"Num groups: {env_shapes['num_groups']}, "
                f"Tokens per group(Query): {env_shapes['tokens_per_group']}")

    # 6. 实例化 Actor-Critic Network
    network = ActorCriticNetwork(cfg, env_shapes, logger).to(cfg.device)

    # 7. 实例化 Agent
    # [关键] 传入分类头引用和参数，实现联合优化
    agent = PPOAgent(
        network=network,
        classifier_heads=env.model.class_heads,
        classifier_params=env.get_classifier_parameters(),
        config=cfg
    ).to(rl_cfg.device)

    # 8. 实例化辅助组件
    # Buffer: 注意 buffer_size = num_steps * env_batch_size
    # 这里 env_batch_size 是隐式的 (由 train_loader 的 batch_size 决定)
    # 我们可以等到 reset 后获取，或者在 config 里硬编码 rl_batch_size
    buffer = RolloutBuffer(rl_cfg, device=rl_cfg.device)

    # Reward Calculator: 需要传入 Group Names 以保证权重顺序对齐
    # 假设 cfg 里有 reward_config
    reward_calc = RewardCalculator(cfg, env_group_names=env.group_names)

    # 9. 实例化 Trainer
    trainer = PPOTrainer(
        config=rl_cfg,
        agent=agent,
        env=env,
        buffer=buffer,
        reward_calculator=reward_calc,
        train_loader=train_loader
    )

    # 10. 开始训练
    trainer.train()


if __name__ == "__main__":
    main()
