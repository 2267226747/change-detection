import torch
import argparse
import os
from types import SimpleNamespace
import yaml

# 引入你的 RL 模块
from rl.env import RLEnv
from rl.networks import ActorCriticNetwork
from rl.agent import PPOAgent
from rl.buffer import RolloutBuffer
from rl.rewards import RewardCalculator
from trainer.rl_trainer import PPOTrainer


# 引入你的预训练模型定义 (假设在 models 目录下)
# from models.assembled_model import AssembledFusionModel
# from utils.dataloader import create_dataloader

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
    cfg = get_config()
    print("Configuration loaded.")

    # 2. 准备数据 (Mockup)
    # 这里需要替换为你真实的数据加载逻辑
    # 关键点: batch_size 对应并行环境数, drop_last=True
    print("Loading data...")
    # train_loader = create_dataloader(batch_size=32, shuffle=True, drop_last=True)
    # 模拟一个 DataLoader
    train_loader = [
        {
            'pixel_values_1': torch.randn(32 * 256, 3, 224, 224),  # 假设 N=256 patches
            'pixel_values_2': torch.randn(32 * 256, 3, 224, 224),
            'labels': torch.randint(0, 2, (32, 53))  # 假设总共 53 个子任务
        }
        for _ in range(10)
    ]  # 仅用于演示代码跑通

    # 3. 加载预训练模型 (Mockup)
    print("Loading pretrained model...")

    # pretrained_model = AssembledFusionModel(model_cfg)
    # pretrained_model.load_state_dict(torch.load("path/to/pretrained.pth"))
    # pretrained_model.to(cfg.device)

    # 假设这里有一个 Mock 对象用于演示
    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.class_heads = torch.nn.ModuleList([torch.nn.Linear(10, 10) for _ in range(6)])
            # 必须模拟必要的属性
            self.full_cfg = SimpleNamespace(data=SimpleNamespace(patches_num=256))
            self.start_classify = 1
            self.class_heads[0].group_names = ['road', 'building']
            self.class_heads[0].sub_counts = [2, 3]  # Total=5

        def parameters(self): return iter([torch.randn(1)])  # Dummy

    pretrained_model = MockModel().to(cfg.device)

    # 4. 实例化环境
    env = TransformerRLEnv(
        pretrained_model=pretrained_model,
        config=cfg,
        device=cfg.device,
        freeze_classifier=cfg.freeze_classifier
    )
    print("Environment initialized.")

    # 5. 定义网络形状 (用于构建 RL Network)
    # 根据 Env 实际解析出的结构
    env_shapes = {
        'query_dim': 1024,  # 需与预训练模型一致
        'vision_dim': 1024,
        'num_groups': env.num_groups,
        'tokens_per_group': env.batch_size // env.num_groups if env.batch_size > 0 else 256,  # 动态获取或硬编码
        'total_subtasks': env.total_subtasks
    }

    # 6. 实例化 Actor-Critic Network
    network = ActorCriticNetwork(cfg, env_shapes).to(cfg.device)

    # 7. 实例化 Agent
    # [关键] 传入分类头引用和参数，实现联合优化
    agent = PPOAgent(
        network=network,
        classifier_heads=env.model.class_heads,
        classifier_params=env.get_classifier_parameters(),
        config=cfg
    ).to(cfg.device)

    # 8. 实例化辅助组件
    # Buffer: 注意 buffer_size = num_steps * env_batch_size
    # 这里 env_batch_size 是隐式的 (由 train_loader 的 batch_size 决定)
    # 我们可以等到 reset 后获取，或者在 config 里硬编码 rl_batch_size
    buffer = RolloutBuffer(cfg, env_shapes, device=cfg.device)

    # Reward Calculator: 需要传入 Group Names 以保证权重顺序对齐
    # 假设 cfg 里有 reward_config
    reward_calc = RewardCalculator(cfg, env_group_names=env.group_names)

    # 9. 实例化 Trainer
    trainer = PPOTrainer(
        config=cfg,
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