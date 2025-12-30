# Step 1: 继承 BaseTrainer，仅训练深度学习部分
# train.py
from data.dataloader import build_dataloader

# ... 你的 cfg 定义 ...

# 1. 构建 DataLoader
train_loader = build_dataloader(cfg, split='train')

for batch in train_loader:
    # 2. 获取数据
    # pv1 的形状是 [Total_Patches, 3, 448, 448]，不是 [Batch, N, 3...]
    pv1 = batch['pixel_values_t1'].cuda()
    num1 = batch['num_patches_t1'].cuda()  # [Batch_Size]

    pv2 = batch['pixel_values_t2'].cuda()
    num2 = batch['num_patches_t2'].cuda()

    labels = batch['labels'].cuda()

    # 3. 传入模型
    # 模型 Forward 需要同时接收 像素值 和 Patch数量
    outputs = model(pv1, num1, pv2, num2)

    # ... 计算 loss ...