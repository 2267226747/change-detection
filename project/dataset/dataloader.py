# 封装 DataLoader，定义 Batch 采样策略
# data/dataloader.py
import torch
from torch.utils.data import DataLoader
from .dataset import SVIPairsDataset


def svi_collate_fn(batch):
    """
    处理变长 Patch 数量的 Batch
    """
    pixel_values_t1_list = []
    pixel_values_t2_list = []
    num_patches_t1 = []
    num_patches_t2 = []
    labels_list = []
    oids = []

    for item in batch:
        pixel_values_t1_list.append(item['pixel_values_t1'])
        pixel_values_t2_list.append(item['pixel_values_t2'])

        # 记录每张图对应的 patch 数量，这对于后续模型拆分至关重要
        num_patches_t1.append(item['num_patches_t1'])
        num_patches_t2.append(item['num_patches_t2'])

        labels_list.append(item['labels'])
        oids.append(item['oid'])

    # #  使用 stack
    # #  List of [N, 3, H, W] -> Tensor [Batch, N, 3, H, W]
    # batch_pixel_values_t1 = torch.stack(pixel_values_t1_list, dim=0)
    # batch_pixel_values_t2 = torch.stack(pixel_values_t2_list, dim=0)

    # 扁平化拼接 (Flatten Concatenation)
    # List of [N, 3, H, W] -> [Batch*N, 3, 224, 224]
    batch_pixel_values_t1 = torch.cat(pixel_values_t1_list, dim=0)
    batch_pixel_values_t2 = torch.cat(pixel_values_t2_list, dim=0)

    # 堆叠标签
    batch_labels = torch.stack(labels_list, dim=0)

    # 转换为 Tensor 方便传入模型
    batch_num_patches_t1 = torch.tensor(num_patches_t1, dtype=torch.long)
    batch_num_patches_t2 = torch.tensor(num_patches_t2, dtype=torch.long)

    return {
        'pixel_values_t1': batch_pixel_values_t1,  # [B*N, 3, H, W]
        'pixel_values_t2': batch_pixel_values_t2,  # [B*N, 3, H, W]
        'num_patches_t1': batch_num_patches_t1,
        'num_patches_t2': batch_num_patches_t2,
        'labels': batch_labels,
        'oid': oids  # List[str]
    }


def build_dataloader(cfg, logger, split='train'):
    """
    对外暴露的接口函数
    """
    dataset = SVIPairsDataset(cfg, logger, split=split)

    shuffle = True if split == 'train' else False
    # 确保 drop_last，避免最后一个 batch 只有一个样本导致 BatchNorm 出错（如果用了 BN）
    # 对于 Transform/Encoder 结构，通常建议 drop_last=True (训练时)
    drop_last = True if split == 'train' else False

    loader = DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        shuffle=shuffle,
        collate_fn=svi_collate_fn,  # 挂载自定义的 collate_fn
        pin_memory=True,
        drop_last=drop_last
    )

    if split == "train":  # 仅在training数据上输出一遍即可
        logger.info(
            f"Batch size: {cfg.data.batch_size}, Num workers: {cfg.data.num_workers}")

    return loader

