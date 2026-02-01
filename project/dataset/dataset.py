# 负责从硬盘读取 t1, t2 时刻的图像对以及对应的多任务标签
# project/data/dataset.py
import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from dataset.transforms import dynamic_preprocess, fixed_preprocess
from models.vision.backbone import VisionEncoder
import torchvision.transforms.functional as F


class SVIPairsDataset(Dataset):
    def __init__(self, cfg, logger, split='train'):
        self.cfg = cfg
        self.split = split

        # --- 切片配置 ---
        # 在 cfg 中配置这些参数
        self.min_num = getattr(self.cfg.data, 'patches_min_num', 1)
        self.max_num = getattr(self.cfg.data, 'patches_max_num', 128)
        self.h_num = getattr(self.cfg.data, 'patches_h_num', 3)
        self.w_num = getattr(self.cfg.data, 'patches_w_num', 9)
        self.image_patches_size = getattr(self.cfg.data, 'image_patches_size', 224)
        self.use_thumbnail = getattr(self.cfg.data, 'use_thumbnail', True)
        # ----------------
        if split == "train":  # 仅在training数据上输出一遍即可
            logger.info(
                f"Patches H/W num: {self.h_num}/{self.w_num}, "
                f"Image patches size: {self.image_patches_size}, "
                f"Use thumbnaill: {self.use_thumbnail}")

        # 路径配置
        if split == 'train':
            self.image_dir = self.cfg.data.train_image_dir
            csv_path = self.cfg.data.train_csv_path
        else:
            self.image_dir = self.cfg.data.val_image_dir
            csv_path = self.cfg.data.val_csv_path

        print(f"[{split.upper()}] Loading metadata from {csv_path} ...")
        self.df = pd.read_csv(csv_path)

        self.col_oid = self.cfg.data.col_oid
        self.col_t1 = self.cfg.data.col_name_t1
        self.col_t2 = self.cfg.data.col_name_t2
        self.label_cols = self.cfg.data.label_columns

        # 初始化 Processor (来自 model/vision/backbone.py)
        # processor 主要做 Normalize 和 ToTensor
        self.processor = VisionEncoder.get_image_processor(cfg.model.vision.backbone)

    def __len__(self):
        return len(self.df)

    def _process_single_image(self, image):
        """
        辅助函数：
        1. 切片 (transforms.py)
        2. 归一化 (processor)
        3. 堆叠 (torch.cat)
                Returns:
                    tensor: [N, 3, H, W]
                    N = h_num * w_num + (1 if thumbnail)
        """
        # 1. 切片 -> 得到 List[PIL.Image]
        # patches = dynamic_preprocess(
        #     image,
        #     min_num=self.min_num,
        #     max_num=self.max_num,
        #     image_patches_size=self.image_patches_size,
        #     use_thumbnail=self.use_thumbnail
        # )

        patches = fixed_preprocess(
            image,
            h_num=self.h_num,
            w_num=self.w_num,
            image_patches_size=self.image_patches_size,
            use_thumbnail=self.use_thumbnail
        )

        # 2. 对每个 Patch 单独处理
        pixel_values_list = []
        for patch in patches:
            # processor 处理单张图，返回 dict, 取 pixel_values
            # 这里的 patch 已经是 224x224 了，processor 主要做 Normalize
            inputs = self.processor(images=patch, return_tensors='pt', do_resize=False)
            pixel_values_list.append(inputs.pixel_values)  # shape [1, 3, 448, 448]

            # # test用，不调用internVit processor
            # pixel_values = F.to_tensor(patch)
            # pixel_values_list.append(pixel_values.unsqueeze(0))

        # 3. 拼接所有 patches
        # 结果 shape: [Num_Patches, 3, 224, 224]
        return torch.cat(pixel_values_list, dim=0)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        oid = str(row[self.col_oid])
        fname_t1 = f"pair{oid}_t1_{str(row[self.col_t1])}.png"
        fname_t2 = f"pair{oid}_t2_{str(row[self.col_t2])}.png"

        path_t1 = os.path.join(self.image_dir, fname_t1)
        path_t2 = os.path.join(self.image_dir, fname_t2)

        try:
            image_t1 = Image.open(path_t1).convert('RGB')
            image_t2 = Image.open(path_t2).convert('RGB')
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Image not found: {path_t1} or {path_t2}") from e

        # 处理两张图片
        # 每一张图都会变成一堆 Patches
        pixel_values_t1 = self._process_single_image(image_t1)
        pixel_values_t2 = self._process_single_image(image_t2)

        labels = torch.tensor(row[self.label_cols].values.astype(float), dtype=torch.float32)

        return {
            'pixel_values_t1': pixel_values_t1,  # [N, 3, 448, 448]
            'pixel_values_t2': pixel_values_t2,  # [N, 3, 448, 448]
            'num_patches_t1': pixel_values_t1.shape[0],  # 记录切了多少块
            'num_patches_t2': pixel_values_t2.shape[0],
            'labels': labels,
            'oid': oid
        }


class MockSVIPairsDataset(Dataset):
    """
    [Feature Mode]
    直接模拟 Vision Encoder 的输出特征。
    用于跳过视觉部分，直接测试 Transformer/RL/Head。
    """

    def __init__(self, cfg, split='train', length=128):
        self.cfg = cfg
        self.length = length
        self.num_tasks = len(cfg.data.label_columns)

        # 1. 获取 InternViT 的输出维度
        self.feature_dim = cfg.model.vision.feature_dim  # 1024

        # 2. 计算 Sequence Length
        # 逻辑: (Image_Size // Patch_Size)^2 + 1 (CLS_Token)
        # InternViT Patch Size 默认为 14
        image_size = cfg.data.image_size  # 448
        patch_size = 14
        num_patches = (image_size // patch_size) ** 2
        self.seq_len = num_patches + 1  # 1024 + 1 = 1025

        print(f"[{split.upper()}] 使用 MOCK 特征数据集 (模拟 InternViT 输出)")
        print(f"Mock Feature Shape: [Seq_Len={self.seq_len}, Dim={self.feature_dim}]")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # 1. 生成随机特征张量 [1025, 1024]
        # 模拟 InternViT 提取后的特征 (features)
        feature_t1 = torch.randn(self.seq_len, self.feature_dim, dtype=torch.float32)
        feature_t2 = torch.randn(self.seq_len, self.feature_dim, dtype=torch.float32)

        # 2. 生成随机标签
        labels = torch.randint(0, 2, (self.num_tasks,), dtype=torch.float32)

        oid = f"mock_{idx}"

        return {
            # 注意：这里键名改为了 feature_t1，而不是 pixel_values
            # 这样后续模型代码里可以区分是 "图片" 还是 "特征"
            'feature_t1': feature_t1,
            'feature_t2': feature_t2,
            'labels': labels,
            'oid': oid
        }


# ================= 测试代码 =================
if __name__ == "__main__":
    import sys

    # 把 project 根目录加到路径以便导入 utils
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
    from utils.config import Config
    from dataset.dataloader import build_dataloader

    # 1. 读取配置
    cfg = Config.from_yaml("../configs/defaults.yaml")

    # 2. 临时修改路径为真实路径进行测试
    cfg.data.train_image_dir = r"G:\MSc dissertation\LLM_VLM test\SVI image(panorama)_Wuhan"
    cfg.data.train_csv_path = r"G:\MSc dissertation\LLM_VLM test\data_samples_Wuhan_v2_val_copy.csv"

    loader = build_dataloader(cfg, split='train')

    for batch in loader:
        print("T1 Shape:", batch['pixel_values_t1'].shape)
        # 预期: [Total_Patches, 3, 448, 448]，例如 [10, 3, 448, 448]

        print("T1 Num Patches:", batch['num_patches_t1'])
        # 预期: [Batch_Size]，例如 tensor([5, 5])

        print("Labels Shape:", batch['labels'].shape)
        # 预期: [2, Num_Labels]

        print("Test Passed!")
        break
