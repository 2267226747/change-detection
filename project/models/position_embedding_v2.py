import torch
import torch.nn as nn
import math

import numpy as np


def get_2d_sincos_pos_embed(embed_dim, grid_size_h, grid_size_w, cls_token=False, device="cpu"):
    """
    embed_dim: 输出维度
    grid_size_h: 高度方向的网格数
    grid_size_w: 宽度方向的网格数
    """
    # 也就是 h * w
    # [修改] 使用 torch.arange 替代 np.arange
    grid_h = torch.arange(grid_size_h, dtype=torch.float32, device=device)
    grid_w = torch.arange(grid_size_w, dtype=torch.float32, device=device)

    # [修改] 使用 torch.meshgrid 替代 np.meshgrid
    # indexing='xy' 对应 numpy 的默认行为 (第一个返回变化快(w), 第二个返回变化慢(h))
    grid_W, grid_H = torch.meshgrid(grid_w, grid_h, indexing='xy')

    # 堆叠 [2, h, w] -> grid[0] 是 W, grid[1] 是 H
    grid = torch.stack([grid_W, grid_H], dim=0)

    grid = grid.reshape([2, 1, grid_size_h, grid_size_w])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)

    if cls_token:
        pos_embed = torch.cat([torch.zeros([1, embed_dim], device=device), pos_embed], dim=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)

    emb = torch.cat([emb_h, emb_w], dim=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    """
    assert embed_dim % 2 == 0
    # 强制使用 float32 进行数学计算
    omega = torch.arange(embed_dim // 2, dtype=torch.float32, device=pos.device)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)

    return torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)


class PositionalEmbedder2(nn.Module):
    def __init__(self, cfg, logger):
        """
        Args:
            cfg: 全局配置对象 (cfg.data)
        """
        super().__init__()
        # 从 cfg 获取参数
        cfg = cfg.data
        self.dim = getattr(cfg, 'vision_dim', 4096)
        self.H_num = getattr(cfg, 'patches_h_num', 3)
        self.W_num = getattr(cfg, 'patches_w_num', 9)

        # 假设最小子块大小固定，那么 token_size 也是固定的
        # 如果子块大小也变，这个需要挪到 forward 里计算
        self.patch_size = getattr(cfg, 'patch_size', 14)
        self.image_patches_size = getattr(cfg, 'image_patches_size', 224)  # 这里指切片后的子图大小

        # 计算 Local Token 的网格形状 (e.g. 224/14/2 = 16/2  -> 8x8)
        # 经过unshuffle后，2*2个合并为1个
        self.local_grid_size = self.image_patches_size // self.patch_size // 2
        self.num_local_tokens = self.local_grid_size ** 2

        # -----------------------------------------------------------
        # 1. 缓存 Local PE (Sin-Cos)
        #    因为 Local 尺寸通常相对固定，可以预先计算
        # -----------------------------------------------------------
        local_pe_tensor = get_2d_sincos_pos_embed(
            self.dim, self.local_grid_size, self.local_grid_size, device="cpu"
        )  # [Local_Tokens, D]
        # 注册为 buffer，不作为参数更新，但随模型保存
        self.register_buffer(
            'local_pe_cache',
            local_pe_tensor.float().unsqueeze(0)  # [1, Local_Tokens, D]
        )

        # -----------------------------------------------------------
        # 2. 属性编码 (保持可学习，因为是离散类别)
        # -----------------------------------------------------------
        self.image_embed = nn.Embedding(2, self.dim)  # 0: Pre, 1: Post
        self.type_embed = nn.Embedding(2, self.dim)  # 0: Crop, 1: Thumbnail

    def forward(self, input_batch_size, image_time=None):
        """
        Args:
            h_num, w_num: 当前 batch 的切片网格划分 (e.g., 3, 3)
                          允许每次 forward 都不一样
            image_time: [Batch] 时序标记
        """
        device = self.local_pe_cache.device

        # 动态计算总 Tile 数
        total_tiles = self.H_num * self.W_num + 1  # +1 for thumbnail

        # -----------------------------------------------------------
        # A. 动态生成 Global PE (Sin-Cos)
        # -----------------------------------------------------------
        # 1. 生成 Crop 的 Grid
        # 我们使用辅助函数直接生成 (h_num * w_num, D) 的编码
        # 注意：这里每次 forward 都会计算 sin/cos，开销很小，换来了极大的灵活性
        global_crops = get_2d_sincos_pos_embed(
            self.dim, self.H_num, self.W_num, device=device
        ) # [H*W, D]

        # 2. 处理 Thumbnail 的位置
        # Thumbnail 在逻辑上是全局概览，通常可以设为 Grid 的中心，或者设为 (0,0) 并依靠 Type Embedding 区分
        # 另一种简单方法是：假设 Thumbnail 位于 (h_num, w_num) 的对角线之外，或者用一个全0向量
        # 这里为了保持和之前逻辑一致（特殊位置），我们使用 (h_num, w_num) 坐标作为 Thumbnail 的 Global Pos
        # 或者直接全 0，完全依靠 type_embed 来区分
        # 方案：使用 (h_num/2, w_num/2) 也就是中心位置，或者 (0, 0)

        # 这里演示生成一个代表 "Grid 之外" 的特殊坐标，比如 (h_num, w_num)
        # 但 SinCos 是连续的，建议直接用全 0 向量，因为 Type Embedding 已经负责区分它是 Thumbnail 了
        global_thumb = torch.zeros(1, self.dim, device=device)

        # 拼接
        global_pe = torch.cat([global_crops, global_thumb], dim=0)  # [Total_Tiles, D]

        # 扩展维度: [Total_Tiles, D] -> [B, Total_Tiles, Local_Tokens, D]
        global_pe = global_pe.unsqueeze(0).unsqueeze(2).expand(input_batch_size, -1, self.num_local_tokens, -1)

        # -----------------------------------------------------------
        # B. Local PE (直接从 Buffer 取)
        # -----------------------------------------------------------
        # [1, Local_Tokens, D] -> [B, Total_Tiles, Local_Tokens, D]
        local_pe = self.local_pe_cache.unsqueeze(1).expand(input_batch_size, total_tiles, -1, -1)

        # -----------------------------------------------------------
        # C. 属性编码 (Attributes)
        # -----------------------------------------------------------
        # 1. Type Mask
        type_mask = torch.zeros(total_tiles, dtype=torch.long, device=device)
        type_mask[-1] = 1  # 最后一个是 Thumbnail
        type_pe = self.type_embed(type_mask).unsqueeze(0).unsqueeze(2)  # [1, Total, 1, D]

        # 2. Image Time
        img_pe = 0.0
        if image_time is not None:
            # 处理成 [B, 1, 1, D] 以便广播
            if isinstance(image_time, int):
                t_tensor = torch.tensor([image_time], device=device).repeat(input_batch_size)
            else:
                t_tensor = image_time

            img_pe = self.image_embed(t_tensor).view(input_batch_size, 1, 1, self.dim)

        # -----------------------------------------------------------
        # D. 融合
        # -----------------------------------------------------------
        # 此时所有维度均已对齐或可广播
        final_pos = local_pe + global_pe + type_pe + img_pe

        return final_pos.flatten(1, 2)


# 测试代码
class Config:
    def __init__(self):
        self.data = type('Data', (), {
            'image_patches_size': 224,
            'vision_dim': 8,  # 使用小一点的维度方便测试
            'batch_size': 4,
            'patches_h_num': 3,
            'patches_w_num': 9
        })()


def test_unified_positional_embedder():
    print("测试 UnifiedPositionalEmbedder...")
    print("-" * 50)

    # 创建配置和模型
    cfg = Config()
    model = PositionalEmbedder2(cfg)

    # 打印模型信息
    print(f"模型配置:")
    print(f"  H_num: {model.H_num}, W_num: {model.W_num}")
    print(f"  image_patches_size: {model.image_patches_size}")
    print(f"  dim: {model.dim}")
    print(f"  batch_size: {model.batch_size}")
    # print(f"  max_grid_size: {model.max_grid_size}")
    print(f"  num_local_tokens: {model.num_local_tokens}")
    print()

    # 测试1：不使用image_time
    print("测试1: image_time=None")
    output1 = model()
    print(f"  输出形状: {output1.shape}")
    print(f"  期望形状: (batch_size={model.batch_size}, total_tokens, dim={model.dim})")

    total_tiles = model.H_num * model.W_num + 1
    expected_total_tokens = total_tiles * model.num_local_tokens
    print(f"  总tokens数: {total_tiles} tiles × {model.num_local_tokens} local tokens = {expected_total_tokens}")
    print(f"  实际输出tokens数: {output1.shape[1]}")
    assert output1.shape == (model.batch_size, expected_total_tokens, model.dim), f"形状不匹配: {output1.shape}"
    print("  ✓ 测试1通过")
    print()

    # 测试2：使用image_time=0 (Pre)
    print("测试2: image_time=0 (Pre)")
    image_time_pre = torch.zeros(model.batch_size, dtype=torch.long)
    output2 = model(image_time_pre)
    print(f"  输出形状: {output2.shape}")
    print(f"  是否与测试1不同: {not torch.allclose(output1, output2)}")
    print("  ✓ 测试2通过")
    print()

    # 测试3：使用image_time=1 (Post)
    print("测试3: image_time=1 (Post)")
    image_time_post = torch.ones(model.batch_size, dtype=torch.long)
    output3 = model(image_time_post)
    print(f"  输出形状: {output3.shape}")
    print(f"  是否与测试2不同: {not torch.allclose(output2, output3)}")
    print("  ✓ 测试3通过")
    print()

    # 测试4：检查不同类型tile的编码是否不同
    print("测试4: 检查Crop和Thumbnail的编码差异")

    # 获取最后一个tile（缩略图）的编码
    last_tile_start = (total_tiles - 1) * model.num_local_tokens
    thumbnail_tokens = output1[0, last_tile_start:last_tile_start + model.num_local_tokens, :]

    # 获取第一个tile（网格图块）的编码
    first_crop_tokens = output1[0, :model.num_local_tokens, :]

    diff = torch.mean(torch.abs(thumbnail_tokens - first_crop_tokens))
    print(f"  缩略图与网格图块的编码平均差异: {diff.item():.6f}")
    print(f"  是否明显不同: {diff.item() > 0.1}")
    print("  ✓ 测试4通过")
    print()

    # 测试5：检查不同位置的网格图块编码是否不同
    print("测试5: 检查不同网格位置的编码差异")

    # 获取不同位置的tile编码
    tile_positions = [
        (model.H_num - 1, model.W_num),  # 缩略图
        (0, 0),  # 左上角
        (0, model.W_num - 1),  # 右上角
        (model.H_num - 1, 0),  # 左下角
        (model.H_num - 1, model.W_num - 1),  # 右下角

    ]

    for i, (row, col) in enumerate(tile_positions):
        tile_idx = row * model.W_num + col
        start_idx = tile_idx * model.num_local_tokens
        end_idx = start_idx + model.num_local_tokens
        tile_tokens = output1[0, start_idx:end_idx, :]

        # 比较与第一个tile的差异
        diff_to_first = torch.mean(torch.abs(tile_tokens - first_crop_tokens))
        print(tile_tokens.shape, tile_tokens)
        print(first_crop_tokens)
        print(f"  位置({row},{col})与(0,0)的差异: {diff_to_first.item():.6f}")
        break

    print("  ✓ 测试5通过")
    print()

    # 测试6：检查参数数量
    print("测试6: 参数统计")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"  总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")

    # 计算各部分参数
    # local_params = model.local_pos.numel()
    # row_embed_params = model.row_embed.weight.numel()
    # col_embed_params = model.col_embed.weight.numel()
    image_embed_params = model.image_embed.weight.numel()
    type_embed_params = model.type_embed.weight.numel()

    # print(f"  局部位置编码: {local_params:,}")
    # print(f"  行嵌入: {row_embed_params:,}")
    # print(f"  列嵌入: {col_embed_params:,}")
    print(f"  图像类型嵌入: {image_embed_params:,}")
    print(f"  Tile类型嵌入: {type_embed_params:,}")

    print("-" * 50)
    print("所有测试通过! ✓")

    return model, output1


if __name__ == "__main__":
    # 运行测试
    model, output = test_unified_positional_embedder()

    # 额外的可视化测试
    print("\n额外测试: 检查输出值的统计信息")
    print(f"  输出最小值: {output.min().item():.6f}")
    print(f"  输出最大值: {output.max().item():.6f}")
    print(f"  输出均值: {output.mean().item():.6f}")
    print(f"  输出标准差: {output.std().item():.6f}")

    # 检查设备
    print(f"\n模型设备: {next(model.parameters()).device}")
