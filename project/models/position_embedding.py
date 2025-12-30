import torch
import torch.nn as nn


class PositionalEmbedder(nn.Module):
    def __init__(self, cfg):
        """
        Args:
            cfg: 配置对象，需包含以下属性:
                 - patches_max_num: int, 最大切片数量 (e.g., 128)
                 - image_patches_size: int, 切片分辨率 (e.g., 224)
                 - patches_min_num: int (可选)
        """
        super().__init__()
        # 从 cfg 获取参数
        cfg = cfg.data
        # 考虑到极端长宽比 (例如 1xN)，我们将 row/col 的 embedding 词表大小设为 20
        # 是为了给缩略图 (Thumbnail) 预留一个特殊的 "溢出/汇总" 索引
        self.max_grid_size = 20
        self.image_patches_size = getattr(cfg, 'image_patches_size', 224)
        self.dim = getattr(cfg, 'vision_dim', 4096)  # 经过unshuffle后的vision token的维度
        self.batch_size = getattr(cfg, 'batch_size', 16)
        self.H_num = getattr(cfg, 'patches_h_num', 3)
        self.W_num = getattr(cfg, 'patches_w_num', 9)

        # 每个patches的token数量（经过unshuffle后的）
        self.num_local_tokens = int((self.image_patches_size / 14) ** 2 / 4)

        # -----------------------------------------------------------
        # 1. 局部位置编码 (Local): 学习子块内部纹理
        # -----------------------------------------------------------
        self.local_pos = nn.Parameter(torch.randn(1, self.num_local_tokens, self.dim) * 0.02)

        # -----------------------------------------------------------
        # 2. 全局位置编码 (Global): 学习 Tile 的行列关系
        # -----------------------------------------------------------
        # 词表大小设为 max_grid_size，足以覆盖任何 h < max_num 或 w < max_num 的情况
        self.row_embed = nn.Embedding(self.max_grid_size, self.dim)
        self.col_embed = nn.Embedding(self.max_grid_size, self.dim)

        # -----------------------------------------------------------
        # 3. 属性编码 (Attributes)
        # -----------------------------------------------------------
        self.image_embed = nn.Embedding(2, self.dim)  # 0: Pre, 1: Post
        self.type_embed = nn.Embedding(2, self.dim)  # 0: Crop, 1: Thumbnail

    def forward(self, image_time=None):
        """
        Args:
            image_time: [Batch] or [Batch, 1] or None.
                        用于标识图片时序 (0: Pre, 1: Post)。
                        默认为 None，此时不叠加该属性编码。
        """
        device = self.row_embed.weight.device
        batch_size = self.batch_size
        # 确定总 Tile 数量
        total_tiles = self.H_num * self.W_num + 1

        # -----------------------------------------------------------
        # A. 构建 Mask (Type & Image)
        # -----------------------------------------------------------

        # 1. 构建 is_thumbnail_mask (标识 Crop/Thumbnail)
        # 逻辑: 前 H*W 个是 Crop(0)，最后一个是 Thumbnail(1)
        type_mask = torch.zeros(total_tiles, dtype=torch.long, device=device)
        type_mask[-1] = 1
        # [Total_Tiles] -> [B, Total_Tiles]
        type_mask_expanded = type_mask.unsqueeze(0).expand(batch_size, -1)

        # 2. 处理 image_time (标识 Pre/Post) - 【条件执行】
        img_pe = 0.0  # 默认为 0，不影响后续加法
        if image_time is not None:
            # 校验输入数值是否合法 (仅允许 0 或 1)
            if isinstance(image_time, torch.Tensor):
                # 检查是否只包含 0 和 1
                assert torch.all((image_time == 0) | (image_time == 1)), "image_time must be 0 or 1"
                image_time = image_time.long().to(device)
                # 扩张: [B] -> [B, 1] -> [B, Total_Tiles]
                image_time_expanded = image_time.view(batch_size, 1).expand(-1, total_tiles)
            else:
                # 处理传入 int 的情况
                assert image_time in [0, 1], "image_time must be 0 or 1"
                image_time_expanded = torch.full((batch_size, total_tiles), image_time, dtype=torch.long, device=device)

            # 查表: (B, Total_Tiles) -> (B, Total_Tiles, Local, D)
            # 只有这里执行了，img_pe 才会变成 Tensor，否则保持为 0.0
            img_pe = self.image_embed(image_time_expanded).unsqueeze(2)  # 待会儿广播

        # -----------------------------------------------------------
        # B. 构建全局坐标网格 (Global Grid)
        # -----------------------------------------------------------

        # 1. 生成基础 Grid 坐标 (针对普通切片)
        # 限制坐标索引不超过 embedding 范围 (虽然逻辑上 H, W 应该小于 patches_max_num)
        y_idxs = torch.arange(self.H_num, device=device).repeat_interleave(self.W_num)
        x_idxs = torch.arange(self.W_num, device=device).repeat(self.H_num)

        # 2. 处理缩略图 (Thumbnail)
        # 缩略图没有实际的 Grid 坐标，我们将其分配到 "最后一位" 的特殊坐标
        # 使用词表最后一个索引作为缩略图的特殊位置
        thumb_idx = self.max_grid_size - 1
        extra_y = torch.tensor([thumb_idx], device=device)
        extra_x = torch.tensor([thumb_idx], device=device)
        y_idxs = torch.cat([y_idxs, extra_y])
        x_idxs = torch.cat([x_idxs, extra_x])

        # -----------------------------------------------------------
        # C. 查表获取 Embedding
        # -----------------------------------------------------------

        # 1. Global: (Total_Tiles, D)
        # 这里的索引必须要 clamp 或者保证 H/W 不超过初始化时的 self.max_grid_size
        global_pe = self.row_embed(y_idxs) + self.col_embed(x_idxs)
        # 扩展维度: (Total_Tiles, D) -> (B, Total_Tiles, Local_Tokens, D)
        global_pe = global_pe.unsqueeze(0).unsqueeze(2).expand(self.batch_size, -1, self.local_pos.size(1), -1)

        # 2. Local: (1, Local_Tokens, D) -> (B, Total_Tiles, Local_Tokens, D)
        local_pe = self.local_pos.unsqueeze(0).expand(self.batch_size, total_tiles, -1, -1)

        # 3. Attributes: (B, Total_Tiles) -> (B, Total_Tiles, Local_Tokens, D)
        # 如果 img_pe 是 tensor，这里也需要 expand 到 Local 维度
        if isinstance(img_pe, torch.Tensor):
            img_pe = img_pe.expand_as(local_pe)
        # Type PE
        type_pe = self.type_embed(type_mask_expanded).unsqueeze(2).expand_as(local_pe)

        # -----------------------------------------------------------
        # C. 融合与输出
        # -----------------------------------------------------------
        final_pos = local_pe + global_pe + img_pe + type_pe

        return final_pos.flatten(1, 2)


# 测试代码
class Config:
    def __init__(self):
        self.data = type('Data', (), {
            'image_patches_size': 448,
            'vision_dim': 512,  # 使用小一点的维度方便测试
            'batch_size': 4,
            'patches_h_num': 3,
            'patches_w_num': 9
        })()


def test_unified_positional_embedder():
    print("测试 UnifiedPositionalEmbedder...")
    print("-" * 50)

    # 创建配置和模型
    cfg = Config()
    model = PositionalEmbedder(cfg)

    # 打印模型信息
    print(f"模型配置:")
    print(f"  H_num: {model.H_num}, W_num: {model.W_num}")
    print(f"  image_patches_size: {model.image_patches_size}")
    print(f"  dim: {model.dim}")
    print(f"  batch_size: {model.batch_size}")
    print(f"  max_grid_size: {model.max_grid_size}")
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
        print(f"  位置({row},{col})与(0,0)的差异: {diff_to_first.item():.6f}")

    print("  ✓ 测试5通过")
    print()

    # 测试6：检查参数数量
    print("测试6: 参数统计")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"  总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")

    # 计算各部分参数
    local_params = model.local_pos.numel()
    row_embed_params = model.row_embed.weight.numel()
    col_embed_params = model.col_embed.weight.numel()
    image_embed_params = model.image_embed.weight.numel()
    type_embed_params = model.type_embed.weight.numel()

    print(f"  局部位置编码: {local_params:,}")
    print(f"  行嵌入: {row_embed_params:,}")
    print(f"  列嵌入: {col_embed_params:,}")
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
