# data/transforms.py
import math
from PIL import Image


def dynamic_preprocess(image, min_num=1, max_num=32, image_patches_size=448, use_thumbnail=True):
    """

    动态切片逻辑
    Args:
        image: PIL Image
        min_num (int): 最少切片数
        max_num (int): 最多切片数
        image_patches_size (int): 基础切片分辨率
        use_thumbnail (bool): 是否附加全局缩略图
    Returns:
        List[PIL.Image]: 切片后的图片列表

    物理分辨率优先
    对于 4096x1280，优先匹配 9x3，而不是按长宽比去匹配 10x3
    """
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # -----------------------------------------------------------
    # 计算基于物理分辨率的"最佳"网格 (Direct Rounding)
    # -----------------------------------------------------------
    # 直接计算原图能塞下几个 448
    # 4096 / 448 = 9.14 -> 9
    # 1280 / 448 = 2.85 -> 3
    prior_blocks_w = max(1, round(orig_width / image_patches_size))
    prior_blocks_h = max(1, round(orig_height / image_patches_size))

    # 检查这个"物理最佳组合"是否在 max_num 允许范围内
    if prior_blocks_w * prior_blocks_h <= max_num:
        best_grid = (prior_blocks_w, prior_blocks_h)
    else:
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
            i * j <= max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # 找最接近的长宽比
        best_grid = min(target_ratios, key=lambda x: abs(x[0] / x[1] - aspect_ratio))

    patches_w, patches_h = best_grid

    # Resize
    target_width = image_patches_size * patches_w
    target_height = image_patches_size * patches_h
    resized_img = image.resize((target_width, target_height))

    # Crop
    processed_images = []
    for i in range(patches_h):
        for j in range(patches_w):
            box = (
                j * image_patches_size,
                i * image_patches_size,
                (j + 1) * image_patches_size,
                (i + 1) * image_patches_size
            )
            processed_images.append(resized_img.crop(box))

    # Thumbnail
    if use_thumbnail and len(processed_images) > 0:
        thumbnail_img = image.resize((image_patches_size, image_patches_size))
        processed_images.append(thumbnail_img)

    return processed_images


def fixed_preprocess(image, h_num=1, w_num=1, image_patches_size=448, use_thumbnail=True):
    """
    固定切片逻辑
    Args:
        image: PIL Image
        h_num (int): 纵向切片数量
        w_num (int): 横向切片数量
        image_patches_size (int): 每个切片的分辨率
        use_thumbnail (bool): 是否附加全局缩略图
    Returns:
        List[PIL.Image]: 切片后的图片列表
    """
    # orig_width, orig_height = image.size

    # 计算目标分辨率
    target_width = w_num * image_patches_size
    target_height = h_num * image_patches_size

    # Resize到目标分辨率（注意：这可能会导致图像变形）
    resized_img = image.resize((target_width, target_height))

    # 切片
    processed_images = []
    for i in range(h_num):
        for j in range(w_num):
            box = (
                j * image_patches_size,
                i * image_patches_size,
                (j + 1) * image_patches_size,
                (i + 1) * image_patches_size
            )
            processed_images.append(resized_img.crop(box))

    # 缩略图
    if use_thumbnail and len(processed_images) > 0:
        thumbnail_img = image.resize((image_patches_size, image_patches_size))
        processed_images.append(thumbnail_img)

    return processed_images