# project/models/vision/backbone.py
import os
import torch
import torch.nn as nn
from transformers import AutoModel, CLIPImageProcessor
from utils.config import Config

os.environ["FLASH_ATTENTION"] = "0"
os.environ["USE_FLASH_ATTENTION"] = "0"
os.environ["FLASH_ATTENTION_VERSION"] = ""


class VisionEncoder(nn.Module):
    def __init__(self, cfg):
        """
        Args:
            cfg: 全局配置对象 (cfg.model.vision)
        """
        super(VisionEncoder, self).__init__()

        # 从 cfg 中读取参数
        model_name = cfg.model.vision.backbone
        self.freeze = cfg.model.vision.freeze_backbone
        # 读取层索引参数，默认为 -1 (即最后一层)
        self.select_layer = getattr(cfg.model.vision, 'select_layer', -1)
        print(f"Loading Vision Backbone: {model_name} ...")
        print(f"Selected Layer Index: {self.select_layer}")

        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
            trust_remote_code=True,
            output_hidden_states=True  # 获取中间层
        )

        if self.freeze:
            print("Frozen Vision Backbone.")
            for param in self.model.parameters():
                param.requires_grad = False

        # 从 cfg 读取维度，或从模型配置读取
        self.output_dim = self.model.config.hidden_size

        # 校验 cfg 中的维度是否与模型一致 (防止写错配置文件)
        if cfg.model.vision.feature_dim != self.output_dim:
            print(f"[Warning] Config feature_dim ({cfg.model.vision.feature_dim}) "
                  f"does not match Model hidden_size ({self.output_dim}). Updating config...")
            # 注意：这里修改的是局部变量，不会回写到 yaml

        # --- [新增代码] Pixel Unshuffle & MLP 配置 ---
        # 1. 获取目标维度 (LLM 的 hidden size，例如 4096), 下采样比率 (默认 0.5)
        # 在 cfg.model.hidden_dim 中定义，如果没有则默认 4096
        self.target_dim = getattr(cfg.model, 'hidden_dim', 4096)
        self.downsample_ratio = getattr(cfg.model.vision, 'downsample_ratio', 0.5)

        # 2. 计算 Scale 和 输入通道数
        # 情况 A (ratio=0.5): scale = 2, in_channels = output_dim * 4
        # 情况 B (ratio=1.0): scale = 1, in_channels = output_dim * 1 (即原维度)
        self.pixel_shuffle_scale = int(1 / self.downsample_ratio)
        in_channels = self.output_dim * (self.pixel_shuffle_scale ** 2)

        # 3. 定义 MLP (结构完全一致)
        self.mlp = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, self.target_dim),
            nn.GELU(),
            nn.Linear(self.target_dim, self.target_dim)
        )

    # --- [新增方法] Pixel Unshuffle 实现 ---
    def pixel_unshuffle(self, x):
        """
        x: [Batch, Seq_Len, Dim] -> [Batch, Seq_Len/4, Dim*4]
        """
        b, s, c = x.shape
        # 假设 Patch 是正方形排列 (对于 448x448, Patch Size 14, 这里的 s 应为 1024, h=w=32)
        h = w = int(s ** 0.5)

        # 1. [B, S, C] -> [B, H, W, C] -> [B, C, H, W]
        x = x.view(b, h, w, c).permute(0, 3, 1, 2)

        # 2. PyTorch 原生 PixelUnshuffle (Space-to-Depth)
        x = nn.functional.pixel_unshuffle(x, downscale_factor=self.pixel_shuffle_scale)

        # 3. [B, C*r^2, H/r, W/r] -> [B, H/r * W/r, C*r^2] (展平回序列)
        x = x.permute(0, 2, 3, 1).flatten(1, 2)

        return x

    def forward(self, pixel_values):
        if pixel_values.dtype != self.model.dtype:
            pixel_values = pixel_values.to(self.model.dtype)

        outputs = self.model(pixel_values)
        # 从 hidden_states 中提取指定层
        # outputs.hidden_states 是一个 tuple，包含 (embeddings, layer_1, ..., layer_N)
        # Python 支持负数索引，-1 是最后一层，-4 就是倒数第四层
        vit_embeds = outputs.hidden_states[self.select_layer]

        # 1. 去除 CLS Token
        vit_embeds = vit_embeds[:, 1:, :]
        # 2. Pixel Unshuffle
        if self.downsample_ratio != 1.0:
            vit_embeds = self.pixel_unshuffle(vit_embeds)

        # 3. MLP Projector (包含 LayerNorm)
        vit_embeds = self.mlp(vit_embeds)

        return vit_embeds

    @staticmethod
    def get_image_processor(model_name):
        return CLIPImageProcessor.from_pretrained(model_name, trust_remote_code=True)


# ================= 测试代码 =================
if __name__ == "__main__":
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))  # 把 project 根目录加到路径
    from utils.config import Config

    # 1. 加载配置
    cfg_path = "../../configs/defaults.yaml"
    if os.path.exists(cfg_path):
        cfg = Config.from_yaml(cfg_path)

        # 2. 初始化模型 (传入 cfg)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        encoder = VisionEncoder(cfg).to(device)

        # 3. 测试
        dummy_input = torch.randn(2, 3, 448, 448).to(device)
        with torch.no_grad():
            out = encoder(dummy_input)
        print(f"Output Shape: {out.shape}")
    else:
        print("configs/defaults.yaml not found")
