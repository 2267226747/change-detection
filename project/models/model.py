import torch
import torch.nn as nn
from models.vision.backbone import VisionEncoder
from models.position_embedding_v2 import PositionalEmbedder2
from models.QueryToken_initialization import QueryGenerator
from models.transformer.late_fusion_block import FusionTransformerBlock2
from models.heads.multitask_head import MultitaskClassifier


class AssembledFusionModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # 保留原始 cfg 引用，以防子模块需要访问不同层级的配置
        self.full_cfg = cfg
        self.model_cfg = cfg.assembled_model

        # 1. Vision Encoder
        self.vision_encoder = VisionEncoder(cfg)

        # 2. Positional Embedder
        self.pos_embedder = PositionalEmbedder2(cfg)

        # 3. Query Generator
        self.query_generator = QueryGenerator(cfg)

        # 4. Transformer Blocks & 5. Class Heads
        # 获取总层数，默认为 12
        self.num_layers = getattr(cfg, 'num_layers', 24)
        self.start_classify = getattr(cfg, 'start_classify', 1)

        self.transformer_blocks = nn.ModuleList()
        self.class_heads = nn.ModuleList()

        # 构建交替层结构：Sensing -> Reasoning -> Sensing -> Reasoning ...
        for i in range(self.num_layers):
            # 逻辑定义: 
            # 偶数层 (0, 2, 4...) -> Sensing (if_query=False)
            # 奇数层 (1, 3, 5...) -> Reasoning (if_query=True)
            is_reasoning_layer = (i % 2 != 0)

            # 添加 Transformer Block
            block = FusionTransformerBlock2(cfg, if_query=is_reasoning_layer)
            self.transformer_blocks.append(block)

            # 如果是 Reasoning 层，则挂载一个分类头
            if is_reasoning_layer and (i+1) / 2 == self.start_classify:
                head = MultitaskClassifier(cfg)
                self.class_heads.append(head)
            else:
                # 为了保持索引对齐（或者单纯占位），这里不需要添加，
                # 但为了 forward 写法简单，我们在 ModuleList 里只存存在的 head
                pass

    def _process_visual_sequence(self, pixel_values, batch_size, patches_num):
        """
        处理序列化图片输入的辅助函数
        Args:
            pixel_values: [Batch*N, 3, H, W]
                          Batch: 批次大小
                          N: 每张样本被切分的图片数量
        Returns:
            features: [Batch, Total_Seq_Len, Dim]
        """

        # 视觉编码
        # output shape [B*N, S_local, Dim] (S_local是单张图的patch数)
        features = self.vision_encoder(pixel_values)

        # 展开维度 (Unfold) 并拼接
        # 每一张切片变成了一组 Tokens
        # [B*N, S_local, Dim] -> [B, N, S_local, Dim]
        _, s_local, dim_v = features.shape
        features = features.view(batch_size, patches_num, s_local, dim_v)

        # 拼接所有切片的 Tokens: [B, N, S_local, Dim] -> [B, N * S_local, Dim]
        features = features.flatten(1, 2)

        return features

    def forward(self, pixel_values_1, pixel_values_2):
        """
        Returns:
            all_results: 一个字典，包含所有 Reasoning 层的分类结果。
                         key: 层索引 (e.g., 'layer_1', 'layer_3')
                         value: 对应层的分类结果
        """

        # --- A. 基础特征提取 ---
        vision_1_feat = self._process_visual_sequence(pixel_values_1, self.full_cfg.data.batch_size,
                                                      self.full_cfg.data.patches_num)
        vision_2_feat = self._process_visual_sequence(pixel_values_2, self.full_cfg.data.batch_size,
                                                      self.full_cfg.data.patches_num)

        # --- B. 位置编码 ---
        pos_embed1 = self.pos_embedder(image_time=0)
        pos_embed2 = self.pos_embedder(image_time=1)

        vision_pos1 = pos_embed1
        vision_pos2 = pos_embed2

        # --- C. Query 初始化 ---
        # 已扩展到 Batch 维度(Batch, Num_Queries, Dim)
        q_t1, q_t2 = self.query_generator.get_queries()

        # --- D. 循环处理 (Sensing-Reasoning) ---
        all_results = {}
        head_idx = 0  # 用于追踪当前用到第几个 head

        for i, block in enumerate(self.transformer_blocks):
            # 检查当前层是否为 Reasoning 层 (奇数层: 1, 3, 5...)
            is_reasoning_layer = (i % 2 != 0)

            if is_reasoning_layer:

                # 执行 Transformer Block reasoning层，不需要query vision feature
                q_t1, q_t2 = block(
                    q_t1=q_t1,
                    q_t2=q_t2,
                )

                if (i+1) / 2 == self.start_classify:
                    # 获取对应的 Head
                    current_head = self.class_heads[head_idx]
                    # 计算分类结果
                    layer_results = current_head(q_t1, q_t2)
                    # 存入结果字典，Key 建议带上层号以便区分
                    all_results[f'Classifier{head_idx+1}_results'] = layer_results
                    # Head 索引递增
                    head_idx += 1
            else:

                # 执行 Transformer Block sensing 层，query vision feature
                q_t1, q_t2 = block(
                    q_t1=q_t1,
                    q_t2=q_t2,
                    vision_1=vision_1_feat,
                    vision_2=vision_2_feat,
                    vision_pos1=vision_pos1,
                    vision_pos2=vision_pos2
                )

        return all_results
