import torch
import torch.nn as nn
import numpy as np


class RLEnv:
    def __init__(self, pretrained_model, config, device, logger):
        """
        Args:
            pretrained_model: AssembledFusionModel 实例
            config: RL 配置
            device: 运行设备
            freeze_classifier: 是否冻结分类头
        """
        self.model = pretrained_model
        self.config = config
        self.device = device
        self.freeze_classifier = self.config.rl.freeze_classifier
        logger.info(f"Freeze classifier: {self.freeze_classifier}")


        # 1. 获取混合精度配置
        # 建议在 config 中添加该字段，默认 False
        self.use_amp = getattr(self.config.rl, 'use_amp', True)
        self.dtype = torch.float16 if self.use_amp else torch.float32
        # 确定设备类型供 autocast 使用 ('cuda' or 'cpu')
        self.device_type = 'cuda' if 'cuda' in str(self.device) else 'cpu'
        
        logger.info(f"RLEnv Mixed Precision (AMP) Enabled: {self.use_amp}")

        # ====================================================
        # 1. 冻结策略管理
        # ====================================================
        # A. 冻结骨干 (Encoder + PosEmbed + QueryGen + TransformerBlocks)
        # 排除 class_heads，其他全部冻结
        for name, param in self.model.named_parameters():
            if "class_heads" not in name:
                param.requires_grad = False

        # 确保骨干处于 Eval 模式 (关闭 Dropout, BatchNorm 统计不更新)
        self.model.eval()

        # B. 动态处理分类头 (class_heads 是 ModuleList)
        if self.freeze_classifier:
            for param in self.model.class_heads.parameters():
                param.requires_grad = False
            self.model.class_heads.eval()
        else:
            for param in self.model.class_heads.parameters():
                param.requires_grad = True
            # 微调模式：开启 train 以启用 Dropout (可选，视数据量而定)
            self.model.class_heads.train()

        # ====================================================
        # 2. 从第一个分类头解析任务结构
        # ====================================================
        # 假设所有层的分类头结构一致，我们读取第 0 个头的信息
        # 注意：如果 start_classify > 1，class_heads 依然是 0-indexed
        first_head = self.model.class_heads[0]

        self.group_names = first_head.group_names
        self.sub_counts = first_head.sub_counts  # list 各个group中子任务的数量 [8, 8,...]
        self.num_groups = len(self.group_names)
        self.total_subtasks = sum(self.sub_counts)
        logger.info(f"Task group list: {self.group_names}, "
                    f"Subtask counts: {self.sub_counts}, "
                    f"Total subtasks: {self.total_subtasks}")


        # 构建 Group -> Subtasks 索引映射 [[0,1,2...n] [n,n+1,...],...]
        self.group_indices = []
        current_idx = 0
        for count in self.sub_counts:
            self.group_indices.append(list(range(current_idx, current_idx + count)))
            current_idx += count

        # ====================================================
        # 3. 提取模型超参
        # ====================================================
        self.patches_num = self.model.full_cfg.data.patches_num
        self.start_classify = self.model.start_classify
        # 计算 RL 的最大步数
        # self.model.num_layers 是 Transformer 的总层数 (e.g. 12)
        # max_steps = 12 // 2 = 6 (对应 6 对 Sensing-Reasoning)
        self.max_steps = len(self.model.class_heads)

        # 运行时状态
        self.batch_size = 0
        self.current_step = 0  # 这里的 step 0 对应 start_classify 的那一层

        # 缓存容器
        self.q_t1 = None
        self.q_t2 = None
        self.vision_1_feat = None
        self.vision_2_feat = None
        self.vision_pos1 = None
        self.vision_pos2 = None

        self.subtask_active_mask = None
        self.final_logits = None
        self.current_labels = None

    def get_classifier_parameters(self):
        """返回所有分类头的参数供 Trainer 使用"""
        if self.freeze_classifier:
            return []
        return list(self.model.class_heads.parameters())

    def reset(self, batch_data):
        """
        初始化环境，并执行 RL 介入前的所有前置计算 (Pre-rollout)
        对应 Forward 的 A, B, C 阶段
        """
        pixel_values_1 = batch_data['pixel_values_t1'].to(self.device)
        pixel_values_2 = batch_data['pixel_values_t2'].to(self.device)
        self.current_labels = batch_data.get('labels', None)

        # 动态 Batch Size
        self.batch_size = pixel_values_1.shape[0] // self.patches_num

        # [NEW] 2. 开启 Autocast
        # 视觉骨干和 Transformer 的计算量最大，必须包裹
        with torch.amp.autocast(device_type=self.device_type, enabled=self.use_amp):
            with torch.no_grad():
                # --- A. 视觉特征提取 ---
                # 直接调用模型内部的辅助函数
                self.vision_1_feat = self.model._process_visual_sequence(
                    pixel_values_1, self.batch_size, self.patches_num
                )
                self.vision_2_feat = self.model._process_visual_sequence(
                    pixel_values_2, self.batch_size, self.patches_num
                )
                target_dtype = self.vision_1_feat.dtype

                # --- B. 位置编码 ---
                self.vision_pos1 = self.model.pos_embedder(self.batch_size, image_time=0).to(target_dtype)
                self.vision_pos2 = self.model.pos_embedder(self.batch_size, image_time=1).to(target_dtype)

                # --- C. Query 初始化 ---
                self.q_t1 = self.model.query_generator(self.batch_size).to(target_dtype)
                self.q_t2 = self.model.query_generator(self.batch_size).to(target_dtype)

                # --- Pre-rollout (关键) ---
                # 如果 start_classify > 1，RL 介入前的 transformer 层需要先跑完
                # start_classify=1 (default): range(0) -> 不跑，直接进入 step 0
                # start_classify=2: range(1) -> 跑完第1阶段(Block 0,1)，准备进入 step 0 (即第2阶段)
                for stage_i in range(self.start_classify - 1):
                    sensing_idx = stage_i * 2
                    reasoning_idx = stage_i * 2 + 1

                    # Sensing
                    layer_pos = self.model.query_generator.get_layer_pos(sensing_idx // 2, self.batch_size)
                    self.q_t1, self.q_t2 = self.model.transformer_blocks[sensing_idx](
                        q_t1=self.q_t1, q_t2=self.q_t2,
                        layer_pos=layer_pos,
                        vision_1=self.vision_1_feat, vision_2=self.vision_2_feat,
                        vision_pos1=self.vision_pos1, vision_pos2=self.vision_pos2
                    )
                    # Reasoning
                    self.q_t1, self.q_t2 = self.model.transformer_blocks[reasoning_idx](
                        q_t1=self.q_t1, q_t2=self.q_t2
                    )

        # 重置计数器
        self.current_step = 0

        # 初始化 Masks [B, task_nums]
        self.subtask_active_mask = torch.ones(
            (self.batch_size, self.total_subtasks),
            dtype=torch.bool, device=self.device
        )
        self.final_logits = torch.zeros(
            (self.batch_size, self.total_subtasks),
            dtype=torch.float32, device=self.device
        )

        return self._get_observation()

    def step(self, action):
        """
        对应 Forward 的 D 阶段 (Loop Body)
        action: {'correction': [B, G, D], 'stop': [B, Total]}
        G = num_groups
        每个 group 对应一个 correction 向量
        不是 token 级

        RL Step: Update Mask -> Correct Query -> Transformer -> Classify -> Update Logits
        """
        group_correction = action['correction']
        stop_decision = action['stop']

        # ====================================================
        # 1. 立即更新 Mask (Action First)
        # ====================================================
        # [关键步骤 1] 备份旧 Mask (Pre-Stop Mask)
        # clone() 很重要，防止后续原地修改影响这里
        pre_subtask_active_mask = self.subtask_active_mask.clone()

        # 如果 Agent 决定停止，对应的 Mask 立即变为 False
        # 这意味着：本层的 Query 修正无效，本层的 Transformer 结果无效
        should_stop = (stop_decision == 1)
        self.subtask_active_mask = self.subtask_active_mask & (~should_stop)

        # ---------------------------------------------------
        # [优化] 极速短路 (Fast Path)
        # 如果所有子任务都不活跃了 (都被 Stop 或之前就已经 Done)
        # 就不需要跑昂贵的 Transformer 和 Classifier 了
        # ---------------------------------------------------
        if (~self.subtask_active_mask).all():
            self.current_step += 1

            # 计算 dones (全是 True，除非 max_steps 还没到且还有任务？不，Mask全False说明都停了)
            # 注意：这里 dones 的逻辑要和下面保持一致
            all_stopped = True  # Mask 全 False 意味着全 Stopped
            max_reached = (self.current_step >= self.max_steps)
            dones = torch.ones(self.batch_size, dtype=torch.bool, device=self.device)  # All True

            info = {
                'step': self.current_step,
                'active_mask': self.subtask_active_mask,
                'pre_action_mask': pre_subtask_active_mask,  # 依然需要传出去算 Reward
                'labels': self.current_labels,
                # 当 skip 时，input_q 保持上一刻的值，或者全0，RewardCalculator 不会用到它
                'cls_input_q': torch.cat([self.q_t1, self.q_t2], dim=-1)
            }

            # 占位 Reward
            rewards = torch.zeros(self.batch_size, device=self.device)

            # 直接返回下一帧 obs (通常这时候 obs 已经是 padding 或者是最终状态的 snapshot)
            return self._get_observation(), rewards, dones, info

        # ====================================================
        # 如果还有任务存活，才继续跑下面的重型计算
        # ====================================================
        # 计算 Group Active Mask (OR 逻辑)
        # group_active_mask: [B, G]
        group_active_mask = torch.zeros((self.batch_size, self.num_groups), dtype=torch.bool, device=self.device)
        for g_idx in range(self.num_groups):
            s_indices = self.group_indices[g_idx]  # len = sub_counts[g]
            group_active_mask[:, g_idx] = self.subtask_active_mask[:, s_indices].any(dim=1)  # [B, sub_counts[g]] → [B]

        tokens_per_group = self.q_t1.shape[1] // self.num_groups  # q_t1: [B, N, D],  N = G * tokens_per_group

        # group_active_mask: [B, G]  扩展 Mask: [B, G] -> [B, N]
        token_active_mask = group_active_mask.repeat_interleave(tokens_per_group, dim=1).unsqueeze(-1)

        # ====================================================
        # 2. 应用 Query 修正 (仅 Active 的 Group 生效) (Broadcast & Mask)
        # ====================================================
        # 广播group_correction: [B, G, D] -> [B, N, D]
        correction_expanded = group_correction.unsqueeze(2).repeat(1, 1, tokens_per_group,
                                                                   1)  # [B, G, 1, D] → [B, G, tokens_per_group, D]
        # → [B, N, D]
        correction_flat = correction_expanded.reshape(self.batch_size, -1, group_correction.shape[-1])

        # 修正 Query
        self.q_t1 = self.q_t1 + (correction_flat * token_active_mask.float())
        self.q_t2 = self.q_t2 + (correction_flat * token_active_mask.float())

        # ====================================================
        # 3. 执行 Transformer Block (RL 仅决定输入，不干涉内部更新)
        # ====================================================
        # 计算当前应当执行的 Block 索引
        # 因为 reset 已经跑过了前置层，这里的 current_step 0 对应 start_classify 那一层
        stage_idx = (self.start_classify - 1) + self.current_step
        sensing_idx = stage_idx * 2
        reasoning_idx = stage_idx * 2 + 1

        # 获取具体的 Block 实例
        block_sensing = self.model.transformer_blocks[sensing_idx]
        block_reasoning = self.model.transformer_blocks[reasoning_idx]
        current_head = self.model.class_heads[self.current_step]

        # 开启 Autocast 进行重计算
        with torch.amp.autocast(device_type=self.device_type, enabled=self.use_amp):
            with torch.no_grad():
                # Sensing Layer
                layer_pos = self.model.query_generator.get_layer_pos(sensing_idx // 2, self.batch_size)
                q_t1_next, q_t2_next = block_sensing(
                    q_t1=self.q_t1, q_t2=self.q_t2,
                    layer_pos=layer_pos,
                    vision_1=self.vision_1_feat, vision_2=self.vision_2_feat,
                    vision_pos1=self.vision_pos1, vision_pos2=self.vision_pos2
                )
                # Reasoning Layer
                q_t1_next, q_t2_next = block_reasoning(
                    q_t1=q_t1_next, q_t2=q_t2_next
                )

                # 模拟 Early Exit：只更新 active 的部分，token级别，但是在一个group内是一致的
                # 4. 执行分类头 (MultitaskClassifier)
                # Forward
                logits_dict = current_head(q_t1_next, q_t2_next)

        # Flatten
        current_logits_flat = self._flatten_logits(logits_dict).float()

        # 更新结果 (Result Locking)
        self.final_logits = torch.where(
            self.subtask_active_mask,
            current_logits_flat,
            self.final_logits
        )

        # 更新query token
        self.q_t1 = q_t1_next
        self.q_t2 = q_t2_next

        self.current_step += 1
        # 检查是否所有任务都停止
        all_stopped = (~self.subtask_active_mask).all(dim=1)
        max_reached = (self.current_step >= self.max_steps)
        dones = all_stopped | max_reached

        info = {
            'step': self.current_step - 1,
            'active_mask': self.subtask_active_mask,
            'pre_action_mask': pre_subtask_active_mask,  # 这是给 Reward 计算用的旧 mask
            'labels': self.current_labels,
            # 显式传出分类头使用的输入
            'cls_input_q': torch.cat([self.q_t1, self.q_t2], dim=-1)
        }

        # 返回占位符 rewards 以符合标准接口
        rewards = torch.zeros(self.batch_size, device=self.device)

        return self._get_observation(), rewards, dones, info

    def _get_observation(self):
        """构造 RL 观测状态"""
        # 1. Query State: [B, N, 2D]
        query_state = torch.cat([self.q_t1, self.q_t2], dim=-1)

        # 2. Vision Summary: [B, D]
        vis_summary = (self.vision_1_feat.mean(dim=1) + self.vision_2_feat.mean(dim=1)) / 2

        # 3. Entropy: [B, Total_Subtasks]
        # 基于当前的 final_logits 计算
        # 如果 logits 全为 0 (前几层)，entropy 也是均匀分布的高熵
        # Binary Cross Entropy 场景 (Sigmoid)
        # reset 时 final_logits 已初始化为 0，Sigmoid(0)=0.5，逻辑是自洽的
        probs = torch.sigmoid(self.final_logits)
        # H(p) = -p log p - (1-p) log (1-p)
        entropy = -(probs * torch.log(probs + 1e-6) + (1 - probs) * torch.log(1 - probs + 1e-6))
        # 如果是多分类，逻辑需调整为 softmax

        # 4. Time: [B, 1]
        time_enc = torch.full((self.batch_size, 1), self.current_step / self.max_steps, device=self.device)

        return {
            'query_state': query_state,  # [B, N, 2D]
            'vision_context': vis_summary,  # [B, D]
            'entropy': entropy,  # [B, Total_tasks] 明确的不确定性信号
            'probs': probs,  # [B, Total_tasks] [新增] 明确的方向性信号
            'time': time_enc,  # [B, 1]
            'active_mask': self.subtask_active_mask  # [B, Total_Subtasks]
        }

    def _flatten_logits(self, logits_dict):
        # 必须确保顺序与 group_names 一致
        flat_list = []
        for g_name in self.group_names:
            sub_logits = logits_dict.get(g_name, [])
            for sl in sub_logits:
                flat_list.append(sl)
        if not flat_list:
            return torch.zeros((self.batch_size, self.total_subtasks), device=self.device)
        return torch.cat(flat_list, dim=1)
