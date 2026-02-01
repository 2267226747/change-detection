# 准确率、F1 等指标计算
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, average_precision_score
import os


class Evaluator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device('cpu')  # 评估计算主要在 CPU 上进行

        # 解析任务配置，用于切分 Targets
        # 结构必须与 Loss 中的解析逻辑一致
        self.task_configs = cfg.loss.tasks
        self.task_slices = {}

        current_idx = 0
        for task_name, task_cfg in self.task_configs.items():
            num_classes = task_cfg.num_classes
            self.task_slices[task_name] = (current_idx, current_idx + num_classes)
            current_idx += num_classes

        self.reset()

    def reset(self):
        """每个 Epoch 开始时重置存储器"""
        # predictions 结构: {layer_name: {task_name: []}}
        self.predictions = {}
        # targets 结构: {task_name: []} (所有层共享标签)
        self.targets = {k: [] for k in self.task_configs.keys()}
        self.oids = []

    @torch.no_grad()
    def update(self, model_outputs, targets, oids):
        """
        每个 Batch 调用一次，积累数据
        Args:
            model_outputs: Dict {'layer_11_results': {'road': logits, ...}}

            targets: [Batch, Total_Cols]
            oids: List[str] 或 Tuple[str], 来自 batch['oid']
        """

        # 1. 记录 OID (如果是 tuple/list 直接 extend)
        self.oids.extend(oids)

        # 2. 获取最后一层的输出 (假设 key 是 layer_X_results)
        # 也可以指定评估哪一层，这里默认取 key 最大的层（最深层）
        layer_keys = sorted(model_outputs.keys(), key=lambda x: int(x.split('_')[1]))

        for layer_key in layer_keys:
            if layer_key not in self.predictions:
                self.predictions[layer_key] = {k: [] for k in self.task_configs.keys()}

            layer_output = model_outputs[layer_key]

            for task_name, logits in layer_output.items():
                if task_name not in self.task_slices:
                    continue

                # 1. detach: 断开梯度
                # 2. float: 强制转 FP32 (防止 BF16 在 CPU Numpy 出错)
                # 3. sigmoid: 计算概率
                # 4. cpu: 移至内存
                # 5. numpy: 转为 numpy 数组
                if isinstance(logits, list):
                    # 将列表中的多个 Tensor 沿着维度 1 (列) 拼接起来
                    # 结果形状会变成 (Batch_Size, 8)
                    logits = torch.cat(logits, dim=1)
                # --------------------
                probs = torch.sigmoid(logits.detach().float()).cpu().numpy()
                self.predictions[layer_key][task_name].append(probs)

        # 2. 存储 Targets (只需存储一次，因为所有层标签相同)
        # 我们只在第一次 update 或者每一批次都存，但需要注意 compute 时拼接逻辑
        for task_name in self.task_configs.keys():
            start, end = self.task_slices[task_name]
            task_target = targets[:, start:end]
            # 同样转为 FP32 numpy
            task_target_np = task_target.detach().float().cpu().numpy()
            self.targets[task_name].append(task_target_np)

    def compute(self, epoch=None):
        """
        计算所有层及平均层的指标
        Returns:
            metrics_results: 嵌套字典
            df_report: 包含 Layer 列的 DataFrame
        """
        report_rows = []
        metrics_results = {}

        # 获取所有层名
        layer_names = sorted(self.predictions.keys(), key=lambda x: int(x.split('_')[1]))

        # 1. 准备 Ground Truth (拼接)
        # 注意：这里 targets 列表长度等于总 batch 数
        y_true_dict = {
            t_name: np.concatenate(self.targets[t_name], axis=0)
            for t_name in self.targets.keys()
        }

        # 2. 逐层计算指标
        for layer in layer_names:
            metrics_results[layer] = {}

            for task_name in self.task_configs.keys():
                # 该层该任务的预测
                preds_list = self.predictions[layer].get(task_name, [])
                if not preds_list: continue

                y_pred_prob = np.concatenate(preds_list, axis=0).astype(np.float32)
                y_true = y_true_dict[task_name]

                y_pred_cls = (y_pred_prob > 0.5).astype(int)
                num_subclasses = y_true.shape[1]

                for i in range(num_subclasses):
                    sub_y_true = y_true[:, i]
                    sub_y_pred = y_pred_cls[:, i]
                    sub_y_prob = y_pred_prob[:, i]

                    # [健壮性检查] 防止全 0 或全 1 导致 sklearn 报警
                    # 虽然 zero_division=0 已经处理了 F1，但 AP/AUC 仍可能报错
                    try:
                        acc = accuracy_score(sub_y_true, sub_y_pred)
                        p, r, f1, _ = precision_recall_fscore_support(
                            sub_y_true, sub_y_pred, average='binary', zero_division=0
                        )
                        # AP 即使在单类别数据上通常也能算，但也可能抛异常
                        ap = average_precision_score(sub_y_true, sub_y_prob)
                        if np.isnan(ap): ap = 0.0  # 处理 NaN

                        # # 增加 AUC (可选，如果只有一类正样本或负样本会报错)
                        # try:
                        #     auc = roc_auc_score(sub_y_true, sub_y_prob)
                        # except:
                        #     auc = 0.5

                    except Exception as e:
                        # print(f"Metric calc failed for {layer}/{task_name}/{i}: {e}")
                        acc, p, r, f1, ap = 0.0, 0.0, 0.0, 0.0, 0.0

                    metrics_results[layer][f"{task_name}_{i}"] = {
                        'Accuracy': acc, 'Precision': p, 'Recall': r, 'F1': f1, 'AP': ap
                    }

                    report_rows.append({
                        'Layer': layer,
                        'Group': task_name,
                        'SubID': i,
                        'Accuracy': round(acc, 4),
                        'Precision': round(p, 4),
                        'Recall': round(r, 4),
                        'F1': round(f1, 4),
                        'AP': round(ap, 4)
                    })

        # 生成基础 DataFrame
        df = pd.DataFrame(report_rows)

        if df.empty:
            return {}, df

        # 3. 计算每一层的 Group Mean (Road MEAN, Building MEAN)
        # grouping by Layer + Group
        layer_means = df.groupby(['Layer', 'Group']).mean(numeric_only=True).reset_index()
        layer_means['SubID'] = 'MEAN'
        df = pd.concat([df, layer_means], ignore_index=True)

        # 4. 计算所有层的平均表现 (AVERAGE_LAYER)
        # grouping by Group + SubID (across all layers)
        avg_layer_df = df.groupby(['Group', 'SubID']).mean(numeric_only=True).reset_index()
        avg_layer_df['Layer'] = 'AVG_ALL'

        # 合并所有结果
        df_final = pd.concat([df, avg_layer_df], ignore_index=True)

        # 排序: Layer (numeric sort logic needed but string sort ok for few layers) -> Group -> SubID
        # 先按层数(数字)排，再按组排
        # 提取层号辅助排序
        # def get_layer_num(s):
        #     if 'layer_' in s: return int(s.split('_')[1])
        #     if s == 'AVG_ALL': return 9999
        #     return -1
        #
        # df_final['LayerNum'] = df_final['Layer'].apply(get_layer_num)
        # df_final = df_final.sort_values(by=['LayerNum', 'Group', 'SubID'])
        # df_final = df_final.drop(columns=['LayerNum'])  # 删掉辅助列

        # 插入 Epoch 列
        if epoch is not None:
            # insert(位置, 列名, 值)
            df_final.insert(0, 'Epoch', epoch)

        return metrics_results, df_final

    def save_predictions(self, save_path):
        """
        保存预测结果 CSV
        列结构:
        oid | road_0_true | road_1_true | ... | L1_road_0_prob | L1_road_0_pred | ... | L3_road_0_prob ...
        """
        if not self.oids:
            print("No predictions to save.")
            return

        # 1. 基础信息：OID
        df_out = pd.DataFrame({'oid': self.oids})

        layer_names = sorted(self.predictions.keys(), key=lambda x: int(x.split('_')[1]))

        # 2. 添加 Ground Truth (只添加一次)
        y_true_dict = {
            t_name: np.concatenate(self.targets[t_name], axis=0)
            for t_name in self.targets.keys()
        }

        for task_name, targets in y_true_dict.items():
            num_sub = targets.shape[1]
            for i in range(num_sub):
                df_out[f"{task_name}_{i}_true"] = targets[:, i].astype(int)

        # 3. 添加每一层的预测
        for layer in layer_names:
            # 简写层名以缩短列名 (layer_11_results -> L11)
            layer_short = f"L{layer.split('_')[1]}"

            for task_name in self.task_configs.keys():
                preds_list = self.predictions[layer].get(task_name, [])
                if not preds_list: continue

                probs = np.concatenate(preds_list, axis=0)
                preds = (probs > 0.5).astype(int)
                num_sub = probs.shape[1]

                for i in range(num_sub):
                    # 列名格式: L11_road_0_prob
                    col_base = f"{layer_short}_{task_name}_{i}"
                    df_out[f"{col_base}_prob"] = probs[:, i].round(4)
                    df_out[f"{col_base}_pred"] = preds[:, i]

        # 保存
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df_out.to_csv(save_path, index=False)
        print(f"Predictions saved to: {save_path}")
