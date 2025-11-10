# AcceptHead 评估功能实现 - 修改总结

本文档总结了为 AcceptHead 回归任务添加支持容差的 accuracy 计算功能对 LLaMA-Factory 代码库的所有修改。

---

## 修改概述

本次修改实现了以下功能：

1. **支持容差的 Accuracy 计算**：为 AcceptHead 回归任务添加了支持容差阈值的 accuracy 计算
2. **额外的回归指标**：自动计算 MAE、MSE、RMSE 等回归指标
3. **自动识别 AcceptHead**：当使用 AcceptHead 时，自动使用专用的 accuracy 计算方法

---

## 修改目的

### 问题背景

- AcceptHead 是一个回归任务，预测的是 0-1 之间的分数，而不是分类任务的 token ID
- 默认的 `ComputeAccuracy` 方法是为分类任务设计的，使用精确匹配（`pred == label`），不适合回归任务
- 回归任务中，要求预测值和真实值完全相等过于苛刻，需要使用容差（tolerance）来判断预测是否正确

### 解决方案

- 创建了 `ComputeAcceptHeadAccuracy` 类，支持容差阈值的 accuracy 计算
- 对于较大的标签值（≥ 0.01）：使用相对误差 `|pred - label| / label`，如果 ≤ tolerance，则视为正确
- 对于很小的标签值（< 0.01）：使用绝对误差 `|pred - label|`，如果 ≤ tolerance，则视为正确
- 同时计算 MAE、MSE、RMSE 等回归指标，提供更全面的评估

---

## 一、修改文件（3个）

### 1. `src/llamafactory/train/sft/metric.py`

**修改位置**：
- 第61-84行：添加 `accept_head_logit_processor` 函数
- 第101-160行：添加 `ComputeAcceptHeadAccuracy` 类

#### 1.1 添加 `accept_head_logit_processor` 函数

**目的**：将 AcceptHead 输出的 logits 转换为概率值（0-1）

**功能**：
- 处理 logits 的 tuple/list 格式（兼容 MoE 模型）
- 使用 sigmoid 将 logits 转换为 0-1 之间的概率值

**代码**：
```python
def accept_head_logit_processor(logits: "torch.Tensor", labels: "torch.Tensor") -> "torch.Tensor":
    r"""Convert AcceptHead logits to probabilities using sigmoid.
    
    AcceptHead outputs logits (unbounded), which need to be converted to
    probabilities (0-1) using sigmoid for accuracy computation.
    
    Args:
        logits: Usually a 2D tensor [batch_size, seq_len] from AcceptHead.
                In some cases (e.g., MoE models), it might be a tuple/list where
                the first element is the main logits.
        labels: Labels tensor (not used here, but required by the interface)
    
    Returns:
        Probabilities tensor [batch_size, seq_len] with values in [0, 1]
    """
    # Handle tuple/list format (e.g., from MoE models)
    if isinstance(logits, (list, tuple)):
        # For AcceptHead, we expect 2D logits [batch_size, seq_len]
        # Take the first element which should be the main logits
        logits = logits[0]
    
    # AcceptHead outputs [batch_size, seq_len] logits (2D tensor)
    # Convert to probabilities using sigmoid
    return torch.sigmoid(logits)
```

**关键点**：
- AcceptHead 输出的是未经过 sigmoid 的 logits（2D tensor: `[batch_size, seq_len]`）
- 需要转换为 0-1 之间的概率值才能与标签（0-1 分数）进行比较
- 兼容 MoE 模型可能返回 tuple/list 的情况

#### 1.2 添加 `ComputeAcceptHeadAccuracy` 类

**目的**：计算支持容差的 accuracy 和额外的回归指标

**功能**：
- 计算带容差的 accuracy（相对误差或绝对误差）
- 计算 MAE（Mean Absolute Error）
- 计算 MSE（Mean Squared Error）
- 计算 RMSE（Root Mean Squared Error）

**代码**：
```python
@dataclass
class ComputeAcceptHeadAccuracy:
    r"""Compute accuracy for AcceptHead regression task with tolerance support.
    
    For regression tasks, predictions are considered correct if they are within
    a tolerance threshold (e.g., 10%) of the true label.
    """

    tolerance: float = 0.1  # Default 10% tolerance

    def _dump(self) -> Optional[dict[str, float]]:
        result = None
        if hasattr(self, "score_dict"):
            result = {k: float(np.mean(v)) for k, v in self.score_dict.items()}

        self.score_dict = {"accuracy": [], "mae": [], "mse": [], "rmse": []}
        return result

    def __post_init__(self):
        self._dump()

    def __call__(self, eval_preds: "EvalPrediction", compute_result: bool = True) -> Optional[dict[str, float]]:
        # eval_preds.predictions: logits from AcceptHead (already sigmoided in preprocess_logits_for_metrics)
        # eval_preds.label_ids: target scores (0-1) or IGNORE_INDEX
        preds, labels = numpify(eval_preds.predictions), numpify(eval_preds.label_ids)
        
        for i in range(len(preds)):
            pred, label = preds[i], labels[i]
            # Only compute metrics on mismatch positions (where label != IGNORE_INDEX)
            label_mask = label != IGNORE_INDEX
            
            if label_mask.sum() > 0:
                pred_valid = pred[label_mask]  # Predicted scores (0-1)
                label_valid = label[label_mask]  # True scores (0-1)
                
                # Compute accuracy with tolerance
                epsilon = 1e-8
                relative_error = np.abs(pred_valid - label_valid) / np.maximum(label_valid, epsilon)
                # For very small labels (< 0.01), use absolute error threshold of tolerance
                small_label_mask = label_valid < 0.01
                absolute_error = np.abs(pred_valid - label_valid)
                correct = np.where(
                    small_label_mask,
                    absolute_error <= self.tolerance,  # Absolute threshold for small labels
                    relative_error <= self.tolerance   # Relative threshold for larger labels
                )
                
                self.score_dict["accuracy"].append(np.mean(correct))
                
                # Compute additional regression metrics
                mae = np.mean(absolute_error)  # Mean Absolute Error
                mse = np.mean((pred_valid - label_valid) ** 2)  # Mean Squared Error
                rmse = np.sqrt(mse)  # Root Mean Squared Error
                
                self.score_dict["mae"].append(mae)
                self.score_dict["mse"].append(mse)
                self.score_dict["rmse"].append(rmse)

        if compute_result:
            return self._dump()
```

**关键特性**：
- **容差计算逻辑**：
  - 对于标签值 ≥ 0.01：使用相对误差 `|pred - label| / label`
  - 对于标签值 < 0.01：使用绝对误差 `|pred - label|`
  - 如果误差 ≤ tolerance，则视为正确
- **只计算 mismatch 位置**：只对 `label != IGNORE_INDEX` 的位置计算指标
- **额外指标**：同时计算 MAE、MSE、RMSE，提供更全面的评估

---

### 2. `src/llamafactory/hparams/finetuning_args.py`

**修改位置**：第509-512行

**修改内容**：添加 `accept_head_accuracy_tolerance` 配置参数

**代码**：
```python
accept_head_accuracy_tolerance: float = field(
    default=0.1,
    metadata={"help": "Tolerance threshold for AcceptHead accuracy computation (e.g., 0.1 means 10% relative error)."},
)
```

**目的**：允许在 YAML 配置文件中通过 `accept_head_accuracy_tolerance` 来设置容差阈值

**说明**：
- 默认值为 0.1（表示 10% 的相对误差）
- 可以根据任务需求调整，例如：
  - `0.05` = 5% 容差（更严格）
  - `0.15` = 15% 容差（更宽松）

---

### 3. `src/llamafactory/train/sft/workflow.py`

**修改位置**：
- 第27行：更新导入语句
- 第72-79行：修改 metric 选择逻辑

#### 3.1 更新导入语句

**修改前**：
```python
from .metric import ComputeAccuracy, ComputeSimilarity, eval_logit_processor
```

**修改后**：
```python
from .metric import ComputeAccuracy, ComputeAcceptHeadAccuracy, ComputeSimilarity, accept_head_logit_processor, eval_logit_processor
```

#### 3.2 修改 metric 选择逻辑

**修改前**：
```python
elif finetuning_args.compute_accuracy:
    metric_module["compute_metrics"] = ComputeAccuracy()
    metric_module["preprocess_logits_for_metrics"] = eval_logit_processor
```

**修改后**：
```python
elif finetuning_args.compute_accuracy:
    # Use AcceptHead accuracy if replace_lm_head is enabled
    if model_args.replace_lm_head:
        metric_module["compute_metrics"] = ComputeAcceptHeadAccuracy(tolerance=finetuning_args.accept_head_accuracy_tolerance)
        metric_module["preprocess_logits_for_metrics"] = accept_head_logit_processor
    else:
        metric_module["compute_metrics"] = ComputeAccuracy()
        metric_module["preprocess_logits_for_metrics"] = eval_logit_processor
```

**目的**：
- 自动检测是否使用 AcceptHead（通过 `model_args.replace_lm_head`）
- 如果使用 AcceptHead，自动使用 `ComputeAcceptHeadAccuracy` 和 `accept_head_logit_processor`
- 否则使用默认的 `ComputeAccuracy` 和 `eval_logit_processor`（保持向后兼容）

---

## 二、配置文件更新

### `examples/train_lora/llama3_1_8b_lora_sft_accept_head.yaml`

**修改位置**：第44-51行（eval 部分）

**修改内容**：添加评估相关配置

**修改前**：
```yaml
### eval
# eval_dataset: alpaca_en_demo
# val_size: 0.1
# per_device_eval_batch_size: 1
# eval_strategy: steps
# eval_steps: 500
```

**修改后**：
```yaml
### eval
# eval_dataset: alpaca_en_demo
val_size: 0.05  # 分出5%的数据作为验证集
per_device_eval_batch_size: 1
eval_strategy: steps  # 按步数进行评估
eval_steps: 250  # 每250步进行一次评估（可以根据需要调整）
compute_accuracy: true  # 启用accuracy计算
accept_head_accuracy_tolerance: 0.1  # Accuracy容差阈值（0.1表示10%的相对误差）
```

**新增配置说明**：
- `val_size: 0.05`：分出 5% 的数据作为验证集
- `eval_strategy: steps`：按步数进行评估
- `eval_steps: 250`：每 250 步进行一次评估
- `compute_accuracy: true`：启用 accuracy 计算
- `accept_head_accuracy_tolerance: 0.1`：设置容差阈值为 10%

---

## 三、使用方法

### 1. 在 YAML 配置文件中启用

在训练配置文件中添加以下配置：

```yaml
### eval
val_size: 0.05  # 验证集比例
eval_strategy: steps  # 评估策略
eval_steps: 250  # 评估步数间隔
compute_accuracy: true  # 启用accuracy计算
accept_head_accuracy_tolerance: 0.1  # 容差阈值（0.1 = 10%）
```

### 2. 评估指标说明

训练时，每 `eval_steps` 步会在验证集上自动计算并记录以下指标：

- **`eval_accuracy`**：准确率（基于容差）
  - 对于标签值 ≥ 0.01：使用相对误差判断
  - 对于标签值 < 0.01：使用绝对误差判断
  - 如果误差 ≤ tolerance，则视为正确

- **`eval_mae`**：平均绝对误差（Mean Absolute Error）
  - 公式：`MAE = mean(|pred - label|)`
  - 越小越好，表示预测的平均偏差

- **`eval_mse`**：均方误差（Mean Squared Error）
  - 公式：`MSE = mean((pred - label)^2)`
  - 越小越好，对大误差更敏感

- **`eval_rmse`**：均方根误差（Root Mean Squared Error）
  - 公式：`RMSE = sqrt(MSE)`
  - 越小越好，与原始数据单位一致

### 3. 容差阈值调整

可以根据任务需求调整 `accept_head_accuracy_tolerance`：

- **更严格**：`0.05`（5% 容差）
- **默认**：`0.1`（10% 容差）
- **更宽松**：`0.15`（15% 容差）或 `0.2`（20% 容差）

---

## 四、工作原理

### 1. 评估流程

1. **模型前向传播**：AcceptHead 输出 logits `[batch_size, seq_len]`
2. **Logits 预处理**：`accept_head_logit_processor` 将 logits 通过 sigmoid 转换为概率 `[batch_size, seq_len]`（值域 0-1）
3. **指标计算**：`ComputeAcceptHeadAccuracy` 计算 accuracy 和其他回归指标
4. **结果记录**：指标被记录到日志和训练历史中

### 2. Accuracy 计算逻辑

对于每个样本的每个位置（只计算 mismatch 位置，即 `label != IGNORE_INDEX`）：

```python
# 计算绝对误差
absolute_error = |pred - label|

# 对于较大的标签值（≥ 0.01）
if label >= 0.01:
    relative_error = absolute_error / label
    correct = (relative_error <= tolerance)
# 对于很小的标签值（< 0.01）
else:
    correct = (absolute_error <= tolerance)

# 计算 accuracy
accuracy = mean(correct)
```

### 3. 为什么需要区分相对误差和绝对误差？

- **相对误差**：适用于较大的标签值，例如 `label = 0.5`，如果 `pred = 0.55`，相对误差为 `0.1`（10%），这是合理的
- **绝对误差**：适用于很小的标签值，例如 `label = 0.001`，如果使用相对误差，`pred = 0.002` 的相对误差为 `100%`，这会导致所有小值都被判为错误

---

## 五、向后兼容性

- **自动检测**：代码会自动检测是否使用 AcceptHead（通过 `model_args.replace_lm_head`）
- **默认行为**：如果不使用 AcceptHead，仍然使用默认的 `ComputeAccuracy` 方法
- **无需修改**：现有的非 AcceptHead 训练配置无需任何修改即可正常工作

---

## 六、示例输出

启用评估后，训练日志中会显示类似以下内容：

```
Step 250: eval_loss=0.1234, eval_accuracy=0.8567, eval_mae=0.0234, eval_mse=0.0012, eval_rmse=0.0345
Step 500: eval_loss=0.0987, eval_accuracy=0.8923, eval_mae=0.0198, eval_mse=0.0009, eval_rmse=0.0300
Step 750: eval_loss=0.0876, eval_accuracy=0.9123, eval_mae=0.0176, eval_mse=0.0008, eval_rmse=0.0283
```

---

## 七、注意事项

1. **只计算 mismatch 位置**：指标只对 `label != IGNORE_INDEX` 的位置进行计算
2. **容差阈值选择**：根据任务需求选择合适的容差阈值，过小会导致 accuracy 过低，过大则失去意义
3. **评估频率**：`eval_steps` 设置过小会增加训练时间，设置过大则监控不够及时
4. **验证集大小**：`val_size` 建议设置为 0.05-0.1（5%-10%），太小可能不够代表性，太大则训练数据减少

---

## 八、相关文件

- `src/llamafactory/train/sft/metric.py`：指标计算实现
- `src/llamafactory/train/sft/workflow.py`：训练流程集成
- `src/llamafactory/hparams/finetuning_args.py`：配置参数定义
- `examples/train_lora/llama3_1_8b_lora_sft_accept_head.yaml`：配置示例

---

## 修改总结

本次修改共涉及 **3 个文件**：

1. ✅ `src/llamafactory/train/sft/metric.py`：添加 `accept_head_logit_processor` 函数和 `ComputeAcceptHeadAccuracy` 类
2. ✅ `src/llamafactory/hparams/finetuning_args.py`：添加 `accept_head_accuracy_tolerance` 配置参数
3. ✅ `src/llamafactory/train/sft/workflow.py`：修改 metric 选择逻辑，自动识别 AcceptHead

所有修改都保持了向后兼容性，不影响现有的非 AcceptHead 训练流程。

