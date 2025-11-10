# AcceptHead 功能实现 - 修改总结

本文档总结了为实现 AcceptHead 功能对 LLaMA-Factory 代码库的所有修改。

---

## 修改概述

本次修改实现了两个核心功能：

1. **替换 LM Head**：将模型的 `lm_head` 替换为 AcceptHead（双层 MLP）
2. **自定义数据处理和损失函数**：支持新的数据格式（train.jsonl + context.jsonl），只对 mismatch 位置计算 BCEWithLogitsLoss

---

## 一、新建文件（2个）

### 1. `src/llamafactory/model/model_utils/accept_head.py`（68行）

**目的**：定义 AcceptHead 模块，用于替换原始的 LM Head

**功能**：
- 定义 `AcceptHead` 类：带归一化的双层 MLP
- 结构：`LayerNorm(H) → Linear(H→H/4) → GELU → Linear(H/4→1)`
- 输出：logits（未经过 sigmoid）

**关键特性**：
- 使用 LayerNorm 对输入进行归一化，提高训练稳定性
- fc1 使用 He 初始化（Kaiming uniform），适合 GELU 激活函数
- fc2 使用小增益 Xavier 初始化（gain=0.1），适合回归任务
- 输出维度为 1（回归任务）

**代码示例**：
```python
class AcceptHead(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 4)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(hidden_size // 4, 1)
        self._init_weights()
    
    def forward(self, hidden_states):
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation(hidden_states)
        logits = self.fc2(hidden_states)
        return logits.squeeze(-1)
```

---

### 2. `src/llamafactory/train/accept_head_loss.py`（94行）

**目的**：实现 AcceptHead 的自定义损失函数

**功能**：
- 使用 BCEWithLogitsLoss 计算回归损失
- **只计算 mismatch 位置的损失**（其他位置被 mask）
- 处理 IGNORE_INDEX 掩码

**关键特性**：
- 只对 labels != IGNORE_INDEX 的位置计算损失
- 使用 BCEWithLogitsLoss（数值稳定）
- 平均损失只考虑 mismatch 位置

**代码示例**：
```python
def compute_accept_head_loss(model, inputs, return_outputs=False):
    outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
    logits = outputs.logits.squeeze(-1)
    labels = inputs.get("labels")
    
    mask = (labels != IGNORE_INDEX).float()
    target_scores = labels * mask
    target_scores = torch.clamp(target_scores, min=0.0, max=1.0)
    
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    loss = loss_fn(logits, target_scores) * mask
    
    num_valid = mask.sum()
    loss = loss.sum() / num_valid if num_valid > 0 else torch.tensor(0.0, ...)
    
    return loss
```

---

## 二、修改文件（7个）

### 1. `src/llamafactory/hparams/model_args.py`

**修改位置**：第161-164行

**修改内容**：添加 `replace_lm_head` 配置参数

```python
replace_lm_head: bool = field(
    default=False,
    metadata={"help": "Whether or not to replace lm_head with AcceptHead (a 2-layer MLP)."},
)
```

**目的**：允许在 YAML 配置文件中通过 `replace_lm_head: true` 来启用 LM head 替换

---

### 2. `src/llamafactory/model/patcher.py`

**修改1：导入 AcceptHead**（第27行）
```python
from .model_utils.accept_head import AcceptHead
```

**修改2：添加替换函数**（第51-110行）
- 函数名：`_replace_lm_head_with_accept_head(model)`
- 功能：查找并替换模型中的 `lm_head` 为 `AcceptHead`

**修改3：在 patch_model 中调用**（第248-249行）
```python
if model_args.replace_lm_head:
    _replace_lm_head_with_accept_head(model)
```

**目的**：在模型加载后自动替换 LM Head

---

### 3. `src/llamafactory/hparams/data_args.py`

**修改位置**：第140-149行

**修改内容**：添加 `use_accept_head_format` 配置参数

```python
use_accept_head_format: bool = field(
    default=False,
    metadata={
        "help": (
            "Whether or not to use AcceptHead dataset format. "
            "Format: [context]+[<sep>]+[draft tokens]+[<sep>]+[target tokens], "
            "with regression scores for draft tokens."
        )
    },
)
```

**目的**：允许在 YAML 配置文件中启用 AcceptHead 数据格式

---

### 4. `src/llamafactory/data/parser.py`

**修改位置**：第33行

**修改内容**：添加 "accept_head" 格式化类型

**修改前**：
```python
formatting: Literal["alpaca", "sharegpt", "openai"] = "alpaca"
```

**修改后**：
```python
formatting: Literal["alpaca", "sharegpt", "openai", "accept_head"] = "alpaca"
```

**目的**：在 `DatasetAttr` 类的 `formatting` 字段中添加 `"accept_head"` 选项，使其成为有效的格式化类型

---

### 5. `src/llamafactory/data/converter.py`

**修改1：添加 `AcceptHeadDatasetConverter` 类**（第370-402行）

```python
@dataclass
class AcceptHeadDatasetConverter(DatasetConverter):
    r"""Converter for AcceptHead dataset format."""
    
    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        output = {
            "req_id": str(example.get("req_id", "")),
            "context_len": int(example.get("context_len", 0)),
            "spd_round_draft_ids": example.get("spd_round_draft_ids", []),
            "spd_round_verifier_ids": example.get("spd_round_verifier_ids", []),
            "mismatch_index": example.get("mismatch_index", []),
            "mismatch_score": example.get("mismatch_score", []),
            "user_text": example.get("user_text"),
            "assistant_text": example.get("assistant_text"),
        }
        return output
```

**目的**：
- 字段筛选：只保留需要的字段，移除不需要的字段（`idx`, `context_last_20_word` 等）
- 类型转换：确保 `req_id` 为 str，`context_len` 为 int
- 提供默认值保护

---

**修改2：更新 `DATASET_CONVERTERS` 字典**（第405-410行）

```python
DATASET_CONVERTERS = {
    "alpaca": AlpacaDatasetConverter,
    "sharegpt": SharegptDatasetConverter,
    "openai": OpenAIDatasetConverter,
    "accept_head": AcceptHeadDatasetConverter,  # 新增
}
```

**目的**：注册 AcceptHeadDatasetConverter，使其可以通过 `formatting: "accept_head"` 使用

---

**修改3：修改 `align_dataset()` 函数**（第429-465行）

在函数开头添加 context 合并逻辑：

```python
def align_dataset(...):
    # Special handling for AcceptHead format: merge context.jsonl
    if dataset_attr.formatting == "accept_head" and dataset_attr.load_from == "file":
        dataset = _merge_accept_head_context(dataset, dataset_attr, data_args, training_args)
    
    # ... 原有代码 ...
```

**目的**：在数据转换前自动加载并合并 context.jsonl

---

**修改4：添加 `_merge_accept_head_context()` 函数**（第468-530行）

```python
def _merge_accept_head_context(
    dataset: Union["Dataset", "IterableDataset"],
    dataset_attr: "DatasetAttr",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
) -> Union["Dataset", "IterableDataset"]:
    r"""Merge context.jsonl with train.jsonl for AcceptHead format."""
    # 1. 查找 context.jsonl 文件
    # 2. 加载 context.jsonl 并创建 context_map (req_id -> context)
    # 3. 使用 dataset.map() 合并 context 数据到每个样本
    # ...
```

**目的**：自动加载 context.jsonl 并根据 `req_id` 合并到训练样本

---

### 6. `src/llamafactory/data/processor/accept_head.py`

**修改类型**：完全重写（245行）

**目的**：处理 AcceptHead 格式的数据，构建输入序列和 labels

**主要功能**：

1. **`__post_init__()`**（第50-55行）
   - 初始化 context_map

2. **`_load_context_map()`**（第57-67行）
   - 加载上下文映射（防御性编程，实际 context 已通过 converter 合并）

3. **`_encode_data_example()`**（第69-163行）
   - 构建完整上下文：`full_context = f"USER: {user_text}\nASSISTANT:{assistant_text}"`
   - 截取上下文（字符级）：`context_current = full_context[:context_len]`
   - Tokenize：`context_current = tokenizer.encode(context_current)`
   - 构建输入序列：`context_current + [<sep>] + draft_ids + [<sep>] + verifier_ids`
   - 设置 labels：只对 mismatch 位置设置归一化分数（0-1），其他位置为 IGNORE_INDEX
   - 位置计算：`actual_context_len + 1 + mismatch_index[i]`（使用 token 数）

4. **`preprocess_dataset()`**（第165-237行）
   - 从 examples 中提取所有字段
   - 调用 `_encode_data_example()` 处理每个样本
   - 返回 `input_ids`, `attention_mask`, `labels`

5. **`print_data_example()`**（第239-245行）
   - 打印数据示例，显示 mismatch 位置数量

**关键逻辑**：
```python
# 1. 构建上下文
full_context = f"USER: {user_text}\nASSISTANT:{assistant_text}"
context_current = full_context[:context_len]  # context_len 是字符数
context_current = tokenizer.encode(context_current, add_special_tokens=False)

# 2. 构建输入序列
input_ids = context_current + [<sep>] + draft_ids + [<sep>] + verifier_ids
labels = [IGNORE_INDEX] * len(input_ids)

# 3. 设置 mismatch 位置的 labels
actual_context_len = len(context_current)  # token 数
for m_idx, m_score in zip(mismatch_index, mismatch_score):
    pos = actual_context_len + 1 + m_idx
    labels[pos] = float(m_score) / 10.0  # 归一化到 0-1
```

---

### 7. `src/llamafactory/train/sft/trainer.py`

**修改1：导入损失函数**（第31行）
```python
from ..accept_head_loss import compute_accept_head_loss
```

**修改2：设置自定义损失函数**（第92-95行）
```python
# Use AcceptHead loss if replace_lm_head is enabled
if model_args is not None and model_args.replace_lm_head:
    self.compute_loss_func = compute_accept_head_loss
    logger.info_rank0("Using AcceptHead regression loss (BCEWithLogitsLoss).")
```

**修改3：使用自定义损失函数**（第122-126行）
```python
@override
def compute_loss(self, model, inputs, *args, **kwargs):
    # Use custom loss function if set (e.g., AcceptHead loss)
    if hasattr(self, "compute_loss_func"):
        return self.compute_loss_func(model, inputs, return_outputs=False)
    return super().compute_loss(model, inputs, *args, **kwargs)
```

**目的**：当 `replace_lm_head=True` 时，使用自定义的 AcceptHead 损失函数

---

## 三、文件修改清单（按类别）

### 模型相关（3个文件）

| 文件 | 修改类型 | 主要内容 |
|------|---------|---------|
| `src/llamafactory/model/model_utils/accept_head.py` | 新建 | AcceptHead 模块定义 |
| `src/llamafactory/hparams/model_args.py` | 添加参数 | `replace_lm_head: bool` |
| `src/llamafactory/model/patcher.py` | 添加函数 | `_replace_lm_head_with_accept_head()` |

### 数据处理相关（4个文件）

| 文件 | 修改类型 | 主要内容 |
|------|---------|---------|
| `src/llamafactory/hparams/data_args.py` | 添加参数 | `use_accept_head_format: bool` |
| `src/llamafactory/data/parser.py` | 修改一行 | 添加 `"accept_head"` 格式化类型 |
| `src/llamafactory/data/converter.py` | 添加类和函数 | `AcceptHeadDatasetConverter` + `_merge_accept_head_context()` |
| `src/llamafactory/data/processor/accept_head.py` | 完全重写 | `AcceptHeadDatasetProcessor` |

### 训练相关（2个文件）

| 文件 | 修改类型 | 主要内容 |
|------|---------|---------|
| `src/llamafactory/train/accept_head_loss.py` | 新建 | `compute_accept_head_loss()` |
| `src/llamafactory/train/sft/trainer.py` | 添加逻辑 | 使用自定义损失函数 |

---

## 四、详细修改说明

### 第一部分：LM Head 替换功能

#### 文件1：`src/llamafactory/model/model_utils/accept_head.py`（新建，68行）

**完整内容**：
```python
import torch.nn as nn

class AcceptHead(nn.Module):
    """AcceptHead: LayerNorm + 2-layer MLP for regression."""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 4)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(hidden_size // 4, 1)
        self._init_weights()
    
    def _init_weights(self):
        nn.init.kaiming_uniform_(self.fc1.weight, a=0, mode='fan_in', nonlinearity='relu')
        nn.init.xavier_uniform_(self.fc2.weight, gain=0.1)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
    
    def forward(self, hidden_states):
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation(hidden_states)
        logits = self.fc2(hidden_states)
        return logits.squeeze(-1)
```

---

#### 文件2：`src/llamafactory/hparams/model_args.py`（添加参数）

**位置**：第161-164行

**添加**：
```python
replace_lm_head: bool = field(
    default=False,
    metadata={"help": "Whether or not to replace lm_head with AcceptHead (a 2-layer MLP)."},
)
```

---

#### 文件3：`src/llamafactory/model/patcher.py`（3处修改）

**修改1**：第27行
```python
from .model_utils.accept_head import AcceptHead
```

**修改2**：第51-110行（添加函数）
```python
def _replace_lm_head_with_accept_head(model: "PreTrainedModel") -> None:
    """Replace lm_head with AcceptHead."""
    # 1. 获取 hidden_size
    # 2. 创建 AcceptHead
    # 3. 查找并替换 lm_head
    # 4. 保持 device 和 dtype
```

**修改3**：第248-249行
```python
if model_args.replace_lm_head:
    _replace_lm_head_with_accept_head(model)
```

---

### 第二部分：数据处理和损失函数

#### 文件4：`src/llamafactory/hparams/data_args.py`（添加参数）

**位置**：第140-149行

**添加**：
```python
use_accept_head_format: bool = field(
    default=False,
    metadata={"help": "Whether to use AcceptHead dataset format."}
)
```

---

#### 文件5：`src/llamafactory/data/parser.py`（修改一行）

**位置**：第33行

**修改**：添加 `"accept_head"` 到格式化类型

---

#### 文件6：`src/llamafactory/data/converter.py`（4处修改）

**修改1**：第370-402行（添加类）
- `AcceptHeadDatasetConverter` 类

**修改2**：第405-410行（更新字典）
- 在 `DATASET_CONVERTERS` 中添加 `"accept_head": AcceptHeadDatasetConverter`

**修改3**：第429-465行（修改函数）
- 在 `align_dataset()` 中添加 context 合并逻辑

**修改4**：第468-530行（添加函数）
- `_merge_accept_head_context()` 函数，自动加载并合并 context.jsonl

---

#### 文件7：`src/llamafactory/data/processor/accept_head.py`（完全重写，245行）

**主要方法**：

1. **`__post_init__()`**（第50-55行）
   - 初始化 context_map

2. **`_load_context_map()`**（第57-67行）
   - 加载上下文映射（防御性编程）

3. **`_encode_data_example()`**（第69-163行）
   - 核心逻辑：构建输入序列和 labels
   - 接收参数：`req_id`, `context_len`, `draft_ids`, `verifier_ids`, `mismatch_index`, `mismatch_score`, `user_text`, `assistant_text`
   - 返回：`input_ids`, `labels`

4. **`preprocess_dataset()`**（第165-237行）
   - 批量处理数据
   - 返回：`input_ids`, `attention_mask`, `labels`

5. **`print_data_example()`**（第239-245行）
   - 打印数据示例

**关键逻辑**：
```python
# 构建上下文（字符级截取）
context_current = full_context[:context_len]
context_current = tokenizer.encode(context_current)

# 构建输入序列
input_ids = context_current + [<sep>] + draft_ids + [<sep>] + verifier_ids
labels = [IGNORE_INDEX] * len(input_ids)

# 设置 mismatch 位置的 labels
for m_idx, m_score in zip(mismatch_index, mismatch_score):
    pos = len(context_current) + 1 + m_idx  # 使用 actual_context_len (token 数)
    labels[pos] = float(m_score) / 10.0  # 归一化到 0-1
```

---

#### 文件8：`src/llamafactory/train/accept_head_loss.py`（新建，94行）

**主要函数**：`compute_accept_head_loss()`

**功能**：
```python
def compute_accept_head_loss(model, inputs, return_outputs=False):
    # 1. Forward pass（过滤 labels）
    outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
    logits = outputs.logits.squeeze(-1)
    
    # 2. 获取 labels
    labels = inputs.get("labels")
    
    # 3. 创建 mask（只计算 mismatch 位置）
    mask = (labels != IGNORE_INDEX).float()
    
    # 4. 计算 BCEWithLogitsLoss
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    loss = loss_fn(logits, labels * mask) * mask
    
    # 5. 平均损失（只对 mismatch 位置）
    num_valid = mask.sum()
    loss = loss.sum() / num_valid if num_valid > 0 else 0.0
    
    return loss
```

---

#### 文件9：`src/llamafactory/train/sft/trainer.py`（3处修改）

**修改1**：第31行（导入）
```python
from ..accept_head_loss import compute_accept_head_loss
```

**修改2**：第92-95行（设置损失函数）
```python
if model_args is not None and model_args.replace_lm_head:
    self.compute_loss_func = compute_accept_head_loss
    logger.info_rank0("Using AcceptHead regression loss (BCEWithLogitsLoss).")
```

**修改3**：第122-126行（使用损失函数）
```python
def compute_loss(self, model, inputs, *args, **kwargs):
    if hasattr(self, "compute_loss_func"):
        return self.compute_loss_func(model, inputs, return_outputs=False)
    return super().compute_loss(model, inputs, *args, **kwargs)
```

---

## 五、使用指南

### 1. 数据准备

创建两个 JSONL 文件：

**文件结构**：
```
data/
  your_dataset/
    train.jsonl      # 训练样本
    context.jsonl    # 上下文数据（自动加载）
```

**train.jsonl 格式**：
```jsonl
{"idx": 0, "req_id": "8", "context_len": 606, "spd_round_draft_ids": [311, 279, 1495], "spd_round_verifier_ids": [311, 279, 1495], "mismatch_index": [1], "mismatch_score": ["4"]}
```

**字段说明**：
- `req_id`: 请求ID（用于匹配 context）
- `context_len`: 上下文长度（字符数，不是 token 数）
- `spd_round_draft_ids`: Draft token IDs（列表）
- `spd_round_verifier_ids`: Verifier token IDs（列表）
- `mismatch_index`: Mismatch 位置索引（相对于 draft tokens，从0开始）
- `mismatch_score`: Mismatch 分数（0-10，字符串列表）

**context.jsonl 格式**：
```jsonl
{"req_id": "8", "user_text": "User's question", "assistant_text": "Assistant's answer"}
```

---

### 2. 数据集注册

在 `data/dataset_info.json` 中添加：

```json
{
  "your_dataset": {
    "file_name": "train.jsonl",
    "formatting": "accept_head"
  }
}
```

**注意**：`context.jsonl` 会自动加载，无需配置。

---

### 3. YAML 配置

**完整配置示例**：
```yaml
### model
model_name_or_path: meta-llama/Meta-Llama-3.1-8B-Instruct
replace_lm_head: true  # ⭐ 启用 LM Head 替换

### dataset
dataset: your_dataset
use_accept_head_format: true  # ⭐ 启用 AcceptHead 数据格式
template: llama3
cutoff_len: 2048

### method
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 8
lora_target: all
additional_target: lm_head  # ⭐ 使 AcceptHead 可训练

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1.0e-4
num_train_epochs: 3.0
bf16: true
output_dir: ./saves/accept_head
```

**关键配置说明**：
- `replace_lm_head: true` - 必须，用于替换 LM Head
- `use_accept_head_format: true` - 必须，用于选择 AcceptHead 数据处理器
- `formatting: "accept_head"` - 在 dataset_info.json 中指定，用于选择数据转换器
- `additional_target: lm_head` - 可选，如果需要训练 AcceptHead

---

### 4. 运行训练

```bash
llamafactory-cli train examples/train_lora/llama3_1_8b_lora_sft_accept_head.yaml
```

---

## 六、数据处理流程

### 完整流程

```
1. 加载 train.jsonl（HuggingFace datasets）
   ↓
2. 合并 context.jsonl（_merge_accept_head_context）
   - 根据 req_id 查找 context
   - 添加 user_text, assistant_text 到每个样本
   ↓
3. 数据转换（AcceptHeadDatasetConverter）
   - 字段筛选和类型转换
   - 移除不需要的字段
   ↓
4. 数据预处理（AcceptHeadDatasetProcessor）
   - 构建完整上下文：USER: {user_text}\nASSISTANT:{assistant_text}
   - 截取字符：full_context[:context_len]
   - Tokenize：tokenizer.encode()
   - 构建序列：context + [<sep>] + draft + [<sep>] + verifier
   - 设置 labels：只对 mismatch 位置设置分数（0-1）
   ↓
5. 训练（CustomSeq2SeqTrainer）
   ↓
6. 损失计算（compute_accept_head_loss）
   - 只对 mismatch 位置计算 BCEWithLogitsLoss
```

### 关键数据转换

**输入数据**：
```json
{
  "req_id": "8",
  "context_len": 606,
  "spd_round_draft_ids": [311, 279, 1495],
  "spd_round_verifier_ids": [311, 3230, 1495],
  "mismatch_index": [1],
  "mismatch_score": ["4"],
  "user_text": "...",
  "assistant_text": "..."
}
```

**处理步骤**：
```python
# 1. 构建上下文
full_context = "USER: ...\nASSISTANT:..."
context_current = full_context[:606]  # 字符级截取
context_tokens = tokenizer.encode(context_current)  # [tok1, tok2, ...]

# 2. 构建输入序列
input_ids = context_tokens + [<sep>] + [311, 279, 1495] + [<sep>] + [311, 3230, 1495]

# 3. 设置 labels
labels = [IGNORE_INDEX, IGNORE_INDEX, ..., IGNORE_INDEX, IGNORE_INDEX, 0.4, IGNORE_INDEX, ...]
#         |---- context -----|  sep  |--- draft ---|  sep  |--- verifier ---|
#                                          ^
#                                    mismatch_index[0]=1
#                                    score = 4/10 = 0.4
```

**输出数据**：
```python
{
  "input_ids": [...],
  "attention_mask": [...],
  "labels": [...]  # 只有 mismatch 位置有分数，其他为 IGNORE_INDEX
}
```

---

## 七、关键概念

### 1. context_len 的含义

- **类型**：整数
- **单位**：字符数（不是 token 数）
- **用途**：从完整上下文中截取前 N 个字符
- **处理**：先截取字符，再 tokenize

### 2. mismatch_index 的含义

- **类型**：整数列表
- **单位**：相对于 draft tokens 的索引（从0开始）
- **用途**：标识 draft tokens 和 verifier tokens 不匹配的位置
- **位置计算**：`actual_context_len + 1 + mismatch_index[i]`
  - `actual_context_len`：tokenize 后的 context token 数
  - `+ 1`：第一个 `<sep>` token
  - `+ mismatch_index[i]`：在 draft tokens 中的相对位置

### 3. mismatch_score 的含义

- **类型**：字符串列表
- **范围**：0-10
- **用途**：对 mismatch 位置的打分
- **处理**：归一化到 0-1（`float(score) / 10.0`）

### 4. 损失计算策略

- **只计算 mismatch 位置的损失**
- 使用 BCEWithLogitsLoss（数值稳定）
- 平均损失只考虑 mismatch 位置（不是所有 draft tokens）

---

## 八、修改总结表

| 序号 | 文件路径 | 修改类型 | 行数 | 修改目的 |
|------|---------|---------|------|---------|
| 1 | `src/llamafactory/model/model_utils/accept_head.py` | 新建 | 68 | 定义 AcceptHead 模块 |
| 2 | `src/llamafactory/hparams/model_args.py` | 添加参数 | 4 | 添加 `replace_lm_head` 参数 |
| 3 | `src/llamafactory/model/patcher.py` | 添加函数 | ~60 | 添加替换逻辑 |
| 4 | `src/llamafactory/hparams/data_args.py` | 添加参数 | 10 | 添加 `use_accept_head_format` 参数 |
| 5 | `src/llamafactory/data/parser.py` | 修改一行 | 1 | 添加 "accept_head" 格式化类型 |
| 6 | `src/llamafactory/data/converter.py` | 添加类和函数 | ~165 | 数据转换和 context 合并 |
| 7 | `src/llamafactory/data/processor/accept_head.py` | 完全重写 | 245 | AcceptHead 数据预处理 |
| 8 | `src/llamafactory/train/accept_head_loss.py` | 新建 | 94 | 自定义损失函数 |
| 9 | `src/llamafactory/train/sft/trainer.py` | 添加逻辑 | ~10 | 使用自定义损失函数 |

**总计**：9个文件，~600行代码

---

## 九、验证清单

在使用修改后的代码前，请确认：

- [ ] 所有新建文件已创建
- [ ] 所有修改文件已更新
- [ ] 数据集格式正确（train.jsonl + context.jsonl）
- [ ] dataset_info.json 已配置
- [ ] YAML 配置正确（包含所有必要参数）
- [ ] context.jsonl 与 train.jsonl 在同一目录
- [ ] req_id 匹配
- [ ] mismatch_index 和 mismatch_score 长度一致

---

## 十、训练时的日志验证

训练开始后，检查日志：

1. **模型替换**：
   - ✅ 应看到 "Replaced lm_head with AcceptHead"

2. **数据加载**：
   - ✅ 应看到 "Loaded X context entries from context.jsonl"

3. **损失函数**：
   - ✅ 应看到 "Using AcceptHead regression loss (BCEWithLogitsLoss)."

4. **训练**：
   - ✅ 损失值应该是合理的正数
   - ❌ 如果损失为 0，检查 mismatch 位置是否正确
   - ❌ 如果损失为 NaN，检查数据格式和分数范围

---

## 十一、常见问题

### Q1: 损失一直为 0？
**A**: 检查 mismatch_index 和 mismatch_score 是否正确，确保有有效的 mismatch 位置。

### Q2: Context 未加载？
**A**: 确保 context.jsonl 与 train.jsonl 在同一目录。

### Q3: 损失函数未生效？
**A**: 检查 YAML 配置中的 `replace_lm_head: true` 和 `use_accept_head_format: true`。

### Q4: Mismatch 位置错误？
**A**: 检查位置计算：`actual_context_len + 1 + mismatch_index[i]`，确保 mismatch_index 小于 draft_ids 长度。

### Q5: AcceptHead 不可训练？
**A**: 在 YAML 中添加 `additional_target: lm_head`。

---

## 十二、参考文档

- `ACCEPT_HEAD_MODIFICATION.md` - LM Head 替换实现
- `ACCEPT_HEAD_BCE_LOSS_UPDATE.md` - 损失函数更新说明
- `ACCEPT_HEAD_DATASET_AND_LOSS.md` - 数据集和损失函数详细说明
- `ACCEPT_HEAD_DATA_PROCESSING_CALL_CHAIN.md` - 数据处理调用链详解
- `CODE_REVIEW_REPORT.md` - 代码审查报告

---

## 十三、配置文件自动保存功能

### 功能概述

实现了自动将训练时使用的 YAML/JSON 配置文件保存到 checkpoint 目录的功能，方便追溯训练配置和恢复训练。

### 修改文件（3个）

#### 1. `src/llamafactory/hparams/parser.py`

**修改位置**：第68-88行

**修改内容**：在 `read_args()` 函数中记录配置文件路径

**修改前**：
```python
def read_args(args: Optional[Union[dict[str, Any], list[str]]] = None) -> Union[dict[str, Any], list[str]]:
    r"""Get arguments from the command line or a config file."""
    if args is not None:
        return args

    if sys.argv[1].endswith(".yaml") or sys.argv[1].endswith(".yml"):
        override_config = OmegaConf.from_cli(sys.argv[2:])
        dict_config = OmegaConf.load(Path(sys.argv[1]).absolute())
        return OmegaConf.to_container(OmegaConf.merge(dict_config, override_config))
    # ...
```

**修改后**：
```python
def read_args(args: Optional[Union[dict[str, Any], list[str]]] = None) -> Union[dict[str, Any], list[str]]:
    r"""Get arguments from the command line or a config file."""
    if args is not None:
        return args

    if sys.argv[1].endswith(".yaml") or sys.argv[1].endswith(".yml"):
        config_path = Path(sys.argv[1]).absolute()
        # Store config file path in environment variable for later use
        os.environ["LLAMAFACTORY_CONFIG_PATH"] = str(config_path)
        override_config = OmegaConf.from_cli(sys.argv[2:])
        dict_config = OmegaConf.load(config_path)
        return OmegaConf.to_container(OmegaConf.merge(dict_config, override_config))
    elif sys.argv[1].endswith(".json"):
        config_path = Path(sys.argv[1]).absolute()
        # Store config file path in environment variable for later use
        os.environ["LLAMAFACTORY_CONFIG_PATH"] = str(config_path)
        override_config = OmegaConf.from_cli(sys.argv[2:])
        dict_config = OmegaConf.load(config_path)
        return OmegaConf.to_container(OmegaConf.merge(dict_config, override_config))
    # ...
```

**目的**：
- 读取 YAML/JSON 配置文件时，将配置文件路径保存到环境变量 `LLAMAFACTORY_CONFIG_PATH`
- 支持后续 callback 读取并复制配置文件到 checkpoint 目录

---

#### 2. `src/llamafactory/train/callbacks.py`

**修改1：导入模块**（第15-23行）
```python
import json
import os
import shutil  # 新增
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path  # 新增
from typing import TYPE_CHECKING, Any, Optional
```

**修改2：添加 SaveConfigCallback 类**（第387-420行）

**完整代码**：
```python
class SaveConfigCallback(TrainerCallback):
    r"""A callback for saving the training configuration file to checkpoint directories."""

    def __init__(self) -> None:
        self.config_path: Optional[str] = os.getenv("LLAMAFACTORY_CONFIG_PATH")

    @override
    def on_save(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        r"""Save config file to checkpoint directory when checkpoint is saved."""
        if args.should_save and self.config_path and os.path.exists(self.config_path):
            output_dir = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
            # Ensure directory exists (should already exist, but check for safety)
            os.makedirs(output_dir, exist_ok=True)
            try:
                config_filename = Path(self.config_path).name
                dest_path = os.path.join(output_dir, config_filename)
                shutil.copy2(self.config_path, dest_path)
                logger.info_rank0(f"Configuration file saved to {dest_path}")
            except Exception as e:
                logger.warning_rank0(f"Failed to save configuration file to checkpoint: {e}")

    @override
    def on_train_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        r"""Save config file to final output directory when training ends."""
        if args.should_save and self.config_path and os.path.exists(self.config_path):
            # Ensure directory exists
            os.makedirs(args.output_dir, exist_ok=True)
            try:
                config_filename = Path(self.config_path).name
                dest_path = os.path.join(args.output_dir, config_filename)
                shutil.copy2(self.config_path, dest_path)
                logger.info_rank0(f"Configuration file saved to {dest_path}")
            except Exception as e:
                logger.warning_rank0(f"Failed to save configuration file to output directory: {e}")
```

**功能说明**：
- `on_save()`: 每次保存 checkpoint 时，自动将配置文件复制到对应的 checkpoint 目录（如 `checkpoint-500/`）
- `on_train_end()`: 训练结束时，将配置文件复制到最终输出目录
- 使用 `shutil.copy2()` 保留文件的元数据（时间戳等）
- 包含错误处理，失败时只记录警告，不影响训练流程

---

#### 3. `src/llamafactory/train/tuner.py`

**修改1：导入 SaveConfigCallback**（第30行）
```python
from .callbacks import LogCallback, PissaConvertCallback, ReporterCallback, SaveConfigCallback
```

**修改2：注册 callback**（第67行）
```python
callbacks.append(SaveConfigCallback())  # Save config file to checkpoints
callbacks.append(ReporterCallback(model_args, data_args, finetuning_args, generating_args))  # add to last
```

**目的**：在训练流程中自动注册 `SaveConfigCallback`，无需额外配置

---

### 使用方式

**无需额外配置**，使用 YAML/JSON 文件启动训练时自动生效：

```bash
llamafactory-cli train examples/train_lora/llama3_1_8b_lora_sft_accept_head.yaml
```

### 保存位置

配置文件会自动保存到以下位置：

1. **每个 checkpoint 目录**：
   ```
   saves/llama3.1-8b/lora/sft-accept-head/checkpoint-500/llama3_1_8b_lora_sft_accept_head.yaml
   saves/llama3.1-8b/lora/sft-accept-head/checkpoint-1000/llama3_1_8b_lora_sft_accept_head.yaml
   ...
   ```

2. **最终输出目录**：
   ```
   saves/llama3.1-8b/lora/sft-accept-head/llama3_1_8b_lora_sft_accept_head.yaml
   ```

### 功能优势

1. **自动保存**：无需手动配置，使用 YAML/JSON 文件启动时自动保存
2. **便于追溯**：每个 checkpoint 都包含完整的训练配置，方便查看训练参数
3. **便于恢复**：恢复训练时可以直接使用 checkpoint 目录中的配置文件
4. **向后兼容**：不使用配置文件启动时不影响原有功能

### 日志输出

训练时会看到以下日志：

```
Configuration file saved to saves/llama3.1-8b/lora/sft-accept-head/checkpoint-500/llama3_1_8b_lora_sft_accept_head.yaml
Configuration file saved to saves/llama3.1-8b/lora/sft-accept-head/llama3_1_8b_lora_sft_accept_head.yaml
```

### 修改总结表（更新）

| 序号 | 文件路径 | 修改类型 | 行数 | 修改目的 |
|------|---------|---------|------|---------|
| 1 | `src/llamafactory/model/model_utils/accept_head.py` | 新建 | 68 | 定义 AcceptHead 模块 |
| 2 | `src/llamafactory/hparams/model_args.py` | 添加参数 | 4 | 添加 `replace_lm_head` 参数 |
| 3 | `src/llamafactory/model/patcher.py` | 添加函数 | ~60 | 添加替换逻辑 |
| 4 | `src/llamafactory/hparams/data_args.py` | 添加参数 | 10 | 添加 `use_accept_head_format` 参数 |
| 5 | `src/llamafactory/data/parser.py` | 修改一行 | 1 | 添加 "accept_head" 格式化类型 |
| 6 | `src/llamafactory/data/converter.py` | 添加类和函数 | ~165 | 数据转换和 context 合并 |
| 7 | `src/llamafactory/data/processor/accept_head.py` | 完全重写 | 245 | AcceptHead 数据预处理 |
| 8 | `src/llamafactory/train/accept_head_loss.py` | 新建 | 94 | 自定义损失函数 |
| 9 | `src/llamafactory/train/sft/trainer.py` | 添加逻辑 | ~10 | 使用自定义损失函数 |
| 10 | `src/llamafactory/hparams/parser.py` | 修改函数 | ~10 | 记录配置文件路径 |
| 11 | `src/llamafactory/train/callbacks.py` | 添加类 | ~35 | 保存配置文件到 checkpoint |
| 12 | `src/llamafactory/train/tuner.py` | 注册 callback | 2 | 注册 SaveConfigCallback |

**总计**：12个文件，~650行代码

---

**文档版本**：v1.1
**最后更新**：2025-01
**维护者**：LLaMA-Factory Team

