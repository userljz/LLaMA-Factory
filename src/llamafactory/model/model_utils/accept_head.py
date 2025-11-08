# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# jz1108
import torch
import torch.nn as nn


class AcceptHead(nn.Module):
    """
    轻量级回归头，将隐藏状态映射为接受概率的 logits

    结构：LayerNorm → Linear(H→H/4) → GELU → Linear(H/4→1)

    注意：输出 logits（未经过 sigmoid），使用 BCEWithLogitsLoss 计算损失以提高数值稳定性
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 4)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_size // 4, 1)
        
        # 权重初始化
        self._init_weights()

    def _init_weights(self):
        """
        初始化权重：
        - fc1: 使用 He 初始化（适合 GELU 激活函数）
        - fc2: 使用较小的 Xavier 初始化（回归到单个值）
        """
        # fc1: He 初始化（Kaiming uniform），适合 GELU
        nn.init.kaiming_uniform_(self.fc1.weight, a=0, mode='fan_in', nonlinearity='relu')
        if self.fc1.bias is not None:
            nn.init.zeros_(self.fc1.bias)
        
        # fc2: 较小的 Xavier 初始化，适合回归任务
        nn.init.xavier_uniform_(self.fc2.weight, gain=0.1)
        if self.fc2.bias is not None:
            nn.init.zeros_(self.fc2.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [B, K, H] 或 [B, L, H] - K个mismatch位置的隐藏状态或序列的隐藏状态

        Returns:
            accept_logits: [B, K] 或 [B, L] - 接受概率的 logits（未经过 sigmoid）
        """
        x = self.layer_norm(hidden_states)  # [B, K, H] 或 [B, L, H]
        x = self.fc1(x)  # [B, K, H/4] 或 [B, L, H/4]
        x = self.act(x)  # [B, K, H/4] 或 [B, L, H/4]
        x = self.fc2(x)  # [B, K, 1] 或 [B, L, 1]
        return x.squeeze(-1)  # [B, K] 或 [B, L]

