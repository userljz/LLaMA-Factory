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

import torch
import torch.nn as nn
from typing import TYPE_CHECKING

from ..extras import logging
from ..extras.constants import IGNORE_INDEX

if TYPE_CHECKING:
    from transformers import PreTrainedModel

logger = logging.get_logger(__name__)


def compute_accept_head_loss(
    model: "PreTrainedModel",
    inputs: dict[str, torch.Tensor],
    return_outputs: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict]:
    r"""Compute regression loss for AcceptHead using BCEWithLogitsLoss.
    
    Only computes loss on mismatch positions (where labels != IGNORE_INDEX).
    Mismatch positions are the only positions that have valid scores (0-1).
    
    Args:
        model: The model with AcceptHead as lm_head
        inputs: Dictionary containing:
            - input_ids: Tokenized input sequence
            - attention_mask: Attention mask
            - labels: Regression scores (float values 0-1 for mismatch positions, IGNORE_INDEX otherwise)
        return_outputs: Whether to return model outputs
    
    Returns:
        loss: Regression loss (BCEWithLogitsLoss) computed only on mismatch positions
        outputs: (optional) Model outputs if return_outputs=True
    """
    # Forward pass
    # Filter out labels (not needed for forward pass, only for loss computation)
    outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
    logits = outputs.logits  # [batch_size, seq_len, 1] for AcceptHead
    
    # Get labels (regression scores, 0-1 values for mismatch positions, IGNORE_INDEX otherwise)
    labels = inputs.get("labels")  # [batch_size, seq_len] with float scores (0-1) or IGNORE_INDEX
    
    if labels is None:
        raise ValueError("Labels are required for AcceptHead loss computation.")
    
    # Squeeze logits to [batch_size, seq_len]
    if logits.dim() == 3 and logits.size(-1) == 1:
        logits = logits.squeeze(-1)  # [batch_size, seq_len]
    
    # Create mask for valid positions (mismatch positions, not IGNORE_INDEX)
    # IGNORE_INDEX is typically -100, so we check for that
    mask = (labels != IGNORE_INDEX).float()
    
    # Extract target scores (0-1 values) only for mismatch positions
    # For masked positions, set to 0 (won't contribute to loss due to mask)
    target_scores = labels * mask
    
    # Ensure target_scores are in [0, 1] range (clamp to be safe)
    target_scores = torch.clamp(target_scores, min=0.0, max=1.0)
    
    # Compute BCEWithLogitsLoss only on mismatch positions
    # BCEWithLogitsLoss expects logits and targets in [0, 1]
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    loss = loss_fn(logits, target_scores) * mask
    
    # Average over mismatch positions only
    num_valid = mask.sum()
    if num_valid > 0:
        loss = loss.sum() / num_valid
    else:
        # No mismatch positions, return zero loss
        loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
        logger.warning_rank0("No mismatch positions found in batch, loss is 0.")
    
    if return_outputs:
        return loss, outputs
    return loss

