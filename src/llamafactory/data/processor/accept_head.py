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

import json
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from .processor_utils import DatasetProcessor

if TYPE_CHECKING:
    from ..mm_plugin import AudioInput, ImageInput, VideoInput


logger = logging.get_logger(__name__)


@dataclass
class AcceptHeadDatasetProcessor(DatasetProcessor):
    r"""Processor for AcceptHead dataset format.
    
    Expected format from converter:
    - req_id: Request ID to lookup context
    - context_len: Length of context to use
    - spd_round_draft_ids: List of draft token IDs
    - spd_round_verifier_ids: List of verifier token IDs
    - mismatch_index: List of indices where draft and verifier tokens mismatch (relative to draft tokens)
    - mismatch_score: List of scores (0-10) for mismatch positions
    - user_text: User text from context
    - assistant_text: Assistant text from context
    
    Input sequence: context_current + [<sep>] + draft_ids + [<sep>] + verifier_ids
    Labels: Only mismatch positions have scores (normalized 0-1), others are IGNORE_INDEX
    """

    def __post_init__(self):
        """Initialize context mapping."""
        self.context_map = {}
        # Try to load context.jsonl from dataset directory
        # This will be populated during preprocessing if context data is available
        self._context_loaded = False

    def _load_context_map(self, examples: dict[str, list[Any]]) -> None:
        """Load context mapping from examples if available."""
        if self._context_loaded:
            return
        
        # Check if context data is in examples (from converter)
        if "_context_map" in examples and examples["_context_map"]:
            # Context map is already loaded by converter
            self.context_map = examples["_context_map"][0] if examples["_context_map"] else {}
            self._context_loaded = True
            logger.info_rank0(f"Loaded {len(self.context_map)} context entries.")

    def _encode_data_example(
        self,
        req_id: str,
        context_len: int,
        draft_ids: list[int],
        verifier_ids: list[int],
        mismatch_index: list[int],
        mismatch_score: list[str],
        user_text: Optional[str] = None,
        assistant_text: Optional[str] = None,
    ) -> tuple[list[int], list[int]]:
        r"""Encode a single example.
        
        Args:
            req_id: Request ID for context lookup
            context_len: Length of context to use
            draft_ids: List of draft token IDs
            verifier_ids: List of verifier token IDs
            mismatch_index: List of mismatch positions (relative to draft tokens)
            mismatch_score: List of scores (0-10) for mismatch positions
            user_text: User text from context (if available)
            assistant_text: Assistant text from context (if available)
        
        Returns:
            input_ids: Tokenized input sequence
            labels: Labels for loss computation (scores 0-1 for mismatch positions, IGNORE_INDEX otherwise)
        """
        # Get separator token ID
        sep_token_id = self.tokenizer.sep_token_id if hasattr(self.tokenizer, "sep_token_id") and self.tokenizer.sep_token_id is not None else self.tokenizer.eos_token_id
        
        # Build full context if user_text and assistant_text are provided
        if user_text and assistant_text:
            full_context_text = f"USER: {user_text}\nASSISTANT:{assistant_text}"
            # truncation is performed on the full context text
            context_current = full_context_text[:context_len]
            context_current = self.tokenizer.encode(context_current, add_special_tokens=False)
        else:
            raise ValueError("user_text and assistant_text are required")
        
        # Build input sequence: context_current + [<sep>] + draft_ids + [<sep>] + verifier_ids
        input_ids = []
        labels = []
        
        # Add context (all masked)
        input_ids.extend(context_current)
        labels.extend([IGNORE_INDEX] * len(context_current))
        
        # Add first separator
        input_ids.append(sep_token_id)
        labels.append(IGNORE_INDEX)
        
        # Add draft tokens (all masked initially, will set mismatch positions)
        input_ids.extend(draft_ids)
        labels.extend([IGNORE_INDEX] * len(draft_ids))
        
        # Add second separator
        input_ids.append(sep_token_id)
        labels.append(IGNORE_INDEX)
        
        # Add verifier tokens (all masked)
        input_ids.extend(verifier_ids)
        labels.extend([IGNORE_INDEX] * len(verifier_ids))
        
        # Set labels for mismatch positions
        # Mismatch positions in the sequence: actual_context_len + 1 + mismatch_index[i]
        # Note: We use actual_context_len (token count) instead of context_len (character count)
        actual_context_len = len(context_current)
        for m_idx, m_score_str in zip(mismatch_index, mismatch_score):
            # Normalize score from 0-10 to 0-1
            try:
                m_score = float(m_score_str) / 10.0
                m_score = max(0.0, min(1.0, m_score))  # Clamp to [0, 1]
            except (ValueError, TypeError):
                logger.warning_rank0(f"Invalid mismatch score: {m_score_str}, using 0.0")
                m_score = 0.0
            
            # Position in sequence: after context + separator + draft position
            pos_in_seq = actual_context_len + 1 + m_idx
            
            # Check bounds
            if 0 <= m_idx < len(draft_ids) and pos_in_seq < len(labels):
                labels[pos_in_seq] = m_score
            else:
                logger.warning_rank0(
                    f"Mismatch index {m_idx} out of bounds (draft_len={len(draft_ids)}, "
                    f"seq_len={len(labels)}, pos={pos_in_seq})"
                )
        
        # Truncate to cutoff_len
        if len(input_ids) > self.data_args.cutoff_len:
            input_ids = input_ids[:self.data_args.cutoff_len]
            labels = labels[:self.data_args.cutoff_len]
        
        return input_ids, labels

    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        r"""Preprocess dataset examples.
        
        Expected columns from converter:
        - req_id: Request ID
        - context_len: Context length
        - spd_round_draft_ids: Draft token IDs
        - spd_round_verifier_ids: Verifier token IDs
        - mismatch_index: Mismatch positions
        - mismatch_score: Mismatch scores (0-10)
        - user_text: User text (optional, from context)
        - assistant_text: Assistant text (optional, from context)
        """
        # Load context map if available
        self._load_context_map(examples)
        
        model_inputs = defaultdict(list)
        
        # Get batch size
        batch_size = len(examples.get("req_id", []))
        if batch_size == 0:
            return model_inputs
        
        for i in range(batch_size):
            try:
                req_id = str(examples.get("req_id", [""])[i])
                context_len = int(examples.get("context_len", [0])[i])
                draft_ids = examples.get("spd_round_draft_ids", [[]])[i]
                verifier_ids = examples.get("spd_round_verifier_ids", [[]])[i]
                mismatch_index = examples.get("mismatch_index", [[]])[i]
                mismatch_score = examples.get("mismatch_score", [[]])[i]
                user_text = examples.get("user_text", [None])[i]
                assistant_text = examples.get("assistant_text", [None])[i]
                
                # Ensure lists
                if not isinstance(draft_ids, list):
                    draft_ids = list(draft_ids) if draft_ids else []
                if not isinstance(verifier_ids, list):
                    verifier_ids = list(verifier_ids) if verifier_ids else []
                if not isinstance(mismatch_index, list):
                    mismatch_index = list(mismatch_index) if mismatch_index else []
                if not isinstance(mismatch_score, list):
                    mismatch_score = list(mismatch_score) if mismatch_score else []
                
                # Validate mismatch data
                if len(mismatch_index) != len(mismatch_score):
                    logger.warning_rank0(
                        f"Mismatch index and score length mismatch: {len(mismatch_index)} vs {len(mismatch_score)}"
                    )
                    min_len = min(len(mismatch_index), len(mismatch_score))
                    mismatch_index = mismatch_index[:min_len]
                    mismatch_score = mismatch_score[:min_len]
                
                input_ids, labels = self._encode_data_example(
                    req_id=req_id,
                    context_len=context_len,
                    draft_ids=draft_ids,
                    verifier_ids=verifier_ids,
                    mismatch_index=mismatch_index,
                    mismatch_score=mismatch_score,
                    user_text=user_text,
                    assistant_text=assistant_text,
                )
                
                model_inputs["input_ids"].append(input_ids)
                model_inputs["attention_mask"].append([1] * len(input_ids))
                model_inputs["labels"].append(labels)
                
            except Exception as e:
                logger.warning_rank0(f"Error processing example {i}: {e}")
                continue
        
        return model_inputs

    def print_data_example(self, example: dict[str, list[int]]) -> None:
        valid_labels = [x for x in example["labels"] if x != IGNORE_INDEX]
        print("input_ids:\n{}".format(example["input_ids"]))
        print("inputs:\n{}".format(self.tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
        print("label_ids:\n{}".format(example["labels"]))
        print(f"valid_labels (scores):\n{valid_labels}")
        print(f"Number of mismatch positions: {len(valid_labels)}")

