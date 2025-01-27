from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch
from transformers.utils import ModelOutput

keys_ordering = ['loss', 'm1_ids', 'm2_ids', 'm1_embeds', 'm2_embeds']

@dataclass
class VLMOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    m1_ids: torch.LongTensor = None
    m2_ids: torch.LongTensor = None
    m1_embeds: torch.FloatTensor = None
    m2_embeds: torch.FloatTensor = None
    m1_m2_logits: Optional[torch.FloatTensor] = None

    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            self[k] for k in keys_ordering if k in self.keys() and k is not None
        )
