"""Category-conditioned global object-identity retrieval head."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class ObjectIdentityHead(nn.Module):
    """Retrieve a geometry asset from one shared, category-masked asset bank.

    The asset embeddings are learned jointly with the rest of the model.  A
    category may own any number of assets; no padding or per-category output
    layer is needed because invalid global assets are masked before softmax.
    """

    def __init__(
        self,
        hidden_size: int,
        asset_category_ids: Sequence[int],
        embedding_dim: int = 256,
    ) -> None:
        super().__init__()
        if not asset_category_ids:
            raise ValueError("ObjectIdentityHead needs at least one asset")
        if min(asset_category_ids) < 0:
            raise ValueError("asset category IDs must be non-negative")

        self.query = nn.Sequential(
            nn.Linear(hidden_size, embedding_dim),
            nn.ELU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.asset_embeddings = nn.Embedding(len(asset_category_ids), embedding_dim)
        self.logit_scale = nn.Parameter(torch.tensor(1.0).log())
        self.register_buffer(
            "asset_category_ids",
            torch.as_tensor(asset_category_ids, dtype=torch.long),
            persistent=True,
        )
        nn.init.normal_(self.asset_embeddings.weight, std=embedding_dim**-0.5)

    @property
    def num_assets(self) -> int:
        return self.asset_embeddings.num_embeddings

    def forward(
        self,
        hidden_states: torch.Tensor,
        category_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Return global asset logits with other categories set to ``-inf``."""

        if hidden_states.shape[:-1] != category_ids.shape:
            raise ValueError(
                "category_ids must match hidden_states without its feature axis: "
                f"{tuple(category_ids.shape)} vs {tuple(hidden_states.shape)}"
            )
        query_dtype = self.query[0].weight.dtype
        query = self.query(hidden_states.to(dtype=query_dtype))
        query = F.normalize(query.float(), dim=-1)
        assets = F.normalize(self.asset_embeddings.weight.float(), dim=-1)
        logits = query @ assets.transpose(0, 1)
        logits = logits * self.logit_scale.exp().clamp(max=100.0)
        valid = category_ids[..., None] == self.asset_category_ids
        if not torch.all(valid.any(dim=-1)):
            missing = torch.unique(category_ids[~valid.any(dim=-1)]).tolist()
            raise ValueError(f"No identity candidates for category IDs {missing}")
        return logits.masked_fill(~valid, torch.finfo(logits.dtype).min)

    def loss(
        self,
        hidden_states: torch.Tensor,
        target_asset_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return masked CE loss and top-1 accuracy; ``-100`` labels are ignored."""

        target_asset_ids = target_asset_ids.to(device=hidden_states.device, dtype=torch.long)
        valid = target_asset_ids != -100
        if not torch.any(valid):
            zero = hidden_states.sum() * 0.0
            return zero, zero.detach()
        targets = target_asset_ids[valid]
        if torch.any((targets < 0) | (targets >= self.num_assets)):
            raise ValueError("target asset index is outside the global asset bank")
        categories = self.asset_category_ids[targets]
        logits = self(hidden_states[valid], categories)
        loss = F.cross_entropy(logits, targets)
        accuracy = (logits.argmax(dim=-1) == targets).float().mean()
        return loss, accuracy

    @torch.no_grad()
    def predict(
        self,
        hidden_states: torch.Tensor,
        category_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self(hidden_states, category_ids)
        probabilities = torch.softmax(logits, dim=-1)
        return probabilities.argmax(dim=-1), probabilities
