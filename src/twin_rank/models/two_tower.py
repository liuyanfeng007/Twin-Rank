"""双塔模型 = 用户塔 + 物品塔。

forward 同时返回两个向量用于训练；线上只用 user_vector；
all_item_vectors 一次性算出全量物品向量，用于评估和构建 Faiss 索引。
"""

import torch
import torch.nn as nn

from twin_rank.models.towers import ItemTower, UserTower


class TwoTowerModel(nn.Module):
    def __init__(self, item_feats: torch.Tensor, vocab: dict, cfg):
        super().__init__()
        self.user_tower = UserTower(vocab["n_users"], item_feats.shape[0], cfg)
        self.item_tower = ItemTower(item_feats, vocab, cfg)

    def user_vector(self, user_idx, hist):
        return self.user_tower(user_idx, hist)

    def item_vector(self, item_idx):
        return self.item_tower(item_idx)

    def forward(self, user_idx, hist, item_idx):
        return self.user_vector(user_idx, hist), self.item_vector(item_idx)

    @torch.no_grad()
    def all_item_vectors(self, batch_size: int = 8192) -> torch.Tensor:
        """全量物品向量 [n_items, D]（第 0 行是 padding），供评估与 Faiss 建索引。"""
        device = next(self.parameters()).device
        n_items = self.item_tower.item_feats.shape[0]
        chunks = [
            self.item_vector(torch.arange(s, min(s + batch_size, n_items), device=device))
            for s in range(0, n_items, batch_size)
        ]
        return torch.cat(chunks, dim=0)
