"""深度精排网络的公共部分。

每个网络的输入都是: item_idx [B]、dense [B, 9]（标准化后的数值特征）、hist [B, L]（用户历史商品编号）。
5 个稀疏字段 = 商品 ID + 类目 + 品牌 + 价格桶 + 热度桶，各有一张 embedding 表；
后四个字段由 item_idx 查物品特征表得到。输出 logits [B]（未过 sigmoid）。
"""
import numpy as np
import torch
import torch.nn as nn


def build_dnn(in_dim: int, hidden_dims: list, out_dim: int, dropout: float = 0.0) -> nn.Sequential:
    """Linear-ReLU-Dropout 堆叠，最后一层只有 Linear（不带激活）。"""
    layers, prev = [], in_dim
    for hidden in hidden_dims:
        layers += [nn.Linear(prev, hidden), nn.ReLU(), nn.Dropout(dropout)]
        prev = hidden
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class RankNet(nn.Module):
    N_SPARSE = 5

    def __init__(self, item_feats: np.ndarray, n_dense: int, cfg):
        super().__init__()
        feats = torch.as_tensor(item_feats, dtype=torch.long)
        self.register_buffer("item_feats", feats)          # [n_items, 4] = 类目, 品牌, 价格桶, 热度桶
        sizes = [feats.shape[0]] + [int(s) + 1 for s in feats.max(dim=0).values]
        self.d, self.n_dense = cfg.embedding_dim, n_dense
        self.tables = nn.ModuleList([nn.Embedding(n, self.d, padding_idx=0) for n in sizes])

    def sparse_ids(self, item_idx: torch.Tensor) -> torch.Tensor:
        """[B] -> [B, 5]：商品 ID 与它的 4 个类别特征。"""
        return torch.cat([item_idx.unsqueeze(1), self.item_feats[item_idx]], dim=1)

    def sparse_embeddings(self, item_idx: torch.Tensor) -> torch.Tensor:
        """[B] -> [B, 5, d]。"""
        ids = self.sparse_ids(item_idx)
        return torch.stack([table(ids[:, i]) for i, table in enumerate(self.tables)], dim=1)

    def forward(self, item_idx, dense, hist) -> torch.Tensor:
        raise NotImplementedError
