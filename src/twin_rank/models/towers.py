"""用户塔与物品塔，输出 L2 归一化向量（因此向量内积 = 余弦相似度）。

物品塔: 物品 ID embedding + 类目/品牌/价格桶/热度桶 embedding -> 拼接 -> MLP。
用户塔: 用户 ID embedding + 历史商品 embedding 的 mask 平均池化 -> 拼接 -> MLP。
两塔完全独立，所以物品向量可以离线算好建索引，线上只需算一次用户向量。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from twin_rank.features.item_features import VOCAB_KEYS
from twin_rank.models.layers import build_mlp


class ItemTower(nn.Module):
    """物品 ID embedding + 类目/品牌/价格桶/热度桶 embedding -> MLP。"""

    def __init__(self, item_feats: torch.Tensor, vocab: dict, cfg):
        super().__init__()
        self.register_buffer("item_feats", item_feats)       # [n_items, 4]，随模型保存
        n_items = item_feats.shape[0]
        self.id_emb = nn.Embedding(n_items, cfg.embedding_dim, padding_idx=0)
        self.side_embs = nn.ModuleList(
            [nn.Embedding(vocab[key], cfg.side_dim, padding_idx=0) for key in VOCAB_KEYS]
        )
        in_dim = cfg.embedding_dim + cfg.side_dim * len(VOCAB_KEYS)
        self.mlp = build_mlp(in_dim, cfg.hidden_dims, cfg.output_dim)

    def forward(self, item_idx: torch.Tensor) -> torch.Tensor:
        feats = self.item_feats[item_idx]
        parts = [self.id_emb(item_idx)]
        parts += [emb(feats[:, i]) for i, emb in enumerate(self.side_embs)]
        return F.normalize(self.mlp(torch.cat(parts, dim=-1)), dim=-1)


class UserTower(nn.Module):
    """用户 ID embedding + 历史行为（mask 平均池化）-> MLP。"""

    def __init__(self, n_users: int, n_items: int, cfg):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, cfg.embedding_dim, padding_idx=0)
        self.hist_emb = nn.Embedding(n_items, cfg.embedding_dim, padding_idx=0)
        self.mlp = build_mlp(cfg.embedding_dim * 2, cfg.hidden_dims, cfg.output_dim)

    def forward(self, user_idx: torch.Tensor, hist: torch.Tensor) -> torch.Tensor:
        hist = hist.long()
        mask = (hist > 0).unsqueeze(-1).float()
        pooled = (self.hist_emb(hist) * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        features = torch.cat([self.user_emb(user_idx), pooled], dim=-1)
        return F.normalize(self.mlp(features), dim=-1)
