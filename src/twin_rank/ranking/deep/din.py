"""DIN（Deep Interest Network）：用候选商品对用户历史做"目标注意力"。

用户历史里每个商品对"当前这个候选"的相关程度不同：买过很多游戏手柄的用户，面对手柄候选时
手柄相关的历史应占更大权重。做法: 对每个历史商品，把 [候选, 历史, 候选-历史, 候选*历史]
送进小网络得到一个分数，在有效历史上 softmax 后加权求和，得到"针对这个候选的兴趣向量"，
再与候选的稀疏特征、稠密特征一起过 DNN 输出。商品表示 = 商品 ID embedding + 类目 embedding。
"""
import torch
import torch.nn as nn

from twin_rank.ranking.deep.networks import RankNet, build_dnn


class DIN(RankNet):
    def __init__(self, item_feats, n_dense, cfg):
        super().__init__(item_feats, n_dense, cfg)
        item_dim = 2 * self.d
        self.attention = build_dnn(4 * item_dim, cfg.attention_hidden, 1)
        in_dim = item_dim + self.N_SPARSE * self.d + n_dense
        self.dnn = build_dnn(in_dim, cfg.hidden_dims, 1, cfg.dropout)

    def item_repr(self, idx: torch.Tensor) -> torch.Tensor:
        """商品表示 = [ID embedding, 类目 embedding]；idx 可以是 [B] 或 [B, L]。"""
        category = self.item_feats[idx, 0]
        return torch.cat([self.tables[0](idx), self.tables[1](category)], dim=-1)

    def forward(self, item_idx, dense, hist):
        query = self.item_repr(item_idx)                        # [B, 2d]
        keys = self.item_repr(hist)                             # [B, L, 2d]
        q = query.unsqueeze(1).expand_as(keys)
        scores = self.attention(torch.cat([q, keys, q - keys, q * keys], dim=-1)).squeeze(-1)   # [B, L]

        mask = hist > 0                                         # 左侧补 0 的位置不参与注意力
        weights = torch.softmax(scores.masked_fill(~mask, -1e9), dim=-1) * mask   # 全是 padding 时权重为 0
        interest = (weights.unsqueeze(-1) * keys).sum(dim=1)    # [B, 2d]

        x = torch.cat([interest, self.sparse_embeddings(item_idx).flatten(1), dense], dim=1)
        return self.dnn(x).squeeze(-1)
