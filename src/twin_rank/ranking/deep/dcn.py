"""DCN-V2（Deep & Cross Network v2）：交叉网络 + DNN 并联。

交叉层: x_{l+1} = x0 ⊙ (W_l · x_l + b_l) + x_l。每多一层，可表达的特征交叉阶数就 +1，
所以 cross_layers=3 最高到 4 阶交叉；相比 FM 只有二阶，DCN 的高阶交叉是"显式、有界"的。
最后把交叉网络输出与 DNN 输出拼接，再接一层线性输出。
"""
import torch
import torch.nn as nn

from twin_rank.ranking.deep.networks import RankNet, build_dnn


class DCN(RankNet):
    def __init__(self, item_feats, n_dense, cfg):
        super().__init__(item_feats, n_dense, cfg)
        in_dim = self.N_SPARSE * self.d + n_dense
        self.cross = nn.ModuleList([nn.Linear(in_dim, in_dim) for _ in range(cfg.cross_layers)])
        self.dnn = build_dnn(in_dim, cfg.hidden_dims[:-1], cfg.hidden_dims[-1], cfg.dropout)
        self.out = nn.Linear(in_dim + cfg.hidden_dims[-1], 1)

    def forward(self, item_idx, dense, hist):
        x0 = torch.cat([self.sparse_embeddings(item_idx).flatten(1), dense], dim=1)   # [B, 5d + D]
        x = x0
        for layer in self.cross:
            x = x0 * layer(x) + x
        deep = torch.relu(self.dnn(x0))
        return self.out(torch.cat([x, deep], dim=1)).squeeze(-1)
