"""DeepFM = 一阶线性项 + FM 二阶交叉 + DNN 高阶交叉，三部分共享同一组 embedding。

* FM 部分显式建模"任意两个字段两两相乘"的低阶交叉（如 类目×价格）；
* DNN 部分隐式学习高阶非线性交叉；
* 稠密特征用"数值 × 该字段的可学习向量"变成字段向量，与稀疏字段一起参与 FM。
"""
import torch
import torch.nn as nn

from twin_rank.ranking.deep.networks import RankNet, build_dnn


class DeepFM(RankNet):
    def __init__(self, item_feats, n_dense, cfg):
        super().__init__(item_feats, n_dense, cfg)
        self.dense_vectors = nn.Parameter(torch.randn(n_dense, self.d) * 0.01)
        self.first_order = nn.ModuleList([nn.Embedding(t.num_embeddings, 1, padding_idx=0) for t in self.tables])
        self.first_dense = nn.Linear(n_dense, 1)
        n_fields = self.N_SPARSE + n_dense
        self.dnn = build_dnn(n_fields * self.d, cfg.hidden_dims, 1, cfg.dropout)

    def forward(self, item_idx, dense, hist):
        ids = self.sparse_ids(item_idx)
        sparse = self.sparse_embeddings(item_idx)                              # [B, 5, d]
        fields = torch.cat([sparse, dense.unsqueeze(-1) * self.dense_vectors], dim=1)   # [B, 5+D, d]

        first = self.first_dense(dense)                                        # 一阶项 [B, 1]
        for i, emb in enumerate(self.first_order):
            first = first + emb(ids[:, i])
        # FM 二阶项: 0.5 * ((Σv)^2 - Σ(v^2))，对 d 维求和
        second = 0.5 * (fields.sum(dim=1).pow(2) - fields.pow(2).sum(dim=1)).sum(dim=-1, keepdim=True)
        deep = self.dnn(fields.flatten(1))                                     # [B, 1]
        return (first + second + deep).squeeze(-1)
