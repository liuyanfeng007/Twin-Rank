"""训练/评估所需数据的读取：样本数组、DataLoader、"已交互商品"稀疏矩阵。

评估时需要把用户已交互过的商品排除出候选，否则 Recall 会被已看过的商品抬高。
"""
import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix
from torch.utils.data import DataLoader, TensorDataset


def load_arrays(path) -> dict:
    with np.load(path) as f:
        return {key: f[key] for key in f.files}


def make_train_loader(arrays: dict, batch_size: int) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(arrays["user_idx"]).long(),
        torch.from_numpy(arrays["hist"]).long(),
        torch.from_numpy(arrays["item_idx"]).long(),
    )
    # drop_last: 避免最后一个 batch 过小导致 in-batch 负样本太少
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)


def build_seen_matrix(encoded: pd.DataFrame, splits: list, vocab: dict) -> csr_matrix:
    """用户已交互物品的稀疏矩阵，评估时把这些物品从候选里排除。"""
    seen = encoded[encoded["split"].isin(splits)]
    shape = (vocab["n_users"], vocab["n_items"])
    values = np.ones(len(seen), dtype="int8")
    return csr_matrix((values, (seen["user_idx"], seen["item_idx"])), shape=shape)
