"""双塔训练损失：batch 内负采样的 softmax 交叉熵 + logQ 修正。

* batch 内负采样：第 i 个用户的正样本是第 i 个商品，同 batch 其余商品都当负样本（免费且高效）；
* logQ 修正：热门商品更容易被当作负样本采到，会被过度打压。把 logits 减去 log(商品被采样概率) 抵消这种偏置；
* 同一 batch 里重复出现的同一商品不是真负样本，用 -inf 屏蔽。
"""

import numpy as np
import torch
import torch.nn.functional as F


def compute_log_q(train_item_idx: np.ndarray, n_items: int) -> torch.Tensor:
    """物品在训练样本中的出现概率的对数，用于 logQ 修正（打压热门偏置）。"""
    counts = np.bincount(train_item_idx, minlength=n_items).astype("float64")
    q = np.maximum(counts, 1.0) / counts.sum()
    return torch.from_numpy(np.log(q)).float()


def in_batch_softmax_loss(user_vec, item_vec, item_idx, log_q, temperature: float):
    """
    batch 内负采样的 softmax 损失：第 i 个用户的正样本是第 i 个物品，其余为负样本。
    - logQ 修正：logits 减去 log(物品被采样概率)
    - 同一 batch 内重复出现的同一物品不当作负样本（避免假负例）
    """
    logits = user_vec @ item_vec.T / temperature
    logits = logits - log_q[item_idx].unsqueeze(0)

    same_item = item_idx.unsqueeze(0) == item_idx.unsqueeze(1)
    diagonal = torch.eye(len(item_idx), dtype=torch.bool, device=logits.device)
    logits = logits.masked_fill(same_item & ~diagonal, float("-inf"))

    labels = torch.arange(len(item_idx), device=logits.device)
    return F.cross_entropy(logits, labels)
