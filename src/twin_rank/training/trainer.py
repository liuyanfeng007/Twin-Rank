"""训练循环：每个 epoch 训练 -> 验证 -> 记录指标 -> 早停并保留最佳权重。

日志通过 log_fn 回调解耦（默认接到 MLflow），所以这里不依赖 mlflow，也便于单测。
"""
import copy
import logging
from typing import Callable

import torch

from twin_rank.training.data import make_train_loader
from twin_rank.training.evaluate import evaluate
from twin_rank.training.loss import in_batch_softmax_loss

logger = logging.getLogger(__name__)


def train_one_epoch(model, loader, optimizer, log_q, temperature, device) -> float:
    model.train()
    total, steps = 0.0, 0
    for user_idx, hist, item_idx in loader:
        user_idx, hist, item_idx = user_idx.to(device), hist.to(device), item_idx.to(device)
        user_vec, item_vec = model(user_idx, hist, item_idx)
        loss = in_batch_softmax_loss(user_vec, item_vec, item_idx, log_q, temperature)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()
        steps += 1
    return total / max(steps, 1)


def fit(model, train_arrays, val_arrays, val_seen, log_q, cfg, device,
        log_fn: Callable[[dict, int], None]) -> dict:
    """返回验证集上最佳的 state_dict。log_fn(metrics, step) 用于对接 MLflow。"""
    loader = make_train_loader(train_arrays, cfg.train.batch_size)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )
    log_q = log_q.to(device)

    best_score, best_state, bad_epochs = -1.0, None, 0
    for epoch in range(1, cfg.train.epochs + 1):
        loss = train_one_epoch(model, loader, optimizer, log_q, cfg.model.temperature, device)
        val = evaluate(model, val_arrays, val_seen, cfg.train.eval_ks, device)

        metrics = {"train_loss": loss, **{f"val_{k}": v for k, v in val.items()}}
        log_fn(metrics, epoch)
        logger.info("epoch %d | loss %.4f | %s", epoch, loss,
                    {k: round(v, 4) for k, v in val.items()})

        score = val[cfg.train.monitor_metric]
        if score > best_score:
            best_score, bad_epochs = score, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.train.patience:
                logger.info("早停：%d 个 epoch 无提升", bad_epochs)
                break
    return best_state
