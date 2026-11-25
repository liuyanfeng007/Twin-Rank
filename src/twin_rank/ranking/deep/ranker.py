"""深度精排的训练与推理封装：DeepFM / DCN / DIN 共用。

训练: 点估计的二分类（BCE，候选里是真实目标 = 1，其余 = 0）；每轮在验证集上算 NDCG@10，
早停并保留最佳权重。输出分数为点击概率 sigmoid(logit)。
推荐理由: 深度模型没有树模型那样的特征贡献，这里用"遮挡法"近似——把某个稠密特征换成训练集均值，
看分数下降多少，下降越多说明该特征对这条推荐贡献越大。类别特征(类目/价格桶/热度桶)不参与遮挡。
"""
import logging
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from twin_rank.ranking.base import Ranker
from twin_rank.ranking.deep.dcn import DCN
from twin_rank.ranking.deep.deepfm import DeepFM
from twin_rank.ranking.deep.din import DIN
from twin_rank.ranking.deep.features import DENSE_FEATURES, Scaler, raw_dense
from twin_rank.ranking.features import FEATURE_NAMES
from twin_rank.ranking.metrics import metrics_from_positions, positions_by_score
from twin_rank.utils.common import resolve_device

logger = logging.getLogger(__name__)
NETWORKS = {"deepfm": DeepFM, "dcn": DCN, "din": DIN}


class DeepRanker(Ranker):
    def __init__(self, model_type: str, cfg_ranker, tables):
        self.model_type = model_type
        self.cfg = cfg_ranker.deep
        self.tables = tables
        self.net = NETWORKS[model_type](tables.item_feats, len(DENSE_FEATURES), self.cfg)
        self.scaler = None

    # ---------- 数据准备 ----------
    def _prepare(self, frame, hist, with_label: bool = False) -> dict:
        data = {
            "item": torch.from_numpy(frame["item_idx"].to_numpy()).long(),
            "dense": torch.from_numpy(self.scaler.transform(raw_dense(frame))),
            "user_row": torch.from_numpy(frame["user_row"].to_numpy()).long(),
            "hist": torch.from_numpy(np.asarray(hist)).long(),        # [用户数, L]，按 user_row 取行，避免逐行复制
        }
        if with_label:
            data["label"] = torch.from_numpy(frame["label"].to_numpy().astype("float32"))
        return data

    def _logits(self, data: dict, rows: torch.Tensor, device) -> torch.Tensor:
        return self.net(data["item"][rows].to(device), data["dense"][rows].to(device),
                        data["hist"][data["user_row"][rows]].to(device))

    @torch.no_grad()
    def _predict_logits(self, data: dict, device="cpu", batch_size: int = 8192) -> np.ndarray:
        self.net.eval()
        outputs = []
        for start in range(0, len(data["item"]), batch_size):
            rows = torch.arange(start, min(start + batch_size, len(data["item"])))
            outputs.append(self._logits(data, rows, device).cpu().numpy())
        return np.concatenate(outputs) if outputs else np.zeros(0, dtype="float32")

    # ---------- 训练 ----------
    def fit(self, train_df, train_hist, valid_df, valid_hist) -> dict:
        cfg = self.cfg
        device = resolve_device(cfg.device)
        self.scaler = Scaler.fit(raw_dense(train_df))
        train = self._prepare(train_df, train_hist, with_label=True)
        valid = self._prepare(valid_df, valid_hist)
        valid_groups = np.unique(valid_df["user_row"].to_numpy(), return_inverse=True)[1]
        valid_label = valid_df["label"].to_numpy()

        self.net.to(device)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        curves = {"train": {"loss": []}, "valid": {"ndcg@10": []}}
        best_score, best_state, bad_epochs = -1.0, None, 0
        n = len(train["item"])

        for epoch in range(1, cfg.epochs + 1):
            self.net.train()
            order, total, steps = torch.randperm(n), 0.0, 0
            for start in range(0, n, cfg.batch_size):
                rows = order[start:start + cfg.batch_size]
                loss = F.binary_cross_entropy_with_logits(self._logits(train, rows, device),
                                                          train["label"][rows].to(device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total, steps = total + loss.item(), steps + 1

            scores = self._predict_logits(valid, device)
            positions = positions_by_score(valid_groups, valid_label, scores, valid_groups.max() + 1)
            ndcg = metrics_from_positions(positions, [10])["ndcg_at_10"]
            curves["train"]["loss"].append(total / max(steps, 1))
            curves["valid"]["ndcg@10"].append(ndcg)
            logger.info("%s epoch %d | loss %.4f | valid ndcg@10 %.4f", self.model_type, epoch,
                        total / max(steps, 1), ndcg)

            if ndcg > best_score:
                best_score, bad_epochs = ndcg, 0
                best_state = {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}
            else:
                bad_epochs += 1
                if bad_epochs >= cfg.patience:
                    logger.info("早停：%d 个 epoch 无提升", bad_epochs)
                    break

        self.net.load_state_dict(best_state)
        self.net.cpu().eval()
        return curves

    # ---------- 推理 ----------
    def predict(self, frame, hist) -> np.ndarray:
        logits = self._predict_logits(self._prepare(frame, hist))
        return (1.0 / (1.0 + np.exp(-logits))).astype("float32")

    def contributions(self, frame, hist) -> np.ndarray:
        data = self._prepare(frame, hist)
        base = self._predict_logits(data)
        contrib = np.zeros((len(frame), len(FEATURE_NAMES) + 1), dtype="float32")
        for j, name in enumerate(DENSE_FEATURES):
            saved = data["dense"][:, j].clone()
            data["dense"][:, j] = 0.0                    # 标准化后的 0 = 训练集均值
            contrib[:, FEATURE_NAMES.index(name)] = base - self._predict_logits(data)
            data["dense"][:, j] = saved
        return contrib

    # ---------- 保存 / 加载 ----------
    def save(self, path) -> None:
        torch.save({"model_type": self.model_type, "deep_cfg": vars(self.cfg),
                    "state_dict": self.net.state_dict(), "scaler": self.scaler.state()}, path)

    @classmethod
    def load(cls, path, tables) -> "DeepRanker":
        ckpt = torch.load(path, map_location="cpu")
        ranker = cls(ckpt["model_type"], SimpleNamespace(deep=SimpleNamespace(**ckpt["deep_cfg"])), tables)
        ranker.net.load_state_dict(ckpt["state_dict"])
        ranker.net.eval()
        ranker.scaler = Scaler.from_state(ckpt["scaler"])
        return ranker
