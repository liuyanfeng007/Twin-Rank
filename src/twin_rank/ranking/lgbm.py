"""LightGBM 精排：训练、预测、特征贡献、特征重要性。

objective: lambdarank 直接优化排序（需要按用户分组）；binary 做点击率二分类。
必须先 import torch 再 import lightgbm（macOS 上避免两份 libomp 冲突）。
"""
import torch  # noqa: F401  必须先于 lightgbm 加载，避免 macOS 双 libomp 冲突
import lightgbm as lgb
import numpy as np
import pandas as pd

from twin_rank.ranking.base import Ranker
from twin_rank.ranking.features import CATEGORICAL, FEATURE_NAMES

OBJECTIVES = {
    "lambdarank": {"objective": "lambdarank", "metric": "ndcg", "eval_at": [10]},
    "binary": {"objective": "binary", "metric": "auc"},
}


def _dataset(df: pd.DataFrame, objective: str, reference=None) -> lgb.Dataset:
    # lambdarank 需要每个用户一组；同一用户的行在 frame 里是连续的
    group = df.groupby("user_row", sort=False).size().to_numpy() if objective == "lambdarank" else None
    return lgb.Dataset(df[FEATURE_NAMES], label=df["label"], group=group,
                       categorical_feature=CATEGORICAL, reference=reference, free_raw_data=False)


class LightGBMRanker(Ranker):
    model_type = "lightgbm"

    def __init__(self, cfg_ranker, booster: lgb.Booster = None):
        self.cfg = cfg_ranker
        self.booster = booster

    def fit(self, train_df, train_hist, valid_df, valid_hist) -> dict:
        cfg = self.cfg
        params = {**vars(cfg.params), **OBJECTIVES[cfg.objective], "verbosity": -1}
        dtrain = _dataset(train_df, cfg.objective)
        dvalid = _dataset(valid_df, cfg.objective, reference=dtrain)
        curves = {}
        self.booster = lgb.train(
            params, dtrain, num_boost_round=cfg.num_boost_round,
            valid_sets=[dvalid], valid_names=["valid"],
            callbacks=[lgb.early_stopping(cfg.early_stopping_rounds),
                       lgb.log_evaluation(25), lgb.record_evaluation(curves)],
        )
        return curves

    def _predict(self, frame: pd.DataFrame, contrib: bool = False):
        return self.booster.predict(frame[FEATURE_NAMES], num_iteration=self.booster.best_iteration or None,
                                    pred_contrib=contrib)

    def predict(self, frame, hist) -> np.ndarray:
        return self._predict(frame)

    def contributions(self, frame, hist) -> np.ndarray:
        return self._predict(frame, contrib=True)

    def save(self, path) -> None:
        self.booster.save_model(str(path))

    @classmethod
    def load(cls, path, cfg_ranker) -> "LightGBMRanker":
        return cls(cfg_ranker, lgb.Booster(model_file=str(path)))

    def feature_importance(self) -> pd.DataFrame:
        return pd.DataFrame({
            "feature": FEATURE_NAMES,
            "gain": self.booster.feature_importance(importance_type="gain"),
        }).sort_values("gain", ascending=False)
