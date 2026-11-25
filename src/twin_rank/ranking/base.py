"""精排模型的统一接口。

LightGBM 与 DeepFM / DCN / DIN 都实现这套接口，所以训练流水线、评估、在线服务、模型注册
都只面对 Ranker，不关心具体是哪种模型；换模型只需改配置 ranker.model_type。

约定：frame 是 ranking/features.py 生成的"(用户, 候选)一行"的表；
hist 是 [用户数, max_history_len] 的历史数组，第 i 行的历史是 hist[frame.user_row[i]]。
LightGBM 不用 hist；DIN 用它做目标注意力。
"""
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd


class Ranker(ABC):
    model_type: str = ""

    @abstractmethod
    def fit(self, train_df: pd.DataFrame, train_hist: np.ndarray,
            valid_df: pd.DataFrame, valid_hist: np.ndarray) -> dict:
        """训练。返回学习曲线 {"组名": {"指标名": [每轮的值]}}，由流水线写入 MLflow。"""

    @abstractmethod
    def predict(self, frame: pd.DataFrame, hist: np.ndarray) -> np.ndarray:
        """每行一个分数，同一用户内分数越大越靠前。"""

    @abstractmethod
    def contributions(self, frame: pd.DataFrame, hist: np.ndarray) -> np.ndarray:
        """每个特征对分数的贡献，形状 [n, 特征数+1]（最后一列为偏置），用于生成推荐理由。"""

    @abstractmethod
    def save(self, path) -> None:
        """保存到文件。加载由 ranking/factory.py 的 load_ranker 负责。"""

    def feature_importance(self) -> Optional[pd.DataFrame]:
        """特征重要性表（feature, gain）；不支持的模型返回 None。"""
        return None
