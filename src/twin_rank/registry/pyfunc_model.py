"""MLflow pyfunc 封装：让整套推荐系统可以用 mlflow.pyfunc.load_model(...).predict(DataFrame) 调用。

加载时（load_context）：从 bundle 里构建与线上完全相同的 RecommendService；
预测时（predict）：逐行解析请求 -> 调 RecommendService.recommend -> 展开成 DataFrame。
输入输出格式见 registry/io.py。
"""
from pathlib import Path

import mlflow.pyfunc
import pandas as pd

from twin_rank.registry.bundle import load_bundle_config
from twin_rank.registry.io import OUTPUT_COLUMNS, flatten_response, parse_requests


class TwinRankModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        from twin_rank.serving.recommender import RecommendService   # 延迟导入：重依赖只在加载时需要
        cfg = load_bundle_config(Path(context.artifacts["bundle"]))
        self.service = RecommendService(cfg)

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        frames = []
        for req in parse_requests(model_input):
            result = self.service.recommend(req.user_id, req.top_k, req.mode, req.explain, req.history)
            frames.append(flatten_response(result))
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=OUTPUT_COLUMNS)
