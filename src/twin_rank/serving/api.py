"""FastAPI 接口层，只做参数校验与路由，业务逻辑都在 RecommendService。

模型来源: 环境变量 TWIN_RANK_MODEL_URI 或配置 serving.model_uri 指向 MLflow 注册表时，从注册表加载；否则读本地 data/。
"""
import os

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from twin_rank.config import load_config
from twin_rank.serving.recommender import MODES, RecommendService
from twin_rank.serving.schemas import FeedbackRequest
from twin_rank.utils.common import setup_logging


def create_app() -> FastAPI:
    setup_logging()
    cfg, _ = load_config(os.environ.get("TWIN_RANK_CONFIG", "configs/config.yaml"))
    # 优先从 MLflow 注册表加载（环境变量 > 配置），否则用本地 data/ 目录里的文件
    model_uri = os.environ.get("TWIN_RANK_MODEL_URI") or cfg.serving.model_uri
    if model_uri:
        from twin_rank.registry.loader import load_service
        from twin_rank.utils.mlflow_utils import setup_mlflow
        setup_mlflow(cfg)
        service = load_service(model_uri)
    else:
        service = RecommendService(cfg)

    app = FastAPI(title="Twin-Rank", version="1.0.0")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/users/sample")
    def sample_users():
        return {"users": service.sample_users(cfg.serving.sample_users)}

    @app.get("/users/{user_id}/history")
    def history(user_id: str, n: int = Query(10, ge=1, le=50)):
        return {"user_id": user_id, "items": service.user_history(user_id, n)}

    @app.get("/recommend")
    def recommend(user_id: str, top_k: int = Query(10, ge=1, le=50),
                  mode: str = "full", explain: bool = False):
        if mode not in MODES:
            raise HTTPException(400, f"mode must be one of {MODES}")
        return service.recommend(user_id, top_k, mode, explain)

    @app.post("/feedback")
    def feedback(req: FeedbackRequest):
        try:
            service.add_feedback(req.user_id, req.item_id)
        except KeyError:
            raise HTTPException(404, "unknown user or item")
        return {"status": "ok"}

    return app
