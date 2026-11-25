"""从 MLflow 注册表加载模型。"""
import mlflow.pyfunc


def default_model_uri(cfg) -> str:
    return f"models:/{cfg.registry.model_name}@{cfg.registry.alias}"


def load_model(uri: str):
    """返回 mlflow 的 PyFuncModel，直接 .predict(DataFrame) 即可。"""
    return mlflow.pyfunc.load_model(uri)


def load_service(uri: str):
    """取出模型内部的 RecommendService（带 sample_users / user_history / add_feedback 等方法），
    供 FastAPI 服务直接使用注册表里的版本。"""
    model = load_model(uri)
    python_model = model.unwrap_python_model() if hasattr(model, "unwrap_python_model") \
        else model._model_impl.python_model
    return python_model.service
