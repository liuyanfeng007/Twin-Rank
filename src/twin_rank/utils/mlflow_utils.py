"""MLflow 相关的小工具：连接已有服务、参数展开、跨版本的 PyTorch 模型记录。
"""
import os

import mlflow


def setup_mlflow(cfg) -> None:
    """优先使用配置里的 tracking_uri，否则用环境变量 MLFLOW_TRACKING_URI。"""
    uri = cfg.mlflow.tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")
    if uri:
        mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)


def flatten_dict(d: dict, prefix: str = "") -> dict:
    """{'train': {'lr': 1}} -> {'train.lr': 1}，方便 mlflow.log_params。"""
    flat = {}
    for key, value in d.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten_dict(value, prefix=f"{name}."))
        else:
            flat[name] = value
    return flat


def log_pytorch_model(model, name: str = "model") -> None:
    """
    兼容不同版本的 MLflow：
    - MLflow 3 默认 'pt2' 格式需要 input_example，且参数名改为 name；这里显式用 pickle。
    - MLflow 2 没有 serialization_format / name，回退到 artifact_path。
    """
    import mlflow.pytorch
    try:
        mlflow.pytorch.log_model(model, name=name, serialization_format="pickle")
    except TypeError:
        mlflow.pytorch.log_model(model, artifact_path=name)
