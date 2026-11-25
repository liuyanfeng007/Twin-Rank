"""按配置 ranker.model_type 创建 / 加载精排模型。

模型实现都延迟导入：选 lightgbm 时不会加载深度网络，反之亦然，也让配置错误能先给出清晰的提示。
"""
from pathlib import Path

from twin_rank.constants import RANKER_TYPES
from twin_rank.paths import ranker_path, ranker_type
from twin_rank.ranking.base import Ranker


def _check(kind: str) -> None:
    if kind not in RANKER_TYPES:
        raise ValueError(f"ranker.model_type 必须是 {RANKER_TYPES} 之一，收到 {kind!r}")


def build_ranker(cfg, tables) -> Ranker:
    """创建一个未训练的精排模型。tables 是 ItemTables（深度模型需要物品特征表）。"""
    kind = ranker_type(cfg)
    _check(kind)
    if kind == "lightgbm":
        from twin_rank.ranking.lgbm import LightGBMRanker
        return LightGBMRanker(cfg.ranker)
    from twin_rank.ranking.deep.ranker import DeepRanker
    return DeepRanker(kind, cfg.ranker, tables)


def load_ranker(cfg, tables) -> Ranker:
    """加载已训练好的精排模型（文件位置由 paths.ranker_path 决定）。"""
    kind = ranker_type(cfg)
    _check(kind)
    path = Path(ranker_path(cfg))
    if not path.exists():
        raise FileNotFoundError(
            f"找不到精排模型 {path}（ranker.model_type={kind}）。请先运行 train_ranker，"
            f"或检查 model_type 是否与已训练的类型一致（也可用 TWIN_RANK_RANKER 临时切换）")
    if kind == "lightgbm":
        from twin_rank.ranking.lgbm import LightGBMRanker
        return LightGBMRanker.load(path, cfg.ranker)
    from twin_rank.ranking.deep.ranker import DeepRanker
    return DeepRanker.load(path, tables)
