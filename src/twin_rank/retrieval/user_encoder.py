"""在线/离线共用：加载训练好的双塔，只用用户塔把 (用户, 历史) 编码成向量。
"""
import numpy as np
import torch

from twin_rank.models.two_tower import TwoTowerModel
from twin_rank.paths import two_tower_path


class UserEncoder:
    def __init__(self, model: TwoTowerModel):
        self.model = model.eval()

    @classmethod
    def load(cls, cfg, vocab: dict, item_feats: np.ndarray) -> "UserEncoder":
        model = TwoTowerModel(torch.from_numpy(item_feats), vocab, cfg.model)
        model.load_state_dict(torch.load(two_tower_path(cfg), map_location="cpu"))
        return cls(model)

    @torch.no_grad()
    def encode(self, user_idx: np.ndarray, hist: np.ndarray, batch_size: int = 4096) -> np.ndarray:
        outputs = []
        for s in range(0, len(user_idx), batch_size):
            users = torch.from_numpy(user_idx[s:s + batch_size]).long()
            hists = torch.from_numpy(hist[s:s + batch_size]).long()
            outputs.append(self.model.user_vector(users, hists).numpy())
        return np.concatenate(outputs)
