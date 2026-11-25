"""通用小组件：可配置层数的 MLP。
"""

import torch.nn as nn


def build_mlp(in_dim: int, hidden_dims: list, out_dim: int) -> nn.Sequential:
    layers, prev = [], in_dim
    for hidden in hidden_dims:
        layers += [nn.Linear(prev, hidden), nn.ReLU()]
        prev = hidden
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)
