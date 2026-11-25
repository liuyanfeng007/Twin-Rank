"""生成一份合成的"自定义数据"，用于演示如何接入自己的数据，或在没有网络时端到端冒烟测试。

用法:
    python tools/make_demo_custom_data.py            # 输出到 data/custom/
    然后把 configs/config.yaml 里 data.source 改成 custom，
    items_file 改成 data/custom/items.csv，columns.rating 改成 rating，再运行流水线。
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/custom")
    parser.add_argument("--users", type=int, default=3000)
    parser.add_argument("--items", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    n_cat, n_brand = 20, 60
    item_cat = rng.integers(0, n_cat, args.items)
    items = pd.DataFrame({
        "item_id": [f"I{i:05d}" for i in range(args.items)],
        "title": [f"商品 {i} (类目{c})" for i, c in enumerate(item_cat)],
        "category": [f"cat_{c}" for c in item_cat],
        "brand": [f"brand_{b}" for b in rng.integers(0, n_brand, args.items)],
        "price": np.round(rng.lognormal(3.0, 0.8, args.items), 2),
    })

    pop = 1.0 / np.arange(1, args.items + 1) ** 0.6              # 长尾热度
    rows = []
    for u in range(args.users):
        liked = rng.choice(n_cat, size=3, replace=False)          # 每个用户偏好 3 个类目
        weights = pop * np.where(np.isin(item_cat, liked), 8.0, 1.0)
        k = int(rng.integers(8, 40))
        chosen = rng.choice(args.items, size=k, replace=False, p=weights / weights.sum())
        times = np.sort(rng.integers(1_600_000_000, 1_700_000_000, k))
        for i, t in zip(chosen, times):
            rows.append((f"U{u:05d}", items.item_id[i], pd.to_datetime(t, unit="s").strftime("%Y-%m-%d %H:%M:%S"),
                         int(rng.integers(3, 6))))
    interactions = pd.DataFrame(rows, columns=["user_id", "item_id", "timestamp", "rating"])

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    interactions.to_csv(out / "interactions.csv", index=False)
    items.to_csv(out / "items.csv", index=False)
    print(f"已生成 {len(interactions)} 条行为、{len(items)} 个商品 -> {out}/")


if __name__ == "__main__":
    main()
