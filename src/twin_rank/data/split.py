"""按时间的 leave-last-out 切分。

每个用户：最后一次交互 = 测试，倒数第二次 = 验证，其余 = 训练。
这样训练只看得到"过去"，不会把未来信息泄露给模型；也模拟了"预测用户下一个行为"的线上场景。
"""
import numpy as np
import pandas as pd


def assign_split(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["user_id", "timestamp"], kind="stable").reset_index(drop=True)
    rank_from_end = df.groupby("user_id").cumcount(ascending=False)
    df["split"] = np.select([rank_from_end == 0, rank_from_end == 1], ["test", "val"], "train")
    return df
