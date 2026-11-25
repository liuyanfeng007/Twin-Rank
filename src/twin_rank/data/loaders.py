"""逐行读取 jsonl.gz，只保留需要的列，避免一次性把几 GB 的文件读进内存。
"""
import gzip
import json
from itertools import islice
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd


def _iter_records(path: Path) -> Iterator[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def read_jsonl_gz(path: Path, columns: list, max_rows: Optional[int] = None) -> pd.DataFrame:
    records = _iter_records(path)
    if max_rows:
        records = islice(records, max_rows)
    rows = ({col: rec.get(col) for col in columns} for rec in records)
    return pd.DataFrame(rows, columns=columns)
