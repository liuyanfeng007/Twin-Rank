"""调用 Twin-Rank 推荐 API。地址可用环境变量 TWIN_RANK_API_URL 覆盖。"""
import os

import httpx

BASE_URL = os.environ.get("TWIN_RANK_API_URL", "http://127.0.0.1:8100")


async def get(path: str, **params) -> dict:
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=30) as client:
        resp = await client.get(path, params=params)
        resp.raise_for_status()
        return resp.json()


async def post(path: str, payload: dict) -> dict:
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=30) as client:
        resp = await client.post(path, json=payload)
        resp.raise_for_status()
        return resp.json()
