import random

import httpx
import reflex as rx

from . import api_client
from .formatters import MODE_LABELS, to_card, to_history


class State(rx.State):
    user_options: list[str] = []
    user_id: str = ""
    mode_label: str = "召回+精排+重排"
    top_k: str = "10"
    history: list[dict[str, str]] = []
    recall_items: list[dict[str, str]] = []
    final_items: list[dict[str, str]] = []
    timings: str = ""
    cold_start: bool = False
    error: str = ""
    loading: bool = False

    # ---------- 页面事件 ----------
    async def init(self):
        try:
            data = await api_client.get("/users/sample")
        except httpx.HTTPError as exc:
            self.error = f"无法连接推荐服务：{exc}"
            return
        self.user_options = data["users"]
        self.user_id = self.user_options[0] if self.user_options else ""
        return State.load_all

    def pick_user(self, value: str):
        self.user_id = value
        return State.load_all

    def random_user(self):
        self.user_id = random.choice(self.user_options)
        return State.load_all

    def pick_mode(self, label: str):
        self.mode_label = label
        return State.load_all

    def pick_top_k(self, value: str):
        self.top_k = value
        return State.load_all

    async def click_item(self, item_id: str):
        """模拟点击：写入行为后重新推荐，观察推荐结果如何变化。"""
        try:
            await api_client.post("/feedback", {"user_id": self.user_id, "item_id": item_id})
        except httpx.HTTPError as exc:
            self.error = f"反馈失败：{exc}"
            return
        return State.load_all

    # ---------- 核心：并排拉取 召回-only 与 完整链路 ----------
    async def load_all(self):
        self.loading, self.error = True, ""
        yield
        try:
            k, mode = int(self.top_k), MODE_LABELS[self.mode_label]
            hist = await api_client.get(f"/users/{self.user_id}/history", n=8)
            recall = await api_client.get("/recommend", user_id=self.user_id, top_k=k, mode="recall")
            final = await api_client.get("/recommend", user_id=self.user_id, top_k=k, mode=mode, explain=True)
        except httpx.HTTPError as exc:
            self.error = f"请求失败：{exc}"
            self.loading = False
            return
        self.history = [to_history(i) for i in hist["items"]]
        self.recall_items = [to_card(i, show_change=False) for i in recall["items"]]
        self.final_items = [to_card(i, show_change=True) for i in final["items"]]
        self.cold_start = final["cold_start"]
        t = final["timings"]
        self.timings = " | ".join(f"{k.replace('_ms', '')} {v}ms" for k, v in t.items())
        self.loading = False
