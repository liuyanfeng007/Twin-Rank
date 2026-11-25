import reflex as rx

from .components import controls, history_panel, recommendation_column
from .state import State


def index() -> rx.Component:
    return rx.container(
        rx.vstack(
            rx.heading("Twin-Rank 推荐系统 Demo", size="8"),
            rx.text("双塔向量召回  →  LightGBM 精排  →  MMR 重排与业务规则", color_scheme="gray"),
            controls(),
            rx.cond(State.error != "", rx.callout(State.error, color_scheme="red", width="100%")),
            rx.cond(State.cold_start,
                    rx.callout("该用户无历史行为，已走热门冷启动策略", color_scheme="amber", width="100%")),
            history_panel(),
            rx.grid(
                recommendation_column("① 仅召回 Top-K", "双塔向量检索的原始排序",
                                      State.recall_items, clickable=False),
                recommendation_column("② 最终推荐", "精排 / 重排后的结果，含位次变化与推荐理由",
                                      State.final_items, clickable=True),
                columns="2", spacing="5", width="100%",
            ),
            rx.text("耗时：", State.timings, size="1", color_scheme="gray"),
            spacing="4", padding_y="2em", align="start",
        ),
        size="4",
    )


app = rx.App(theme=rx.theme(accent_color="indigo"))
app.add_page(index, title="Twin-Rank Demo", on_load=State.init)
