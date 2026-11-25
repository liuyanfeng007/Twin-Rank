import reflex as rx

from .state import State


def item_card(item, clickable: bool = True) -> rx.Component:
    return rx.card(
        rx.hstack(
            rx.badge(item["rank"], size="3", variant="solid", color_scheme="indigo"),
            rx.vstack(
                rx.text(item["title"], weight="bold", size="2"),
                rx.text(item["meta"], size="1", color_scheme="gray"),
                rx.text(item["scores"], size="1", color_scheme="gray"),
                rx.cond(item["reason"] != "",
                        rx.text("推荐理由：", item["reason"], size="1", color_scheme="blue")),
                rx.cond(item["change"] != "",
                        rx.badge(item["change"], color_scheme=item["color"], variant="soft")),
                align="start", spacing="1",
            ),
            rx.spacer(),
            rx.cond(clickable,
                    rx.button("点击", size="1", variant="soft",
                              on_click=State.click_item(item["item_id"]))),
            width="100%", align="center",
        ),
        width="100%",
    )


def recommendation_column(title: str, subtitle: str, items, clickable: bool) -> rx.Component:
    return rx.vstack(
        rx.heading(title, size="4"),
        rx.text(subtitle, size="1", color_scheme="gray"),
        rx.foreach(items, lambda it: item_card(it, clickable)),
        spacing="2", width="100%", align="start",
    )


def history_panel() -> rx.Component:
    return rx.card(
        rx.vstack(
            rx.heading("用户最近行为", size="3"),
            rx.foreach(State.history, lambda h: rx.text("• ", h["title"], size="2")),
            align="start", spacing="1",
        ),
        width="100%",
    )


def controls() -> rx.Component:
    return rx.hstack(
        rx.select(State.user_options, value=State.user_id, on_change=State.pick_user, width="340px"),
        rx.button("随机用户", on_click=State.random_user, variant="soft"),
        rx.select(["召回+精排", "召回+精排+重排"], value=State.mode_label, on_change=State.pick_mode),
        rx.select(["5", "10", "20"], value=State.top_k, on_change=State.pick_top_k, width="80px"),
        rx.cond(State.loading, rx.spinner()),
        spacing="3", wrap="wrap", align="center",
    )
