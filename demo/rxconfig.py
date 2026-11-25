import reflex as rx

config = rx.Config(
    app_name="twin_rank_demo",
    frontend_port=3000,
    backend_port=8001,      # 推荐 API 用 8100，避免与 Reflex 默认的 8000 冲突
)
