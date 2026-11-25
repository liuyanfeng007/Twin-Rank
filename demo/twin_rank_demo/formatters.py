"""把 API 返回的商品转成前端展示用的字符串字典（Reflex 里全用 str 最省事）。"""

MODE_LABELS = {"召回+精排": "rank", "召回+精排+重排": "full"}


def to_card(item: dict, show_change: bool) -> dict:
    meta = " · ".join(p for p in [item["store"], f"${item['price']}" if item["price"] else ""] if p)
    scores = []
    if item["recall_score"] is not None:
        scores.append(f"召回分 {item['recall_score']:.3f}")
    if item["rank_score"] is not None:
        scores.append(f"精排分 {item['rank_score']:.3f}")

    change, color = "", "gray"
    if show_change and item["recall_rank"]:
        delta = item["recall_rank"] - item["final_rank"]
        arrow = "▲" if delta > 0 else "▼" if delta < 0 else "＝"
        change = f"召回 #{item['recall_rank']} → 最终 #{item['final_rank']}  {arrow}{abs(delta) or ''}"
        color = "green" if delta > 0 else "red" if delta < 0 else "gray"

    return {
        "item_id": item["item_id"],
        "rank": str(item["final_rank"]),
        "title": item["title"],
        "meta": meta or "—",
        "scores": " | ".join(scores),
        "reason": "；".join(item["reasons"]),
        "change": change,
        "color": color,
    }


def to_history(item: dict) -> dict:
    return {"title": item["title"], "meta": item["store"] or "—"}
