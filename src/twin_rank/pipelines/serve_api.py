"""步骤 8：启动 FastAPI 推荐服务（--port 可覆盖配置端口）。
"""

import argparse
import os

import uvicorn

from twin_rank.config import load_config


def main():
    parser = argparse.ArgumentParser(description="启动 Twin-Rank 推荐 API")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    args = parser.parse_args()

    os.environ["TWIN_RANK_CONFIG"] = args.config          # 供 create_app 读取
    cfg, _ = load_config(args.config)
    uvicorn.run("twin_rank.serving.api:create_app", factory=True,
                host=args.host or cfg.serving.host, port=args.port or cfg.serving.port)


if __name__ == "__main__":
    main()
