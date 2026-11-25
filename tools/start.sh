#!/usr/bin/env bash
# Twin-Rank 一键启动（macOS / Linux）
# 用法: tools/start.sh [--retrain]
#   首次运行：建虚拟环境 -> 装依赖 -> 环境自检 -> 跑完整流水线(含注册模型) -> 启动 API -> 启动 Reflex
#   再次运行：检测到已有模型则直接启动服务；加 --retrain 强制重新训练
# 可用环境变量（也可写进项目根目录的 .env）:
#   PYTHON=python3.11  API_PORT=8100  SKIP_PIPELINE=1  MLFLOW_TRACKING_URI=http://127.0.0.1:5000
#   TWIN_RANK_MODEL_URI=models:/twin-rank-recommender@champion   # 让 API 从 MLflow 注册表加载模型
set -euo pipefail

[ "${DEBUG:-0}" = "1" ] && set -x      # DEBUG=1 tools/start.sh 可看到每一步执行的命令
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
[ -f .env ] && { set -a; . ./.env; set +a; }

PYTHON="${PYTHON:-python3}"
API_PORT="${API_PORT:-8100}"
FORCE_PIPELINE=0
[ "${1:-}" = "--retrain" ] && FORCE_PIPELINE=1
mkdir -p logs

echo "==> [1/5] 准备 Python 环境 ($(uname -s) $(uname -m))"
if [ ! -d .venv ]; then
  "$PYTHON" -m venv .venv || {
    echo "创建虚拟环境失败。Debian/Ubuntu 请先: sudo apt-get install -y python3-venv python3-pip"; exit 1; }
fi
# shellcheck disable=SC1091
source .venv/bin/activate
STAMP=.venv/.twin_rank_installed
if [ ! -f "$STAMP" ] || [ pyproject.toml -nt "$STAMP" ]; then
  INDEX="${PIP_INDEX_URL:-https://pypi.org/simple}"
  echo "检测软件源 $INDEX ..."
  if python -c "import urllib.request as u; u.urlopen('$INDEX', timeout=8)" 2>/dev/null; then
    echo "  软件源可访问"
  else
    echo "  ⚠ 8 秒内无法连接 $INDEX，后面的安装很可能超时重试。国内请先设置镜像后重新运行:"
    echo "    export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple"
  fi
  echo "安装依赖：首次安装需下载 PyTorch 等大文件（Linux 默认约数 GB），请耐心等待，下面会显示进度"
  echo "  国内网络慢可设置镜像，如: export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple"
  echo "  没有 NVIDIA 显卡可只装 CPU 版（体积小很多）: export TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu"
  export PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT:-100}"
  python -m pip install --upgrade pip
  if [ -n "${TORCH_INDEX_URL:-}" ]; then
    python -m pip install torch --index-url "$TORCH_INDEX_URL"     # 先装指定来源的 torch，后面就不会再装一遍
  fi
  python -m pip install -e ".[demo]"
  touch "$STAMP"
else
  echo "依赖已安装，跳过"
fi

# macOS 上 LightGBM 依赖 OpenMP 运行库 libomp（系统默认不带）
if [ "$(uname)" = "Darwin" ] && [ ! -f /opt/homebrew/opt/libomp/lib/libomp.dylib ] \
   && [ ! -f /usr/local/opt/libomp/lib/libomp.dylib ]; then
  if command -v brew >/dev/null 2>&1; then
    echo "安装 libomp (LightGBM 依赖)"; brew install libomp
  else
    echo "缺少 libomp：请先安装 Homebrew (https://brew.sh)，再执行 brew install libomp"; exit 1
  fi
fi

echo "==> 环境自检"
python tools/doctor.py

echo "==> [2/5] 训练流水线"
if [ "${SKIP_PIPELINE:-0}" != "1" ] && { [ "$FORCE_PIPELINE" = 1 ] || ! ls data/artifacts/*ranker* >/dev/null 2>&1; }; then
  for step in download_data preprocess_data build_features train_two_tower build_index train_ranker register_model; do
    echo "---- $step"
    python -m "twin_rank.pipelines.$step"
  done
else
  echo "已有模型产物，跳过（--retrain 可强制重训）"
fi

echo "==> [3/5] 启动推荐 API (端口 $API_PORT)"
python -m twin_rank.pipelines.serve_api --port "$API_PORT" > logs/api.log 2>&1 &
echo $! > .api.pid
cleanup() { [ -f .api.pid ] && kill "$(cat .api.pid)" 2>/dev/null || true; rm -f .api.pid; }
trap cleanup EXIT INT TERM

echo "==> [4/5] 等待 API 就绪"
for _ in $(seq 1 90); do
  if python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:${API_PORT}/health', timeout=2)" 2>/dev/null; then
    echo "API 已就绪"; break
  fi
  if ! kill -0 "$(cat .api.pid)" 2>/dev/null; then
    echo "API 启动失败，日志: logs/api.log"; tail -n 20 logs/api.log; exit 1
  fi
  sleep 2
done

echo "==> [5/5] 启动 Reflex 前端  http://localhost:3000   (Ctrl+C 退出并关闭 API)"
cd demo
TWIN_RANK_API_URL="http://127.0.0.1:${API_PORT}" reflex run
