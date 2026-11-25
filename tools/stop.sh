#!/usr/bin/env bash
# 停止 Twin-Rank 的 API 与 Reflex 进程
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [ -f .api.pid ]; then kill "$(cat .api.pid)" 2>/dev/null && echo "API 已停止"; rm -f .api.pid; fi
pkill -f "reflex run" 2>/dev/null && echo "Reflex 已停止" || true
