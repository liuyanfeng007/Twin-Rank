@echo off
rem Twin-Rank 一键启动 (Windows)
rem 用法: tools\start.bat [/retrain]
rem   首次运行: 建虚拟环境 -> 装依赖 -> 环境自检 -> 跑完整流水线(含注册模型) -> 启动 API -> 启动 Reflex
rem   再次运行: 检测到已有模型则直接启动服务; 加 /retrain 强制重新训练
setlocal enabledelayedexpansion
chcp 65001 >nul
cd /d "%~dp0.."

set "FORCE_PIPELINE=0"
if /i "%~1"=="/retrain" set "FORCE_PIPELINE=1"
if exist .env (
  for /f "usebackq eol=# tokens=1,* delims==" %%a in (".env") do set "%%a=%%b"
)
if not defined API_PORT set "API_PORT=8100"
if not exist logs mkdir logs

echo ==^> [1/5] 准备 Python 环境
if not exist .venv\Scripts\python.exe (
  py -3 -m venv .venv 2>nul || python -m venv .venv
  if errorlevel 1 goto :fail
)
call .venv\Scripts\activate.bat
if not exist .venv\.twin_rank_installed (
  echo 安装依赖: 首次安装需下载 PyTorch 等大文件, 请耐心等待
  echo   国内网络慢可设置镜像: set PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
  python -m pip install --upgrade pip
  if defined TORCH_INDEX_URL pip install torch --index-url %TORCH_INDEX_URL%
  pip install -e ".[demo]"
  if errorlevel 1 goto :fail
  echo done> .venv\.twin_rank_installed
)

echo ==^> 环境自检
python tools\doctor.py
if errorlevel 1 goto :fail

echo ==^> [2/5] 训练流水线
set "NEED_PIPELINE=0"
if "%FORCE_PIPELINE%"=="1" set "NEED_PIPELINE=1"
if not exist data\artifacts\*ranker* set "NEED_PIPELINE=1"
if "%SKIP_PIPELINE%"=="1" set "NEED_PIPELINE=0"
if "%NEED_PIPELINE%"=="1" (
  for %%s in (download_data preprocess_data build_features train_two_tower build_index train_ranker register_model) do (
    echo ---- %%s
    python -m twin_rank.pipelines.%%s
    if errorlevel 1 goto :fail
  )
) else (
  echo 已有模型产物, 跳过 ^(/retrain 可强制重训^)
)

echo ==^> [3/5] 启动推荐 API ^(端口 %API_PORT%^)
powershell -NoProfile -Command "$p = Start-Process -FilePath '.venv\Scripts\python.exe' -ArgumentList '-m','twin_rank.pipelines.serve_api','--port','%API_PORT%' -RedirectStandardOutput 'logs\api.log' -RedirectStandardError 'logs\api.err.log' -WindowStyle Hidden -PassThru; $p.Id | Out-File -Encoding ascii .api.pid"

echo ==^> [4/5] 等待 API 就绪
for /l %%i in (1,1,90) do (
  python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:%API_PORT%/health', timeout=2)" >nul 2>&1
  if not errorlevel 1 goto :api_ok
  timeout /t 2 /nobreak >nul
)
echo API 启动超时, 请查看 logs\api.log 与 logs\api.err.log
goto :fail

:api_ok
echo API 已就绪
echo ==^> [5/5] 启动 Reflex 前端  http://localhost:3000  ^(Ctrl+C 退出^)
cd demo
set "TWIN_RANK_API_URL=http://127.0.0.1:%API_PORT%"
reflex run
cd ..
call tools\stop.bat
exit /b 0

:fail
echo.
echo 启动失败, 请检查上面的错误信息
call tools\stop.bat >nul 2>&1
pause
exit /b 1
