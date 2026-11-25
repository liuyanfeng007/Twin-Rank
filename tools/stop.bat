@echo off
rem 停止 Twin-Rank 的 API 进程 (Reflex 在 start.bat 窗口里按 Ctrl+C 即可)
cd /d "%~dp0.."
if not exist .api.pid exit /b 0
set /p API_PID=<.api.pid
taskkill /PID %API_PID% /T /F >nul 2>&1
del .api.pid >nul 2>&1
echo API 已停止
exit /b 0
