@echo off
setlocal
cd /d %~dp0
echo Starting XLance-MSR App with uv...
uv run python app.py
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Application failed to start. Try running install.bat first.
    pause
)
pause
