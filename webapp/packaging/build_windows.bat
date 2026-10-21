@echo off
REM ===========================================================================
REM  PPG 运动伪影去除 · Web 工作台 —— Windows 一键打包
REM  用法：双击本文件（需要本机已装 Python 3.10+；只是"构建"需要，成品 exe 不需要）
REM  产物：dist\PPG-Workbench\PPG-Workbench.exe（onedir，双击即用）
REM        dist\PPG-Workbench-Windows.zip（可直接发给别人）
REM  变体：set BUILD_MODE=onefile  再运行 → 单个 exe（启动慢，体积大）
REM        set BUNDLE_FULL=1      再运行 → 额外打包 full 预设（+109 MB）
REM ===========================================================================
setlocal enabledelayedexpansion
chcp 65001 >nul
cd /d "%~dp0\..\.."

if "%BUILD_MODE%"=="" set BUILD_MODE=onedir
if "%BUNDLE_FULL%"=="" set BUNDLE_FULL=0

echo ============================================================
echo   打包 PPG Web 工作台   BUILD_MODE=%BUILD_MODE%  BUNDLE_FULL=%BUNDLE_FULL%
echo ============================================================

where python >nul 2>nul
if errorlevel 1 (
  echo [错误] 未找到 python。请先安装 Python 3.10+ 并勾选 "Add to PATH"。
  pause & exit /b 1
)

echo [1/5] 创建构建环境 .venv-build ...
if not exist .venv-build (
  python -m venv .venv-build || goto :err
)
call .venv-build\Scripts\activate.bat

echo [2/5] 升级 pip ...
python -m pip install -U pip -q || goto :err

echo [3/5] 安装依赖（CPU 版 torch，避免把 CUDA 打进包里，约 3-5 分钟）...
pip install -q numpy scipy matplotlib flask pyinstaller || goto :err
pip install -q torch --index-url https://download.pytorch.org/whl/cpu || goto :err

echo [4/5] PyInstaller 打包（约 3-10 分钟）...
pyinstaller --noconfirm --clean webapp\packaging\ppg_workbench.spec || goto :err

echo [5/5] 压缩分发包 ...
if /i "%BUILD_MODE%"=="onefile" (
  powershell -NoProfile -Command "Compress-Archive -Force -Path 'dist\PPG-Workbench.exe' -DestinationPath 'dist\PPG-Workbench-Windows.zip'"
) else (
  powershell -NoProfile -Command "Compress-Archive -Force -Path 'dist\PPG-Workbench\*' -DestinationPath 'dist\PPG-Workbench-Windows.zip'"
)

echo.
echo ============================================================
echo   完成！
echo   可运行：dist\PPG-Workbench\PPG-Workbench.exe   （或解压 zip 后双击）
echo   分发用：dist\PPG-Workbench-Windows.zip
echo ============================================================
pause
exit /b 0

:err
echo.
echo [失败] 上一步出错，请把错误信息截图反馈。
pause
exit /b 1
