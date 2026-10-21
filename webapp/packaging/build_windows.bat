@echo off
REM ===========================================================================
REM  PPG 运动伪影去除 · Web 工作台 —— Windows 一键打包（**单文件 exe + 无控制台**）
REM  产物：dist\PPG-Workbench-v1.exe 与 dist\PPG-Workbench-v2.exe（各自单文件）
REM  用法：双击本文件（本机只需装 Python 3.10+ 用于构建；成品 exe 不需要 Python）
REM  变体：set BUNDLE_FULL=1 再运行 → 额外打包 full 预设（+109 MB）
REM ===========================================================================
setlocal enabledelayedexpansion
chcp 65001 >nul
cd /d "%~dp0\..\.."

if "%BUNDLE_FULL%"=="" set BUNDLE_FULL=0

echo ============================================================
echo   打包 PPG Web 工作台（onefile · 无控制台）  BUNDLE_FULL=%BUNDLE_FULL%
echo ============================================================

where python >nul 2>nul
if errorlevel 1 (
  echo [错误] 未找到 python。请先安装 Python 3.10+ 并勾选 "Add to PATH"。
  pause & exit /b 1
)

echo [1/4] 创建构建环境 .venv-build ...
if not exist .venv-build (
  python -m venv .venv-build || goto :err
)
call .venv-build\Scripts\activate.bat

echo [2/4] 安装依赖（CPU 版 torch，避免把 CUDA 打进包里，约 3-5 分钟）...
python -m pip install -U pip -q || goto :err
pip install -q numpy scipy matplotlib flask pyinstaller || goto :err
pip install -q torch --index-url https://download.pytorch.org/whl/cpu || goto :err

echo [3/4] 打包 v1 与 v2（每个约 3-10 分钟；单文件、无控制台）...
set VARIANT=v1
pyinstaller --noconfirm --clean webapp\packaging\ppg_workbench_onefile.spec || goto :err
set VARIANT=v2
pyinstaller --noconfirm --clean webapp\packaging\ppg_workbench_onefile.spec || goto :err

echo [4/4] 完成
echo.
echo ============================================================
echo   dist\PPG-Workbench-v1.exe   原始版（演示为合成正弦、显示 PRD）
echo   dist\PPG-Workbench-v2.exe   增强版（真实形态演示 + 演示态链路自检）
echo   单文件可直接拷贝；双击后会自动打开浏览器，右上角 ⏻ 退出应用
echo ============================================================
pause
exit /b 0

:err
echo.
echo [失败] 上一步出错，请把错误信息截图反馈。
pause
exit /b 1
