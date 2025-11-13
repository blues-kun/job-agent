@echo off
chcp 65001
echo ========================================
echo    推送到 GitHub
echo    https://github.com/blues-kun/job-agent
echo ========================================
echo.

cd /d "%~dp0"

echo [1/8] 检查 Git...
git --version
if errorlevel 1 (
    echo [错误] Git 不可用！
    pause
    exit /b 1
)

echo.
echo [2/8] 初始化 Git 仓库...
git init

echo.
echo [3/8] 配置远程仓库...
git remote remove origin 2>nul
git remote add origin https://github.com/blues-kun/job-agent.git

echo.
echo [4/8] 添加所有文件...
git add .

echo.
echo [5/8] 查看待提交文件（确认没有敏感信息）...
git status
echo.
echo 请检查上面的文件列表！
echo 如发现敏感信息，请按 Ctrl+C 中止！
pause

echo.
echo [6/8] 提交更改...
git commit -m "Initial commit: AI Job Recommendation System with XGBoost and LLM"

echo.
echo [7/8] 设置主分支...
git branch -M main

echo.
echo [8/8] 推送到 GitHub...
echo 可能需要输入 GitHub 用户名和密码/token
git push -u origin main --force

echo.
echo ========================================
echo    完成！
echo    访问: https://github.com/blues-kun/job-agent
echo ========================================
pause

