@echo off
chcp 65001 >nul
echo ========================================
echo    Git 推送脚本
echo    目标仓库: https://github.com/blues-kun/job-agent.git
echo ========================================
echo.

REM 检查 Git 是否安装
where git >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 未检测到 Git，请先安装 Git：
    echo https://git-scm.com/download/win
    echo.
    pause
    exit /b 1
)

echo [1/7] 初始化 Git 仓库...
git init
if %errorlevel% neq 0 (
    echo [提示] 仓库可能已初始化，继续下一步...
)

echo.
echo [2/7] 配置远程仓库...
git remote add origin https://github.com/blues-kun/job-agent.git 2>nul
if %errorlevel% neq 0 (
    echo [提示] 远程仓库已存在，更新 URL...
    git remote set-url origin https://github.com/blues-kun/job-agent.git
)

echo.
echo [3/7] 检查 .env 文件是否被忽略...
if exist .env (
    echo [警告] 发现 .env 文件，确认已在 .gitignore 中...
    findstr /C:".env" .gitignore >nul
    if %errorlevel% neq 0 (
        echo [错误] .env 未在 .gitignore 中！请手动检查！
        pause
        exit /b 1
    ) else (
        echo [安全] .env 已被 .gitignore 忽略
    )
) else (
    echo [安全] 未发现 .env 文件
)

echo.
echo [4/7] 添加所有文件...
git add .

echo.
echo [5/7] 查看待提交文件...
echo ----------------------------------------
git status
echo ----------------------------------------
echo.
echo [重要] 请检查上面的文件列表，确认没有敏感信息！
echo 如发现 .env 或个人简历文件，请按 Ctrl+C 中止！
echo.
pause

echo.
echo [6/7] 提交更改...
git commit -m "Initial commit: AI Job Recommendation System with XGBoost and LLM"

echo.
echo [7/7] 推送到 GitHub...
git branch -M main
git pull origin main --allow-unrelated-histories --no-edit 2>nul
git push -u origin main

echo.
echo ========================================
echo    推送完成！
echo    访问: https://github.com/blues-kun/job-agent
echo ========================================
echo.
pause

