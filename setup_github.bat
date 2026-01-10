@echo off
REM Setup GitHub repository for cloud training
echo ========================================
echo ArduScratch - GitHub Setup for Cloud Training
echo ========================================
echo.

REM Check if git is installed
where git >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Git not found! Please install Git first.
    echo Download from: https://git-scm.com/download/win
    pause
    exit /b 1
)

echo [1/6] Initializing git repository...
git init
git branch -M main

echo.
echo [2/6] Creating .gitignore...
(
echo __pycache__/
echo *.pyc
echo .venv/
echo *.log
echo .DS_Store
) > .gitignore

echo.
echo [3/6] Adding files...
git add .

echo.
echo [4/6] Creating initial commit...
git commit -m "Initial commit: ArduScratch AI with cloud training"

echo.
echo [5/6] Setting up Git LFS for large files...
where git-lfs >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    git lfs install
    git lfs track "*.bin"
    git add .gitattributes
    git commit -m "Setup Git LFS"
) else (
    echo WARNING: Git LFS not installed. Large files may fail to push.
    echo Install from: https://git-lfs.github.com/
)

echo.
echo [6/6] Next steps:
echo.
echo 1. Create a NEW repository on GitHub:
echo    https://github.com/new
echo.
echo 2. Run these commands with YOUR repository URL:
echo.
echo    git remote add origin https://github.com/YOUR_USERNAME/ArduScratch.git
echo    git push -u origin main
echo.
echo 3. Enable GitHub Actions:
echo    - Go to Settings -^> Actions -^> General
echo    - Enable "Read and write permissions"
echo.
echo 4. Start cloud training:
echo    - Go to Actions tab
echo    - Run "Autonomous AI Training" workflow
echo.
echo ========================================
echo Setup complete! Follow the steps above.
echo ========================================
pause
