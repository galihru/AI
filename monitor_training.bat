@echo off
REM Monitor Training Progress

set PYTHON=C:\Users\asus\public\.venv\Scripts\python.exe
set SCRIPT_DIR=%~dp0scripts

:menu
cls
echo ========================================
echo   TRAINING MONITOR
echo ========================================
echo.
echo 1. Show current status
echo 2. Watch (auto-refresh every 30s)
echo 3. Generate loss plot
echo 4. Exit
echo.
set /p choice="Choose option: "

if "%choice%"=="1" (
    %PYTHON% %SCRIPT_DIR%\monitor.py
    pause
    goto menu
)

if "%choice%"=="2" (
    %PYTHON% %SCRIPT_DIR%\monitor.py watch 30
    goto menu
)

if "%choice%"=="3" (
    %PYTHON% %SCRIPT_DIR%\monitor.py plot
    echo.
    echo Plot saved as training_plot.png
    pause
    goto menu
)

if "%choice%"=="4" exit

goto menu
