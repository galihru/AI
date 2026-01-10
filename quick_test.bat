@echo off
REM Quick test AI Arduino Generator
echo ====================================
echo AI Arduino Generator - Quick Test
echo ====================================
echo.

set PYTHON=C:\Users\asus\public\.venv\Scripts\python.exe
set SCRIPT_DIR=%~dp0scripts

echo [1/3] Checking model...
if not exist models\latest\model.pt (
    echo ERROR: Model not found! Train the model first.
    pause
    exit /b 1
)

echo [2/3] Creating test spec...
echo Create a simple LED blink program. > specs\quick_test.txt
echo Use pin 13. >> specs\quick_test.txt
echo Blink every 500ms. >> specs\quick_test.txt

echo [3/3] Generating code...
%PYTHON% %SCRIPT_DIR%\generate_project.py ^
    --model models\latest ^
    --tokenizer data\tokenizer ^
    --index data\index.json ^
    --spec specs\quick_test.txt ^
    --out out\QuickTest

echo.
echo ====================================
echo DONE! Check out\QuickTest\generated.ino
echo ====================================
pause
