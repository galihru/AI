@echo off
REM Auto Training Launcher - ArduScratch AI
REM Trains AI continuously until human-level performance

echo ========================================
echo   ARDUSCRATCH AI - AUTO TRAINING
echo ========================================
echo.
echo This will train your AI continuously until:
echo   - Loss reaches 0.5 (human-level)
echo   - Or maximum 100 million steps
echo.
echo Training will checkpoint every 500 steps
echo Validation every 2000 steps
echo.
echo Estimated time: Several days to weeks
echo (depending on your hardware)
echo.
echo Press Ctrl+C anytime to stop safely
echo ========================================
echo.

pause

set PYTHON=C:\Users\asus\public\.venv\Scripts\python.exe
set SCRIPT_DIR=%~dp0scripts

echo.
echo Starting autonomous training...
echo.

REM Start training
%PYTHON% %SCRIPT_DIR%\auto_train.py ^
    --tokenizer data\tokenizer ^
    --data data\dataset.bin ^
    --model models\latest ^
    --logs logs ^
    --target-loss 0.5 ^
    --max-steps 100000000 ^
    --checkpoint-interval 500 ^
    --validation-interval 2000 ^
    --batch-size 4 ^
    --lr 0.0003

echo.
echo ========================================
echo Training completed or stopped!
echo ========================================
pause
