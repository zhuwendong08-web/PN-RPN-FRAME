@echo off
rem PF-RPN slim reproduction - one-click pipeline on Windows (pure CPU)
set PY=%~dp0.venv\Scripts\python.exe
if not exist "%PY%" set PY=python

echo [1/4] unit checks ...
"%PY%" tests\test_modules.py
if errorlevel 1 goto :fail

echo [2/4] training (4 epochs on synthetic data) ...
"%PY%" tools\train.py --config configs\pf_rpn_tiny.py --epochs 4
if errorlevel 1 goto :fail

echo [3/4] evaluation ...
"%PY%" tools\test.py --config configs\pf_rpn_tiny.py --checkpoint outputs\best.pth --topk 100,300
if errorlevel 1 goto :fail

echo [4/4] demo inference ...
"%PY%" tools\demo.py --synthesize --out outputs\demo.png

echo.
echo All steps finished. Checkpoints in outputs\, results in outputs\train_log.json
pause
exit /b 0

:fail
echo.
echo FAILED - see the error message above.
pause
exit /b 1
