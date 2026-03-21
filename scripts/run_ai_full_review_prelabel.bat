@echo off
setlocal
cd /d "%~dp0.."
set "CUDA_PYTHON=C:\Users\isYun\anaconda3\envs\grounded_sam\python.exe"
set "KMP_DUPLICATE_LIB_OK=TRUE"
set "TOKENIZERS_PARALLELISM=false"
if exist "%CUDA_PYTHON%" (
  "%CUDA_PYTHON%" scripts\ai_full_review_prelabel.py --device cuda %*
) else (
  python scripts\ai_full_review_prelabel.py %*
)
endlocal
