@echo off
setlocal
cd /d "%~dp0.."
set "KMP_DUPLICATE_LIB_OK=TRUE"
set "TOKENIZERS_PARALLELISM=false"
python scripts\ai_full_review_prelabel.py --clean --render-previews --device cuda --batch-size 3 %*
endlocal
