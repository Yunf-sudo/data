@echo off
setlocal
cd /d "%~dp0.."
set "KMP_DUPLICATE_LIB_OK=TRUE"
set "TOKENIZERS_PARALLELISM=false"
set "LOG=%CD%\ai_full_review_run.log"
set "ERR=%CD%\ai_full_review_run.err.log"
if exist "%LOG%" del /f /q "%LOG%"
if exist "%ERR%" del /f /q "%ERR%"
"C:\Users\isYun\anaconda3\envs\grounded_sam\python.exe" scripts\ai_full_review_prelabel.py --clean --render-previews --device cuda --batch-size 3 1>"%LOG%" 2>"%ERR%"
endlocal
