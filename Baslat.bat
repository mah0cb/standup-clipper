@echo off
rem Standup Clipper web panelini baslatir ve tarayicida acar (http://127.0.0.1:8770/)
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python -m clipper.web --open
pause
