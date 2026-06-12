@echo off
setlocal
cd /d "%~dp0"
powershell -ExecutionPolicy Bypass -File "scripts\quickstart.ps1" %*
exit /b %ERRORLEVEL%
