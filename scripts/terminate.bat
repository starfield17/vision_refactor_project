@echo off
setlocal
pushd "%~dp0\.." || exit /b 1
powershell -ExecutionPolicy Bypass -File "scripts\terminate.ps1" %*
set EXIT_CODE=%ERRORLEVEL%
popd
exit /b %EXIT_CODE%
