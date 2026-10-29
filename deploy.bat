@echo off
REM deploy.bat -- double-clickable Windows wrapper around deploy.sh.
REM Runs the real deploy script with Git Bash (installed alongside Git for
REM Windows, which this repo already needs for git itself).
setlocal
set SCRIPT_DIR=%~dp0
set GITBASH="%PROGRAMFILES%\Git\bin\bash.exe"
if not exist %GITBASH% set GITBASH="%PROGRAMFILES(X86)%\Git\bin\bash.exe"
if not exist %GITBASH% (
    echo Could not find Git Bash. Install Git for Windows, or run deploy.sh
    echo directly from a shell that has git/bash on PATH.
    pause
    exit /b 1
)
%GITBASH% "%SCRIPT_DIR%deploy.sh" %*
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Deploy failed -- see the output above.
    pause
    exit /b %ERRORLEVEL%
)
pause
