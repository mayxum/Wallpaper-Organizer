@echo off
REM ============================================================
REM Wallpaper Organizer - build script for Windows
REM
REM What this does:
REM   1. Installs/updates PyInstaller
REM   2. Cleans previous build folders
REM   3. Builds dist\WallpaperOrganizer\ (a folder bundle)
REM   4. Zips it to dist\WallpaperOrganizer.zip for distribution
REM
REM Just double-click this file. Takes 2-5 minutes.
REM ============================================================

setlocal
cd /d "%~dp0"

echo.
echo ============================================================
echo   Wallpaper Organizer - Build Script
echo ============================================================
echo.

echo [1/4] Installing/updating PyInstaller...
python -m pip install --upgrade pyinstaller
if errorlevel 1 goto :fail

echo.
echo [2/4] Cleaning previous build artifacts...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist app_icon.ico del /q app_icon.ico

echo.
echo [3/4] Building bundle (this is the slow part - 2 to 5 minutes)...
echo       PyInstaller is collecting torch, transformers, and friends.
echo       Don't worry about warnings unless the build actually fails.
echo.
python -m PyInstaller wallpaper_organizer.spec --noconfirm
if errorlevel 1 goto :fail

echo.
echo [4/4] Zipping bundle for distribution...
powershell -NoProfile -Command "Compress-Archive -Path dist\WallpaperOrganizer -DestinationPath dist\WallpaperOrganizer.zip -Force"
if errorlevel 1 goto :fail

echo.
echo ============================================================
echo   DONE
echo ============================================================
echo.
echo   App folder:  dist\WallpaperOrganizer\
echo   Run locally: dist\WallpaperOrganizer\WallpaperOrganizer.exe
echo.
echo   Send to friends: dist\WallpaperOrganizer.zip
echo.
echo   See BUILDING.md for distribution tips and troubleshooting.
echo.
pause
exit /b 0

:fail
echo.
echo ============================================================
echo   BUILD FAILED
echo ============================================================
echo.
echo Scroll up to see the error. Most common causes:
echo   - Missing dependency: python -m pip install torch transformers pillow
echo   - Python version mismatch: PyInstaller 6.19+ is needed for Python 3.14
echo.
pause
exit /b 1
