# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for Wallpaper Organizer.

Build with:
    pyinstaller wallpaper_organizer.spec

Or just double-click build.bat.

Produces dist/WallpaperOrganizer/WallpaperOrganizer.exe (a folder bundle).
We use the folder (onedir) build instead of onefile because onefile
re-extracts the full ~500 MB bundle on every launch — way too slow for a
PyTorch app. The folder bundle is faster to start and zips fine for
distribution.
"""
import os
import sys
from pathlib import Path

# ---- Build a .ico for the exe icon -------------------------------------
# Generated programmatically so the project stays a single .py + spec.
def _make_ico():
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return None
    size = 256
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    # Three offset rounded squares — a stack of sorted folders.
    colors = [(110, 180, 220, 255), (180, 110, 220, 255), (230, 160, 90, 255)]
    for i, color in enumerate(colors):
        o = i * 28
        d.rounded_rectangle(
            [24 + o, 24 + o, 168 + o, 168 + o],
            radius=24, fill=color,
            outline=(35, 35, 40, 255), width=8,
        )
    ico_path = Path(SPECPATH) / "app_icon.ico"
    img.save(ico_path, format="ICO",
             sizes=[(256, 256), (128, 128), (64, 64), (32, 32), (16, 16)])
    return str(ico_path)

_ICON_PATH = _make_ico()


# ---- Analysis ----------------------------------------------------------
a = Analysis(
    ['wallpaper_organizer.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[
        # Pillow/Tkinter integration sometimes isn't auto-detected
        'PIL._tkinter_finder',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # Modules transitive deps drag in but we don't actually use.
    # Excluding them shrinks the bundle without affecting functionality.
    excludes=[
        # Big siblings of torch we don't import
        'torchvision',
        'torchaudio',
        # Visualization / data-science libs sometimes pulled by transitive deps
        'matplotlib',
        'pandas',
        'scipy',
        'sklearn',
        'IPython',
        'jupyter',
        'notebook',
        # Test/dev tooling
        'pytest',
        'unittest',
        # Tensorboard support inside torch (we don't log to it)
        'tensorboard',
        'torch.utils.tensorboard',
        # Distributed training (single-process classification doesn't need it)
        'torch.distributed',
        'torch.testing',
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)


# ---- EXE ---------------------------------------------------------------
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='WallpaperOrganizer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,           # UPX compression is unreliable with PyTorch DLLs
    console=False,       # GUI app, no console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=_ICON_PATH,
)


# ---- COLLECT -----------------------------------------------------------
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='WallpaperOrganizer',
)
