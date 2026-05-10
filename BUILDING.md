# Building & Distributing Wallpaper Organizer

This guide covers turning the Python script into a shareable Windows app your friends can download and run without installing anything.

## What you'll end up with

A `WallpaperOrganizer.zip` (~250–400 MB). Friends unzip it, double-click `WallpaperOrganizer.exe` inside, done. They don't need Python, don't need to install anything.

## Prerequisites

- The same Windows machine where the app already runs
- Python 3.x with `torch`, `transformers`, `pillow`, and `customtkinter` already installed
- ~3 GB free disk space during the build (the final zip is much smaller)

If you're missing customtkinter (added in v1.1.0 for the modern UI):
```
python -m pip install customtkinter
```
`build.bat` also installs it automatically as part of step 1, so you can skip that if you're running the build script.

## Build it

The easy way:

1. Make sure these three files are in the same folder:
   - `wallpaper_organizer.py`
   - `wallpaper_organizer.spec`
   - `build.bat`
2. Double-click `build.bat`.
3. Wait 2–5 minutes.

When it finishes, you'll have:

- `dist\WallpaperOrganizer\` — the app folder (~500 MB unzipped)
- `dist\WallpaperOrganizer.zip` — zipped for sending (~250–400 MB)

## Test before sharing

Inside `dist\WallpaperOrganizer\`, double-click `WallpaperOrganizer.exe`. If the GUI opens and a small classification works, it's good. First launch on the build machine still hits the cached CLIP weights (since you've already used the app), so this won't catch the "first download" path your friends will see.

To test the *fresh* experience: copy the zip to a different Windows machine (or a VM), unzip, run. First launch there will:

1. Take 30–60 seconds to start (Windows scanning the bundle)
2. Download ~600 MB of CLIP model weights from HuggingFace once
3. Open the GUI normally

Subsequent launches are fast (~5 seconds).

## Sharing with friends

The zip is too big for email, Discord, or most chat apps. Pick one of:

**GitHub Releases** (free, professional, 2 GB max per file)
1. Create a public GitHub repo for the project
2. Push the source files
3. On the repo page → Releases → Draft a new release
4. Drag the zip into the assets area
5. Publish — you get a permanent download link

**MEGA** (free, 20 GB account, downloaders don't need an account)
- Drag-drop the zip → share link

**Google Drive / Dropbox**
- Upload, right-click → Share → "Anyone with the link"
- Note: Drive sometimes scans large zips slowly before allowing download

## Telling your friends what to do

Send them this:

> Download `WallpaperOrganizer.zip`, unzip it anywhere, run `WallpaperOrganizer.exe` inside.
>
> First launch:
> - Windows will say "Windows protected your PC." Click **More info** → **Run anyway**. (This warning happens with any unsigned app — the app is fine.)
> - It'll take a minute and download ~600 MB of model weights from the internet. This only happens once.
>
> After that, double-click `WallpaperOrganizer.exe` whenever you want.

## The Windows SmartScreen warning

Unsigned executables always trigger Windows' "are you sure?" prompt. To make it go away permanently you'd need a code-signing certificate (~$200–500/year). For sharing with friends, just tell them to click **More info → Run anyway**.

If a friend's antivirus deletes the exe outright, that's a false positive — PyInstaller bundles trip some heuristics. They can either add an exclusion or scan it on [virustotal.com](https://www.virustotal.com) first to verify.

## Troubleshooting builds

**"Failed to execute script wallpaper_organizer"**
A hidden import wasn't detected. Run the exe from `cmd` to see the actual error, then add the missing module to `hiddenimports` in `wallpaper_organizer.spec` and rebuild.

**Build is huge (>1.5 GB)**
You probably have CUDA torch. Check with `pip show torch` — the version line shouldn't mention CUDA. If it does and you don't actually have an NVIDIA GPU, reinstall with the CPU wheel:
```
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**Build crashes during PyInstaller analysis**
Check the PyInstaller version: `python -m pip show pyinstaller`. You need 6.19+ for Python 3.14. Update with `python -m pip install --upgrade pyinstaller`.

**Friend's machine: "VCRUNTIME140.dll missing" or similar**
Their Windows is missing the Visual C++ runtime. Have them install the [Microsoft Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe). One-time install.

**Friend's machine: app launches then closes immediately**
Usually missing model weights download (no internet) or write permission issue. Have them run from `cmd` to see the error.

**Build error: "ModuleNotFoundError: No module named 'customtkinter'"**
Install it: `python -m pip install customtkinter`. The build script does this automatically — this error means you're running PyInstaller manually without running step 1.

**Bundled exe crashes with "FileNotFoundError" mentioning a `.json` theme or `.otf` font**
customtkinter's theme files didn't get bundled. Make sure your spec has the `collect_all('customtkinter')` block at the top and that `ctk_datas`, `ctk_binaries`, and `ctk_hiddenimports` are wired into `Analysis(...)`. Re-run `build.bat`.

## Updating

When you change the script and want to ship a new version:

1. Edit `wallpaper_organizer.py`
2. Run `build.bat` again — overwrites the previous build
3. Send the new zip

Friends can replace their existing folder. Their settings (in `~/.wallpaper_organizer.json`) will carry over.

## Cross-platform notes

PyInstaller doesn't cross-compile. To build for Mac or Linux you'd run the same `build.bat` equivalent on a Mac/Linux machine. If a friend asks for a Mac version and you don't have a Mac, tell them to run the script directly with Python — it works on all platforms identically.
