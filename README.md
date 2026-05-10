# Wallpaper Organizer v1.1.0

AI-powered wallpaper sorter using CLIP zero-shot image classification. 
Describe categories in plain English, and the app sorts your wallpaper 
folder automatically.

## Download
Grab `WallpaperOrganizer.zip`in releases. Unzip anywhere, run 
`WallpaperOrganizer.exe`. No installation needed.

## First launch
- **Windows SmartScreen warning:** click **More info** → **Run anyway**. 
  Standard for any unsigned app.
- The app downloads ~600 MB of CLIP model weights from HuggingFace on 
  first run. One-time, then it's cached.
- Launch takes 10–30 seconds the first time, faster after.

## How to use
Hit the **Help / Tips** button inside the app for a full guide. Short version:

1. Pick your wallpaper folder as the source
2. Pick a destination folder
3. Edit the categories — each has a folder name and a plain-English description
4. Keep **Dry run** ticked, click Organize, check the log
5. When the distribution looks right, untick Dry run and run for real

## Requirements
- Windows 10 or 11
- ~500 MB free disk space + ~600 MB for model cache on first run
- Internet connection for first launch only

## Tips
- Better category descriptions = better sorting. Be specific.
- The log shows top-2 confidence scores, so close calls are obvious.
- Threshold ~0.10 for 3 categories, ~0.15 for 10 categories.
