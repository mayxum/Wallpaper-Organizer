# Wallpaper Organizer

Auto-sorts a folder of wallpapers into category subfolders using **CLIP zero-shot image classification**. You define categories with plain-English prompts; CLIP figures out which images belong where. No training, no fixed taxonomy — change the categories whenever your taste shifts.

Built with [customtkinter](https://github.com/TomSchimansky/CustomTkinter) for a modern dark-mode UI. Runs locally — no images leave your machine.

## Setup

```bash
pip install torch transformers pillow customtkinter
```

The first run downloads the CLIP model (~600 MB). After that it's cached.

## Run

```bash
python wallpaper_organizer.py
```

Or grab a prebuilt Windows release from the [Releases page](../../releases) — no Python install needed.

## How it works

The app uses a **classify → review → apply** workflow:

1. Pick a source folder, a destination folder, and your categories.
2. Click **Organize**. The app classifies every source image against your category prompts. **Files don't move yet** — it builds an in-memory plan.
3. **Review the results.** Switch to the Gallery view to see thumbnails grouped by category. Right-click any thumbnail and pick "Move to → [Category]" if CLIP got it wrong.
4. Click **Apply Changes**. Files get copied (or moved) into the destination based on your reviewed plan.
5. A manifest file is written to the destination. On re-runs, already-organized images are skipped automatically — no duplicates.

CLIP scores every image against every category prompt and picks the best match. The quality of the prompts matters more than the folder names — a prompt like `"a wide-angle landscape photograph of mountains, oceans, or open vistas"` works better than just `"landscapes"`.

## Default categories

| Folder | Prompt |
|---|---|
| Landscapes | scenic landscapes, mountains, valleys, beaches |
| Nature | close-up plants, flowers, forests, wildlife |
| Cities | cityscapes, skylines, urban architecture |
| Space | stars, galaxies, planets, nebulae |
| Anime | anime/manga style illustrated character art |
| Portraits | portrait photos of real people |
| Cars | cars, motorcycles, vehicles |
| Abstract | abstract art, patterns, digital designs |
| Minimalist | simple shapes, gradients, solid colors |
| Gaming | game screenshots and game artwork |

Add, edit, or remove any of them via the GUI. Double-click a row to edit.

## Features

- **Two views** of the same data:
  - **Feed** — chronological stream of card-style entries with confidence dots, runner-up scores, and category badges. Best for scanning close calls.
  - **Gallery** — collapsible category sections with thumbnail grids. Best for visual spot-checking and bulk review.
- **Click any card or thumbnail** to preview the full-size image with its CLIP scores.
- **Right-click a thumbnail** to move it to a different category. The plan updates; nothing moves on disk until Apply.
- **Live counter chips** showing running totals per category.
- **Filter dropdown** to focus on one category at a time.
- **Status bar with ETA** during classification.
- **Manifest-based skipping** — re-runs only classify new images, not stuff already organized.
- **Dark/light mode toggle** in the top-right.
- **Window size and position** persist between launches.

## Tips

- **Copy mode is on by default.** Files are duplicated into category folders, originals stay put. Switch to move once you trust the setup — copy is reversible, move is annoying to undo.
- **Confidence threshold** — anything scoring below the threshold goes to an `Unsorted` folder so it doesn't get crammed into a wrong category. Default 0.15 works for ~10 categories. Raise it if unrelated images are sneaking into wrong folders; lower it if too much ends up in `Unsorted`. Rule of thumb: ~1/N for N categories, but for 3-category setups try 0.10.
- **CLIP responds well to descriptive language.** If a category misfires, rewrite the prompt to be more specific. Example: if "Cities" is grabbing interior photos, change it to `"an outdoor cityscape with a skyline and tall buildings, taken from outside"`. The more it sounds like an image caption, the better.
- **Contrastive language helps similar categories.** To separate anime from photo portraits: `"an anime ILLUSTRATED character"` vs `"a PHOTOGRAPH of a real person"`.
- **GPU** kicks in automatically if PyTorch sees CUDA. CPU works too — batch processing keeps it fast (~0.1-0.3 sec per image after model load).
- **Subfolders** are walked recursively.
- **Settings persist** in `~/.wallpaper_organizer.json` — your folders, categories, threshold, and window geometry come back next launch.
- **Destination manifest** lives at `<destination>/.wallpaper_organizer.json` — delete it to force re-classification of everything.

## Tech stack

- [CLIP](https://github.com/openai/CLIP) (via HuggingFace `transformers`) for zero-shot image classification
- [customtkinter](https://github.com/TomSchimansky/CustomTkinter) for the UI
- [Pillow](https://pillow.readthedocs.io/) for image loading and thumbnail generation
- [PyTorch](https://pytorch.org/) as the inference backend

## Building a Windows .exe

See [BUILDING.md](BUILDING.md) for the PyInstaller build process and distribution tips.
