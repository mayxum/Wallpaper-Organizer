#!/usr/bin/env python3
"""
Wallpaper Organizer
-------------------
Auto-sorts a folder of wallpapers into category subfolders using CLIP
zero-shot image classification. You define the categories with text
prompts; CLIP figures out which images belong where.

Setup:
    pip install torch transformers pillow customtkinter

Run:
    python wallpaper_organizer.py
"""
from __future__ import annotations

import hashlib
import json
import shutil
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from queue import Queue, Empty
from typing import Optional

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

try:
    import customtkinter as ctk
except ImportError:
    raise SystemExit(
        "customtkinter is required for the Wallpaper Organizer UI.\n"
        "Install it with:  pip install customtkinter"
    )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff", ".jfif"}

# Default categories: (folder name, CLIP prompt). Edit freely in the GUI.
DEFAULT_CATEGORIES = [
    # (folder_name, clip_prompt, threshold)
    ("Landscapes", "a scenic landscape photograph with mountains, valleys, beaches, or open vistas", 0.15),
    ("Nature",     "a close-up nature photograph of plants, flowers, forests, or wildlife", 0.15),
    ("Cities",     "a cityscape photograph showing buildings, skylines, streets, or urban architecture", 0.15),
    ("Space",      "a photograph of stars, galaxies, planets, nebulae, or outer space", 0.15),
    ("Anime",      "anime or manga style illustrated artwork featuring a character", 0.15),
    ("Portraits",  "a portrait photograph of a real person", 0.15),
    ("Cars",       "a photograph of cars, motorcycles, or other vehicles", 0.15),
    ("Abstract",   "abstract art, geometric patterns, or digital designs", 0.15),
    ("Minimalist", "a minimalist wallpaper with simple shapes, gradients, or solid colors", 0.15),
    ("Gaming",     "a screenshot or artwork from a video game", 0.15),
]

DEFAULT_THRESHOLD = 0.15

CONFIG_FILE = Path.home() / ".wallpaper_organizer.json"

# Accent palette (light mode, dark mode)
ACCENT_COLOR = ("#2563eb", "#3b82f6")
ACCENT_HOVER = ("#1d4ed8", "#2563eb")
GHOST_BORDER = ("gray60", "gray45")
GHOST_HOVER  = ("gray85", "gray25")
GHOST_TEXT   = ("gray20", "gray80")

# Category badge palette. Each tuple: (light_bg, dark_bg, light_text, dark_text).
# Categories cycle through this list in order, so a given setup gets stable
# colors across runs (NSFW always rose-pink, Has_People always amber, etc.
# depending on order). Twelve hand-picked accessible color pairs.
CATEGORY_PALETTE = [
    ("#ccfbf1", "#134e4a", "#0f766e", "#5eead4"),  # teal
    ("#fef3c7", "#78350f", "#b45309", "#fbbf24"),  # amber
    ("#ffe4e6", "#7f1d1d", "#be123c", "#fda4af"),  # rose
    ("#ede9fe", "#4c1d95", "#7c3aed", "#c4b5fd"),  # violet
    ("#dbeafe", "#1e3a8a", "#2563eb", "#93c5fd"),  # sky
    ("#d1fae5", "#064e3b", "#059669", "#6ee7b7"),  # emerald
    ("#ffedd5", "#7c2d12", "#ea580c", "#fdba74"),  # orange
    ("#fce7f3", "#831843", "#db2777", "#f9a8d4"),  # pink
    ("#ecfccb", "#365314", "#65a30d", "#bef264"),  # lime
    ("#cffafe", "#164e63", "#0891b2", "#67e8f9"),  # cyan
    ("#e0e7ff", "#312e81", "#4f46e5", "#a5b4fc"),  # indigo
    ("#fee2e2", "#7f1d1d", "#dc2626", "#fca5a5"),  # red
]
# Reserved color for Unsorted (below-threshold images) — always neutral gray.
UNSORTED_COLOR = ("#e5e7eb", "#374151", "#4b5563", "#d1d5db")

# Confidence dot colors — based on how far the winner beat the runner-up.
# Large margin = clear winner (green), small margin = borderline call (yellow),
# below threshold (Unsorted) = always red.
DOT_HIGH   = "#10b981"  # green
DOT_MEDIUM = "#f59e0b"  # yellow
DOT_LOW    = "#ef4444"  # red


# ---------------------------------------------------------------------------
# Help content - sectioned for the navigable Help dialog
# ---------------------------------------------------------------------------

HELP_SECTIONS: list[dict] = [
    {
        "title": "Quick Start",
        "subtitle": "Classify, review, apply.",
        "blocks": [
            ("steps", [
                "Pick a SOURCE folder — where your wallpapers currently live.",
                "Pick a DESTINATION folder — where the sorted subfolders will go.",
                "Edit the categories. Each has a folder name and a CLIP prompt describing what should land there. Double-click any row to edit. Add creates new categories, Reset brings defaults back.",
                "Click ORGANIZE. The app classifies all source images (no files move yet) and populates the Feed and Gallery views.",
                'Switch to the GALLERY view to review visually. Right-click any thumbnail and pick "Move to → [Category]" if CLIP got something wrong.',
                'When you\'re happy, click APPLY CHANGES. Files are moved/copied based on the plan (with your manual corrections applied) and the destination folder gets a manifest so future runs skip already-organized images.',
            ]),
            ("tip", "On re-runs, source images already organized into the destination are skipped automatically — no duplicates, no re-classification."),
        ],
    },
    {
        "title": "How CLIP Works",
        "subtitle": "Plain-English version.",
        "blocks": [
            ("p", 'CLIP is an AI model that understands both images and text in the same "language." For each image, it scores how well the picture matches every one of your category prompts. Scores add up to 1.00 across all categories. Highest score wins.'),
            ("h", "What this means in practice"),
            ("ul", [
                "The PROMPT matters more than the folder NAME. CLIP doesn't read folder names — only the prompt text.",
                "Better-described prompts = better sorting.",
                'Two visually similar categories will compete for ambiguous images. Contrastive language helps — "an anime ILLUSTRATED character" vs "a PHOTOGRAPH of a real person."',
                "CLIP isn't perfect. For the last few percent, your eye is faster than tweaking prompts forever.",
            ]),
        ],
    },
    {
        "title": "Reading Results",
        "subtitle": "Two views, same data.",
        "blocks": [
            ("h", "Feed view"),
            ("p", "A chronological stream of cards, one per classified image:"),
            ("code", "●  sunset_city.jpg          vs Landscapes 0.31      [Cities]  0.42"),
            ("ul", [
                "● — confidence dot. Green = clear winner (wide margin over runner-up). Yellow = borderline (narrow margin). Red = below threshold or close call.",
                "Filename of the classified image.",
                '"vs Landscapes 0.31" — the runner-up category and its score. Close runners-up signal coin-flip classifications worth eyeballing.',
                "0.42 — the winner's confidence score.",
                "[Cities] — the category badge, color-coded. Shows \"(manual)\" if you've moved the image to a different category.",
            ]),
            ("h", "Gallery view"),
            ("p", "Collapsible category sections with thumbnail grids. Best for visual spot-checking and bulk review."),
            ("ul", [
                "Click a section header to expand or collapse it. Thumbnails load on first expand.",
                "Click a thumbnail to preview the full-size image alongside its CLIP scores.",
                "Right-click a thumbnail to move it to a different category — useful when CLIP got it wrong.",
                "A ✱ marker appears on thumbnails you've manually moved.",
            ]),
            ("h", "Confidence guide"),
            ("kv", [
                ("0.50+",  "very confident"),
                ("0.20+",  "reasonably confident"),
                ("< 0.10", "essentially a guess"),
            ]),
            ("tip", "Use the filter dropdown to focus on one category. Counter chips above the views show running totals."),
        ],
    },
    {
        "title": "Threshold Tuning",
        "subtitle": "Balancing precision against coverage.",
        "blocks": [
            ("p", 'Anything scoring BELOW the threshold is dumped into an "Unsorted" folder instead of being forced into a wrong category.'),
            ("h", "Default threshold"),
            ("p", "The threshold slider (or type-in field) at the bottom of Options sets the DEFAULT threshold — applied to any category that doesn't have its own. With N categories, set it roughly 1/N, then tune based on what you see in the log."),
            ("kv", [
                ("10 categories", "threshold around 0.15"),
                ("5 categories",  "threshold around 0.25"),
                ("3 categories",  "threshold around 0.10"),
            ]),
            ("tip", "3-category setups want LOWER than 1/N (~0.10) — real winners in 3-way contests still hover near 0.40, and you don't want to dump them all to Unsorted."),
            ("h", "Per-category override"),
            ("p", "Each category has its own threshold column. Different prompts naturally score differently — your NSFW prompt may peak at 0.50+ for clear matches while Scenery hovers around 0.20. Per-category thresholds let each prompt have its own sensitivity."),
            ("ul", [
                "Double-click any category row to edit its threshold.",
                "Leave it at the default if you're unsure — the global slider value applies.",
                "Raise it for categories that over-fire (grabbing too much).",
                "Lower it for categories that under-fire (legit matches scoring below their threshold).",
            ]),
            ("h", "Tuning workflow"),
            ("ul", [
                'If too much ends up in "Unsorted" — lower the relevant category\'s threshold.',
                "If wrong-folder mistakes are creeping in — raise that category's threshold.",
            ]),
        ],
    },
    {
        "title": "Writing Prompts",
        "subtitle": "Better prompts mean better sorting.",
        "blocks": [
            ("p", "The prompt is the text CLIP compares each image against. Think of it like an image caption that describes what should land in this folder."),
            ("h", "Be descriptive, not just keywords"),
            ("compare", [
                ("Bad",  '"landscapes"'),
                ("Good", '"a scenic landscape photograph with mountains, valleys, and open vistas"'),
            ]),
            ("h", "Tuning prompts"),
            ("ul", [
                "Mention what makes the category visually distinct.",
                "If a category is OVER-firing (grabbing too much), make the prompt more specific. Add visual qualifiers to narrow it.",
                "If a category is UNDER-firing, broaden the description.",
                'Include "no people" in scenery prompts if you want to exclude human subjects from that bucket.',
            ]),
            ("h", "Distinguishing similar categories"),
            ("p", "To separate two visually similar categories, CONTRAST them explicitly in their prompts:"),
            ("compare", [
                ("Anime",    '"an anime ILLUSTRATED character"'),
                ("Portrait", '"a PHOTOGRAPH of a real person"'),
            ]),
        ],
    },
    {
        "title": "Workflow",
        "subtitle": "The recommended order of operations.",
        "blocks": [
            ("steps", [
                "Set up your categories with descriptive prompts.",
                "Click ORGANIZE. The app classifies every source image and builds an in-memory plan — nothing has moved yet.",
                "Look at the counter strip — does the spread feel right?",
                "Switch to GALLERY view. Expand a category. Spot-check the thumbnails.",
                "Right-click any wrongly-classified thumbnail and pick \"Move to → [Category]\". The Feed entry updates with a (manual) tag.",
                "Use the FEED view to scan for close calls — runner-up scores within 0.05-0.10 of the winner are coin-flips worth eyeballing.",
                "Once the gallery looks right, click APPLY CHANGES. Files move/copy based on the (possibly corrected) plan and the destination manifest is updated.",
                'Hit "Open Destination" to inspect the sorted folders in your file manager.',
            ]),
            ("h", "Copy vs Move"),
            ("ul", [
                "COPY (default) leaves originals in source. Safer. Recommended for first real runs.",
                "MOVE relocates the files out of source into the categorized destination folders. Use this once you trust the setup.",
            ]),
            ("h", "Re-running"),
            ("p", "On the next run, the app reads the destination's manifest, hashes each source image, and skips anything already organized. You can re-run as often as you want — only new source images get classified."),
            ("tip", "If you want to re-classify everything from scratch, delete .wallpaper_organizer.json from the destination folder."),
        ],
    },
    {
        "title": "Common Gotchas",
        "subtitle": "Things that surprise people on first use.",
        "blocks": [
            ("warn", "First run downloads ~600 MB of model weights from HuggingFace. Wait a minute. Subsequent runs use the cached version."),
            ("ul", [
                "With batching, CPU classification runs ~0.1-0.3 sec per image. A few hundred images takes well under a minute.",
                "Classification does NOT move files. Files only move when you click APPLY CHANGES. If you close the app between classify and apply, the plan is lost — just re-Organize.",
                'Borderline images often have legitimate ambiguity (a city at sunset can fairly belong to "Cities" or "Landscapes"). Don\'t try to make CLIP perfect — for the last few percent, use the Gallery view to manually move them, or fix them in your file manager afterward.',
                "App settings persist in ~/.wallpaper_organizer.json (your home folder). The destination manifest is in <destination>/.wallpaper_organizer.json. Different files — don't confuse them.",
            ]),
            ("h", "Undo"),
            ("p", "After Apply, the red UNDO button reverses the most recent apply — deletes the copied files (copy mode) or moves them back to their original locations (move mode). Only the most recent apply is undoable; older applies are committed."),
            ("tip", "Files you've modified between Apply and Undo are skipped to protect your edits. You'll see them listed in the log."),
            ("h", "Supported file types"),
            ("code", ".jpg, .jpeg, .png, .webp, .bmp, .gif, .tiff, .jfif"),
            ("p", "Subfolders inside the source are scanned recursively."),
        ],
    },
]


# ---------------------------------------------------------------------------
# Manifest — tracks which images have been organized into the destination.
# Lets us skip re-classifying things on subsequent runs and remember manual
# category overrides.
# ---------------------------------------------------------------------------

MANIFEST_FILENAME = ".wallpaper_organizer.json"


def _quick_hash(path: Path) -> str:
    """Cheap content-based hash for deduplication.

    Reads only the first 64KB plus the file size — for wallpaper images
    (typically 1-10MB JPEGs/PNGs) this is effectively unique while taking
    microseconds per file. Full SHA1 over 400 multi-MB images takes ~4s;
    this approach is ~50ms total.
    """
    try:
        size = path.stat().st_size
        h = hashlib.sha1()
        h.update(str(size).encode())
        with open(path, "rb") as f:
            h.update(f.read(65536))
        return h.hexdigest()[:16]
    except Exception:
        return ""


class ImageManifest:
    """Persistent record of images already organized into a destination folder.

    Lives at `<dest>/.wallpaper_organizer.json`. Stores one entry per
    organized image with content hash, filename, current category, and a
    flag for whether the user manually moved it from CLIP's choice.

    On classification, images whose hashes appear in the manifest are
    skipped — they've already been sorted. On Apply, new and updated
    entries are written back.
    """
    VERSION = 1

    def __init__(self, dest: Path):
        self.dest = dest
        self.path = dest / MANIFEST_FILENAME
        self.entries: list[dict] = []
        self._by_hash: dict[str, dict] = {}
        # last_apply records the most recent Apply operation so we can undo it.
        # Schema: {"timestamp": iso8601, "copy_mode": bool,
        #          "operations": [{"src": "...", "target": "...", "hash": "..."}]}
        # Only the most recent apply is undoable. Older applies are committed.
        self.last_apply: Optional[dict] = None
        self.load()

    def load(self):
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text())
            self.entries = data.get("entries", []) or []
            self._by_hash = {e["hash"]: e for e in self.entries if e.get("hash")}
            self.last_apply = data.get("last_apply")
        except Exception:
            # Corrupt manifest — start fresh, but don't blow up.
            self.entries = []
            self._by_hash = {}
            self.last_apply = None

    def save(self):
        try:
            self.dest.mkdir(parents=True, exist_ok=True)
            data = {
                "version": self.VERSION,
                "entries": self.entries,
            }
            if self.last_apply is not None:
                data["last_apply"] = self.last_apply
            self.path.write_text(json.dumps(data, indent=2))
        except Exception:
            pass

    def has_hash(self, h: str) -> bool:
        return bool(h) and h in self._by_hash

    def get(self, h: str) -> Optional[dict]:
        return self._by_hash.get(h)

    def upsert(self, entry: dict):
        """Add or replace a manifest entry, keyed by hash."""
        h = entry.get("hash")
        if not h:
            return
        if h in self._by_hash:
            existing = self._by_hash[h]
            existing.update(entry)
        else:
            self.entries.append(entry)
            self._by_hash[h] = entry

    def remove_by_hash(self, h: str):
        """Remove the entry with the given hash, if present.
        Used during Undo to remove records of files we're reverting."""
        if h in self._by_hash:
            del self._by_hash[h]
        self.entries = [e for e in self.entries if e.get("hash") != h]

    def record_apply(self, copy_mode: bool, operations: list[dict]):
        """Store the most recent apply session for potential undo.
        operations: list of {"src": original_path, "target": destination_path, "hash": ...}"""
        self.last_apply = {
            "timestamp":  datetime.now().isoformat(timespec="seconds"),
            "copy_mode":  copy_mode,
            "operations": operations,
        }

    def clear_last_apply(self):
        """Mark the last apply as no longer undoable (e.g. after a fresh run
        invalidates the previous one, or after Undo consumes it)."""
        self.last_apply = None

    def stats(self) -> dict:
        """Counts per category currently tracked in the manifest."""
        by_cat: dict[str, int] = {}
        for e in self.entries:
            cat = e.get("category", "Unsorted")
            by_cat[cat] = by_cat.get(cat, 0) + 1
        return by_cat


# ---------------------------------------------------------------------------
# CLIP classifier
# ---------------------------------------------------------------------------

class CLIPClassifier:
    """Wraps HuggingFace CLIP for zero-shot image classification."""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        # Lazy imports so the GUI opens fast even on cold starts.
        import torch
        from transformers import CLIPModel, CLIPProcessor
        from PIL import Image

        self.torch = torch
        self.Image = Image
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self._text_features = None
        self._categories: list[tuple[str, str]] = []

    def _extract_features(self, output, kind: str):
        """Get a feature tensor from get_text_features / get_image_features.
        Handles both transformers 4.x (Tensor) and 5.x (BaseModelOutputWithPooling
        whose pooler_output is the projected feature)."""
        if isinstance(output, self.torch.Tensor):
            return output
        if hasattr(output, "pooler_output") and output.pooler_output is not None:
            return output.pooler_output
        for attr in ("text_embeds", "image_embeds"):
            if hasattr(output, attr) and getattr(output, attr) is not None:
                return getattr(output, attr)
        raise RuntimeError(f"Unexpected CLIP {kind} output type: {type(output)}")

    def set_categories(self, categories) -> None:
        """Accepts (name, prompt) or (name, prompt, threshold) tuples.
        Threshold is ignored here — applied at the job level."""
        self._categories = list(categories)
        prompts = [cat[1] for cat in self._categories]
        inputs = self.processor(text=prompts, return_tensors="pt", padding=True).to(self.device)
        with self.torch.no_grad():
            feats = self._extract_features(self.model.get_text_features(**inputs), "text")
            feats = feats / feats.norm(dim=-1, keepdim=True)
        self._text_features = feats

    def classify(self, image_path: Path, top_k: int = 2) -> list[tuple[str, float]]:
        """Return top-k (category_name, confidence) tuples, highest first."""
        if self._text_features is None:
            raise RuntimeError("Call set_categories() first")
        img = self.Image.open(image_path).convert("RGB")
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            img_feats = self._extract_features(self.model.get_image_features(**inputs), "image")
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
            sims = (img_feats @ self._text_features.T).softmax(dim=-1)
        probs = sims[0].cpu().tolist()
        ranked = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)
        k = min(max(top_k, 1), len(self._categories))
        return [(self._categories[i][0], float(p)) for i, p in ranked[:k]]

    def classify_batch(
        self,
        image_paths: list[Path],
        top_k: int = 2,
        batch_size: int = 8,
    ):
        """Yield (image_path, ranked_or_None, error_or_None) for each image.

        Batching lets CLIP process many images in a single forward pass,
        which on CPU is roughly N-times faster than calling classify() N
        times (within reason — gains taper after batch_size ~8 on CPU).

        Bad images are reported per-image without killing the whole batch.
        """
        if self._text_features is None:
            raise RuntimeError("Call set_categories() first")

        k = min(max(top_k, 1), len(self._categories))

        i = 0
        while i < len(image_paths):
            chunk = image_paths[i:i + batch_size]
            i += batch_size

            # Load images; failures are reported but don't kill the batch.
            loaded: list[tuple[Path, object]] = []
            for p in chunk:
                try:
                    img = self.Image.open(p).convert("RGB")
                    loaded.append((p, img))
                except Exception as e:
                    yield (p, None, str(e))

            if not loaded:
                continue

            try:
                imgs = [img for _, img in loaded]
                inputs = self.processor(
                    images=imgs, return_tensors="pt", padding=True
                ).to(self.device)
                with self.torch.no_grad():
                    img_feats = self._extract_features(
                        self.model.get_image_features(**inputs), "image"
                    )
                    img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
                    sims = (img_feats @ self._text_features.T).softmax(dim=-1)
                probs_per_image = sims.cpu().tolist()
            except Exception as e:
                # Whole batch failed — likely a bad image slipped past load.
                for p, _ in loaded:
                    yield (p, None, f"batch failure: {e}")
                continue

            for (p, _), row in zip(loaded, probs_per_image):
                ranked = sorted(enumerate(row), key=lambda x: x[1], reverse=True)
                top = [(self._categories[idx][0], float(prob)) for idx, prob in ranked[:k]]
                yield (p, top, None)


# ---------------------------------------------------------------------------
# Background job (with ETA tracking + status events)
# ---------------------------------------------------------------------------

def _format_eta(seconds: float) -> str:
    if seconds < 1:
        return "almost done"
    if seconds < 60:
        return f"~{int(seconds)}s left"
    minutes = seconds / 60
    if minutes < 60:
        return f"~{int(minutes)} min left"
    return f"~{minutes / 60:.1f}h left"


def _format_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{int(seconds)}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m {s}s"
    h, m = divmod(m, 60)
    return f"{h}h {m}m"


class ClassifyJob:
    """Walks the source dir and produces a classification plan — does NOT
    move or copy files. The plan is emitted item-by-item via the queue;
    the GUI assembles it and lets the user review before applying.

    Skips images whose content hash already appears in the destination's
    manifest (i.e. organized on a previous run)."""

    def __init__(
        self,
        source: Path,
        dest: Path,
        categories,
        manifest: ImageManifest,
        log_queue: Queue,
        cancel_event: threading.Event,
    ):
        """categories is a list of (name, prompt, threshold) tuples.
        Threshold is per-category — applied to that category's winning score."""
        self.source = source
        self.dest = dest
        self.categories = categories
        # Build a name → threshold lookup for the run loop. Default 0.0
        # (i.e. no threshold) if missing — defensive against malformed data.
        self.thresholds = {}
        for cat in categories:
            if len(cat) >= 3:
                self.thresholds[cat[0]] = float(cat[2])
            else:
                self.thresholds[cat[0]] = 0.0
        self.manifest = manifest
        self.log_queue = log_queue
        self.cancel_event = cancel_event

    def _emit(self, kind: str, payload):
        self.log_queue.put((kind, payload))

    def run(self):
        try:
            self._emit("status", "Loading CLIP model (first run downloads ~600 MB)…")
            self._emit("log", "Loading CLIP model (first run downloads ~600 MB)…")
            classifier = CLIPClassifier()
            self._emit("log", f"Model loaded on {classifier.device.upper()}.")
            self._emit("status", "Indexing images…")
            classifier.set_categories(self.categories)

            # Walk source, exclude anything already inside the destination tree.
            all_images = [
                p for p in self.source.rglob("*")
                if p.is_file()
                and p.suffix.lower() in SUPPORTED_EXTS
                and self.dest not in p.parents
            ]

            # Hash + manifest filter — skip what we've already organized.
            self._emit("status", "Checking manifest for already-organized images…")
            images: list[Path] = []
            hashes: dict[Path, str] = {}
            skipped = 0
            for p in all_images:
                if self.cancel_event.is_set():
                    self._emit("log", "Cancelled before classification.")
                    self._emit("done", {"by_cat": {}, "errors": 0, "skipped": skipped,
                                        "elapsed": 0})
                    return
                h = _quick_hash(p)
                hashes[p] = h
                if self.manifest.has_hash(h):
                    skipped += 1
                else:
                    images.append(p)

            total = len(images)
            if skipped:
                self._emit("log",
                           f"Skipped {skipped} image(s) already organized "
                           f"(per manifest in destination).")
            self._emit("log", f"Found {total} new image(s) to classify.")

            if total == 0:
                self._emit("status",
                           "Nothing new to classify — all source images are already in the manifest.")
                self._emit("done", {"by_cat": {}, "errors": 0,
                                    "skipped": skipped, "elapsed": 0})
                return

            errors = 0
            timings: deque = deque(maxlen=50)
            start = time.monotonic()
            processed = 0

            BATCH_SIZE = 8
            t_batch_start = time.monotonic()

            for img_path, ranked, err in classifier.classify_batch(
                images, top_k=2, batch_size=BATCH_SIZE
            ):
                if self.cancel_event.is_set():
                    self._emit("log", "Cancelled.")
                    break

                processed += 1
                if err is not None:
                    errors += 1
                    self._emit("log", f"ERROR on {img_path.name}: {err}")
                else:
                    name, conf = ranked[0]
                    runner_up = ranked[1] if len(ranked) > 1 else None
                    # Per-category threshold: each category has its own cutoff.
                    # If the winner's score is below its category's threshold,
                    # send the image to Unsorted instead.
                    cat_threshold = self.thresholds.get(name, 0.0)
                    if conf < cat_threshold:
                        name = "Unsorted"
                    runner_up_str = (
                        f"  [also: {runner_up[0]} {runner_up[1]:.2f}]" if runner_up else ""
                    )
                    msg = f"{img_path.name} -> {name}/  ({conf:.2f}){runner_up_str}"

                    # Emit a "plan_item" — the GUI uses this to populate
                    # both the Feed view and the Gallery view, AND to build
                    # the in-memory plan that Apply will consume.
                    self._emit("plan_item", {
                        "src_path": str(img_path),
                        "filename": img_path.name,
                        "hash": hashes.get(img_path, ""),
                        "category": name,
                        "confidence": conf,
                        "ranked": ranked,
                        "manually_moved": False,
                        "text": msg,
                    })

                self._emit("progress", (processed, total))

                if processed % BATCH_SIZE == 0 or processed == total:
                    batch_elapsed = time.monotonic() - t_batch_start
                    timings.append(batch_elapsed / max(1, BATCH_SIZE))
                    t_batch_start = time.monotonic()
                    avg = sum(timings) / len(timings)
                    remaining_sec = (total - processed) * avg
                    eta = _format_eta(remaining_sec)
                    self._emit("status",
                               f"Classifying — {processed}/{total}, {eta}")

            elapsed = time.monotonic() - start
            self._emit("done", {
                "errors": errors, "skipped": skipped, "elapsed": elapsed,
            })
        except Exception as e:
            self._emit("error", str(e))


class ApplyJob:
    """Takes a classification plan and performs the file operations.

    Each plan item is a dict with at least: src_path, filename, hash,
    category, manually_moved. We copy or move each source file into the
    destination category folder, then upsert the manifest.
    """

    def __init__(
        self,
        dest: Path,
        plan_items: list[dict],
        copy_mode: bool,
        manifest: ImageManifest,
        log_queue: Queue,
        cancel_event: threading.Event,
    ):
        self.dest = dest
        self.plan_items = plan_items
        self.copy_mode = copy_mode
        self.manifest = manifest
        self.log_queue = log_queue
        self.cancel_event = cancel_event

    def _emit(self, kind: str, payload):
        self.log_queue.put((kind, payload))

    def run(self):
        try:
            total = len(self.plan_items)
            self._emit("status", f"Applying {total} file operation(s)…")
            self._emit("log", f"=== Applying changes ({total} image(s)) ===")
            action = "Copying" if self.copy_mode else "Moving"
            errors = 0
            done = 0
            start = time.monotonic()
            by_cat: dict[str, int] = {}
            # Operations log for the undo feature — recorded into the manifest
            # at the end of the run. Each entry has enough info to reverse
            # the op: source path, target path, hash, action type.
            operations: list[dict] = []

            for item in self.plan_items:
                if self.cancel_event.is_set():
                    self._emit("log", "Cancelled mid-apply.")
                    break
                src = Path(item["src_path"])
                category = item.get("category", "Unsorted")
                try:
                    target_dir = self.dest / category
                    target_dir.mkdir(parents=True, exist_ok=True)
                    target = target_dir / src.name
                    # Resolve filename collisions in the destination folder.
                    j = 1
                    while target.exists() and target.resolve() != src.resolve():
                        target = target_dir / f"{src.stem}_{j}{src.suffix}"
                        j += 1
                    if self.copy_mode:
                        shutil.copy2(src, target)
                    else:
                        shutil.move(str(src), str(target))

                    # Record for undo. Hash of the *final* file content (taken
                    # before any later modifications) lets us verify integrity
                    # at undo time and skip files the user has since edited.
                    h = item.get("hash", "")
                    operations.append({
                        "src":    str(src),
                        "target": str(target),
                        "hash":   h,
                    })

                    # Update manifest with the categorization record.
                    self.manifest.upsert({
                        "hash":              h,
                        "filename":          target.name,
                        "source_filename":   src.name,
                        "category":          category,
                        "manually_moved":    bool(item.get("manually_moved")),
                        "organized_at":      datetime.now().isoformat(timespec="seconds"),
                    })
                    by_cat[category] = by_cat.get(category, 0) + 1
                    done += 1
                except Exception as e:
                    errors += 1
                    self._emit("log", f"ERROR on {src.name}: {e}")
                self._emit("progress", (done + errors, total))

            # Record the apply session for potential undo, then persist.
            if operations:
                self.manifest.record_apply(self.copy_mode, operations)
            self.manifest.save()
            elapsed = time.monotonic() - start
            self._emit("apply_done", {
                "action":   action,
                "by_cat":   by_cat,
                "done":     done,
                "errors":   errors,
                "total":    total,
                "elapsed":  elapsed,
                "undoable": bool(operations),
            })
        except Exception as e:
            self._emit("error", str(e))


class UndoJob:
    """Reverses the most recent Apply operation.

    Iterates the manifest's last_apply.operations in reverse and undoes each:
      - If the apply was a COPY: delete the target file
      - If the apply was a MOVE: move the target back to the original source

    Defensive about file changes since apply:
      - If the target file no longer exists, skip silently (already moved/deleted)
      - If the target's quick_hash no longer matches what we recorded, skip and warn
        (user has edited the file — undo could destroy their changes)

    Manifest entries for successfully undone files are removed, so re-running
    Organize after Undo will pick those source images up again.
    """

    def __init__(
        self,
        dest: Path,
        manifest: ImageManifest,
        log_queue: Queue,
        cancel_event: threading.Event,
    ):
        self.dest = dest
        self.manifest = manifest
        self.log_queue = log_queue
        self.cancel_event = cancel_event

    def _emit(self, kind: str, payload):
        self.log_queue.put((kind, payload))

    def run(self):
        try:
            last = self.manifest.last_apply
            if not last:
                self._emit("log", "Nothing to undo — no recent apply on record.")
                self._emit("undo_done", {"done": 0, "skipped": 0, "errors": 0})
                return

            operations = last.get("operations", [])
            copy_mode = bool(last.get("copy_mode", True))
            total = len(operations)
            action = "Deleting copies" if copy_mode else "Moving files back"
            self._emit("status", f"{action} ({total} file(s))…")
            self._emit("log", f"=== Undoing last apply ({total} operation(s)) ===")

            done = 0
            skipped = 0
            errors = 0
            start = time.monotonic()

            # Reverse iteration so newer ops (which may have been collision-renamed
            # like file_1.jpg) get undone before older ones in the same destination.
            for op in reversed(operations):
                if self.cancel_event.is_set():
                    self._emit("log", "Cancelled mid-undo.")
                    break

                src = Path(op.get("src", ""))
                target = Path(op.get("target", ""))
                expected_hash = op.get("hash", "")

                try:
                    if not target.exists():
                        skipped += 1
                        self._emit("log",
                                   f"SKIP {target.name}: target no longer exists")
                        continue

                    # Integrity check — if the file's been modified since
                    # apply, refuse to undo (don't destroy user's edits).
                    if expected_hash:
                        actual_hash = _quick_hash(target)
                        if actual_hash != expected_hash:
                            skipped += 1
                            self._emit("log",
                                       f"SKIP {target.name}: file modified since apply")
                            continue

                    if copy_mode:
                        # Apply was a copy — source is still there; just remove
                        # the target.
                        target.unlink()
                    else:
                        # Apply was a move — restore the file to its original
                        # source location. Make sure the source dir exists
                        # (user may have moved things around since).
                        src.parent.mkdir(parents=True, exist_ok=True)
                        if src.exists():
                            # Edge case: a different file is now at the original
                            # source path. Don't overwrite it; rename instead.
                            j = 1
                            alt = src.with_name(f"{src.stem}_restored{src.suffix}")
                            while alt.exists():
                                alt = src.with_name(
                                    f"{src.stem}_restored_{j}{src.suffix}")
                                j += 1
                            shutil.move(str(target), str(alt))
                            self._emit("log",
                                       f"Restored {target.name} → {alt} "
                                       f"(original path occupied)")
                        else:
                            shutil.move(str(target), str(src))

                    # Drop the manifest entry so re-runs pick this image up again.
                    if expected_hash:
                        self.manifest.remove_by_hash(expected_hash)
                    done += 1
                except Exception as e:
                    errors += 1
                    self._emit("log", f"ERROR undoing {target.name}: {e}")

                self._emit("progress", (done + skipped + errors, total))

            # Clear the last-apply record (consumed) and persist.
            self.manifest.clear_last_apply()
            self.manifest.save()
            elapsed = time.monotonic() - start
            self._emit("undo_done", {
                "done":    done,
                "skipped": skipped,
                "errors":  errors,
                "total":   total,
                "elapsed": elapsed,
            })
        except Exception as e:
            self._emit("error", str(e))


# ---------------------------------------------------------------------------
# Main app window
# ---------------------------------------------------------------------------

class WallpaperOrganizerApp:
    def __init__(self, root: ctk.CTk):
        self.root = root
        self.root.title("Wallpaper Organizer")
        self.root.geometry("940x820")
        self.root.minsize(840, 720)

        self.source_var = tk.StringVar()
        self.dest_var = tk.StringVar()
        self.copy_var = tk.BooleanVar(value=True)
        self.threshold_var = tk.DoubleVar(value=0.15)
        self.status_var = tk.StringVar(value="Ready")
        self.cancel_event = threading.Event()
        self.log_queue: Queue = Queue()
        self.worker: Optional[threading.Thread] = None
        self._loaded_cats: Optional[list[tuple[str, str]]] = None

        # --- Phase state ---
        # The app moves through three phases:
        #   "idle"       — no plan in memory, idle UI
        #   "classifying"— ClassifyJob is running
        #   "ready"      — plan exists in memory, awaiting Apply
        #   "applying"   — ApplyJob is running
        self._phase = "idle"

        # --- Plan ---
        # In-memory classification plan: keyed by source path (str).
        # Each entry has all fields needed for both review and apply.
        # Mutated by user actions in the Gallery (manual move) and by
        # the classify worker emitting plan_item events.
        self.plan: dict[str, dict] = {}

        # --- Manifest ---
        self.manifest: Optional[ImageManifest] = None

        # --- Log feed state (legacy, kept) ---
        self._log_rows: list[dict] = []
        self._category_colors: dict[str, tuple] = {}
        self._counter_labels: dict[str, ctk.CTkLabel] = {}
        self._counts: dict[str, int] = {}
        self.filter_var = tk.StringVar(value="All")
        self._preview: Optional["PreviewDialog"] = None

        # --- Gallery state ---
        # Maps category name -> GallerySection widget
        self._gallery_sections: dict[str, "GallerySection"] = {}
        # Maps src_path -> ThumbnailWidget for fast lookup on move/remove
        self._thumb_widgets: dict[str, "ThumbnailWidget"] = {}
        # Categories list for this run (drives gallery section order)
        self._current_categories: list[tuple[str, str]] = []

        self._load_config()
        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        # Keep the Undo button state in sync with the destination — when
        # the user picks a new dest folder, peek at its manifest to see if
        # there's an undoable apply on record there.
        self.dest_var.trace_add("write", lambda *a: self._refresh_undo_button())
        # Initial state check (in case dest is loaded from config)
        self.root.after(50, self._refresh_undo_button)
        self._poll_queue()

    # ---- UI build ----

    def _build_ui(self):
        self._setup_treeview_style()
        # Pack status bar first so it docks at the bottom
        self._build_statusbar()

        main = ctk.CTkFrame(self.root, fg_color="transparent")
        main.pack(fill="both", expand=True, padx=14, pady=(12, 6))

        self._build_title_row(main)
        self._build_folders(main)
        self._build_categories(main)
        self._build_options(main)
        self._build_run_row(main)
        self._build_log(main)

    def _build_statusbar(self):
        bar = ctk.CTkFrame(self.root, height=28, corner_radius=0,
                           fg_color=("#e5e7eb", "#1f2937"))
        bar.pack(side="bottom", fill="x")
        bar.pack_propagate(False)
        ctk.CTkLabel(
            bar, textvariable=self.status_var,
            font=ctk.CTkFont(size=12),
            anchor="w",
        ).pack(side="left", fill="both", expand=True, padx=14)

    def _build_title_row(self, parent):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", pady=(0, 12))

        ctk.CTkLabel(
            row, text="Wallpaper Organizer",
            font=ctk.CTkFont(size=22, weight="bold"),
        ).pack(side="left")

        ctk.CTkLabel(
            row, text="    AI-powered wallpaper sorter",
            font=ctk.CTkFont(size=13),
            text_color=("gray45", "gray60"),
        ).pack(side="left", pady=(8, 0))

        self.mode_btn = ctk.CTkButton(
            row, text="🌙", width=38, height=32,
            command=self._toggle_mode,
            fg_color="transparent",
            hover_color=GHOST_HOVER,
            text_color=GHOST_TEXT,
            font=ctk.CTkFont(size=15),
        )
        self.mode_btn.pack(side="right")
        self._update_mode_btn()

    def _build_folders(self, parent):
        section = self._section_frame(parent, "Folders")

        ctk.CTkLabel(section, text="Source:", anchor="w", width=90)\
            .grid(row=0, column=0, sticky="w", padx=(14, 8), pady=(10, 4))
        ctk.CTkEntry(section, textvariable=self.source_var, height=32)\
            .grid(row=0, column=1, sticky="ew", pady=(10, 4))
        ctk.CTkButton(
            section, text="Browse", width=84, height=32,
            command=self._pick_source,
        ).grid(row=0, column=2, padx=(8, 14), pady=(10, 4))

        ctk.CTkLabel(section, text="Destination:", anchor="w", width=90)\
            .grid(row=1, column=0, sticky="w", padx=(14, 8), pady=(0, 12))
        ctk.CTkEntry(section, textvariable=self.dest_var, height=32)\
            .grid(row=1, column=1, sticky="ew", pady=(0, 12))
        ctk.CTkButton(
            section, text="Browse", width=84, height=32,
            command=self._pick_dest,
        ).grid(row=1, column=2, padx=(8, 14), pady=(0, 12))

        section.grid_columnconfigure(1, weight=1)

    def _build_categories(self, parent):
        section = self._section_frame(
            parent,
            "Categories  (folder name → CLIP prompt → threshold — double-click to edit)",
        )

        wrap = ctk.CTkFrame(section, fg_color="transparent")
        wrap.pack(fill="both", expand=True, padx=12, pady=(8, 12))

        # Treeview + scrollbar in a card-style frame
        tree_card = ctk.CTkFrame(wrap, fg_color=("gray85", "gray17"), corner_radius=6)
        tree_card.pack(side="left", fill="both", expand=True)

        self.cat_tree = ttk.Treeview(
            tree_card, columns=("name", "prompt", "threshold"), show="headings",
            height=8, style="WO.Treeview",
        )
        self.cat_tree.heading("name", text="Folder")
        self.cat_tree.heading("prompt", text="CLIP prompt")
        self.cat_tree.heading("threshold", text="Threshold")
        self.cat_tree.column("name", width=160, anchor="w")
        self.cat_tree.column("prompt", width=480, anchor="w")
        self.cat_tree.column("threshold", width=80, anchor="center")
        self.cat_tree.pack(side="left", fill="both", expand=True, padx=2, pady=2)
        self.cat_tree.bind("<Double-1>", lambda e: self._edit_cat())

        scroll = ttk.Scrollbar(tree_card, orient="vertical", command=self.cat_tree.yview)
        scroll.pack(side="left", fill="y")
        self.cat_tree.configure(yscrollcommand=scroll.set)

        # Right-side button rail
        rail = ctk.CTkFrame(wrap, fg_color="transparent")
        rail.pack(side="left", fill="y", padx=(10, 0))
        for label, cmd in [
            ("Add",    self._add_cat),
            ("Edit",   self._edit_cat),
            ("Remove", self._remove_cat),
            ("Reset",  self._reset_cats),
        ]:
            ctk.CTkButton(rail, text=label, width=92, height=32, command=cmd)\
                .pack(fill="x", pady=2)

        # Insert loaded or default categories. Normalize old 2-tuple format
        # to (name, prompt, threshold).
        cats = self._loaded_cats or DEFAULT_CATEGORIES
        for cat in cats:
            if len(cat) == 2:
                name, prompt = cat
                threshold = self.threshold_var.get()
            else:
                name, prompt, threshold = cat[0], cat[1], cat[2]
            self.cat_tree.insert(
                "", "end",
                values=(name, prompt, f"{float(threshold):.2f}"),
            )

    def _build_options(self, parent):
        section = self._section_frame(parent, "Options")

        ctk.CTkCheckBox(
            section, text="Copy files (uncheck to move)",
            variable=self.copy_var,
        ).grid(row=0, column=0, sticky="w", padx=14, pady=(10, 4))

        # Default threshold — applies to new categories. Each existing
        # category has its own threshold (editable per-row in the table).
        ctk.CTkLabel(section, text="Default threshold:")\
            .grid(row=1, column=0, sticky="w", padx=14, pady=(4, 12))
        slider_wrap = ctk.CTkFrame(section, fg_color="transparent")
        slider_wrap.grid(row=1, column=1, sticky="w", padx=20, pady=(4, 12))

        self.thresh_slider = ctk.CTkSlider(
            slider_wrap, from_=0.0, to=0.9,
            variable=self.threshold_var, width=240,
        )
        self.thresh_slider.pack(side="left")

        # Editable text entry alongside the slider. Both bound via a
        # shared StringVar — drag the slider, the number updates; type
        # in the entry, the slider moves.
        self.thresh_entry_var = tk.StringVar(value=f"{self.threshold_var.get():.2f}")
        self.thresh_entry = ctk.CTkEntry(
            slider_wrap, textvariable=self.thresh_entry_var,
            width=60, height=28, justify="center",
            font=ctk.CTkFont(size=13, weight="bold"),
        )
        self.thresh_entry.pack(side="left", padx=(10, 0))

        # Sync the entry when the slider moves
        def _sync_entry_from_slider(*_):
            try:
                self.thresh_entry_var.set(f"{self.threshold_var.get():.2f}")
            except Exception:
                pass
        self.threshold_var.trace_add("write", _sync_entry_from_slider)

        # Sync the slider when the entry is edited and committed
        def _sync_slider_from_entry(*_):
            try:
                v = float(self.thresh_entry_var.get())
                v = max(0.0, min(0.9, v))
                self.threshold_var.set(v)
            except (ValueError, tk.TclError):
                # Invalid input — reset to current slider value
                self.thresh_entry_var.set(f"{self.threshold_var.get():.2f}")
        self.thresh_entry.bind("<Return>", _sync_slider_from_entry)
        self.thresh_entry.bind("<FocusOut>", _sync_slider_from_entry)

        ctk.CTkLabel(
            section,
            text="New categories use this value. Per-category thresholds editable in the table above.",
            font=ctk.CTkFont(size=10),
            text_color=("gray50", "gray60"),
        ).grid(row=2, column=0, columnspan=2, sticky="w", padx=14, pady=(0, 12))

    def _build_run_row(self, parent):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", pady=(0, 10))

        # Primary action — kicks off classification (always; produces a plan)
        self.run_btn = ctk.CTkButton(
            row, text="Organize", width=120, height=38,
            command=self._start,
            fg_color=ACCENT_COLOR, hover_color=ACCENT_HOVER,
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        self.run_btn.pack(side="left")

        # Apply Changes — commits the in-memory plan to disk. Disabled until
        # classification finishes and a plan exists.
        self.apply_btn = ctk.CTkButton(
            row, text="Apply Changes", width=140, height=38,
            command=self._start_apply,
            fg_color=("#16a34a", "#22c55e"),  # green to distinguish from Organize
            hover_color=("#15803d", "#16a34a"),
            font=ctk.CTkFont(size=14, weight="bold"),
            state="disabled",
        )
        self.apply_btn.pack(side="left", padx=(8, 0))

        # Undo last Apply — reverses the most recent commit. Disabled when
        # no undoable apply is recorded in the destination's manifest.
        self.undo_btn = ctk.CTkButton(
            row, text="Undo", width=80, height=38,
            command=self._start_undo,
            fg_color=("#dc2626", "#ef4444"),  # red to flag it as destructive
            hover_color=("#b91c1c", "#dc2626"),
            font=ctk.CTkFont(size=13, weight="bold"),
            state="disabled",
        )
        self.undo_btn.pack(side="left", padx=(8, 0))

        self.cancel_btn = self._ghost_button(row, "Cancel", self._cancel, width=92)
        self.cancel_btn.configure(state="disabled")
        self.cancel_btn.pack(side="left", padx=(8, 14))

        self.progress = ctk.CTkProgressBar(row, height=14)
        self.progress.set(0)
        self.progress.pack(side="left", fill="x", expand=True, padx=(0, 12))

        self.progress_label = ctk.CTkLabel(
            row, text="", width=80,
            font=ctk.CTkFont(size=12),
            text_color=("gray40", "gray70"),
        )
        self.progress_label.pack(side="left", padx=(0, 12))

        self._ghost_button(row, "Open Destination", self._open_dest, width=140)\
            .pack(side="left", padx=(0, 6))
        self._ghost_button(row, "Help & Tips", self._show_help, width=110)\
            .pack(side="left")

    def _build_log(self, parent):
        # Custom section build (vs _section_frame helper) because the title row
        # also hosts the view tabs and filter dropdown.
        wrap = ctk.CTkFrame(parent, fg_color="transparent")
        wrap.pack(fill="both", expand=True, pady=(0, 10))

        title_row = ctk.CTkFrame(wrap, fg_color="transparent")
        title_row.pack(fill="x", padx=4, pady=(0, 4))

        ctk.CTkLabel(
            title_row, text="Results",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=("gray35", "gray70"),
            anchor="w",
        ).pack(side="left", padx=(0, 14))

        # View tabs — Feed (chronological stream) and Gallery (grouped thumbs)
        self._view_var = tk.StringVar(value="feed")
        self.feed_tab = self._tab_button(
            title_row, "Feed", "feed",
            tooltip="Streamed log of each classification",
        )
        self.feed_tab.pack(side="left", padx=(0, 4))
        self.gallery_tab = self._tab_button(
            title_row, "Gallery", "gallery",
            tooltip="Visual review grouped by category",
        )
        self.gallery_tab.pack(side="left")

        # Filter dropdown
        self.filter_combo = ctk.CTkComboBox(
            title_row, values=["All"], width=180, height=26,
            variable=self.filter_var,
            command=lambda *_: self._apply_filter(),
            state="readonly",
            font=ctk.CTkFont(size=11),
            dropdown_font=ctk.CTkFont(size=11),
        )
        self.filter_combo.pack(side="right", padx=(0, 4))
        ctk.CTkLabel(
            title_row, text="Filter:",
            font=ctk.CTkFont(size=11),
            text_color=("gray45", "gray60"),
        ).pack(side="right", padx=(0, 6))

        section = ctk.CTkFrame(wrap, corner_radius=8)
        section.pack(fill="both", expand=True)

        # Counter strip — small horizontal row of category chips, always visible.
        self.counter_strip = ctk.CTkFrame(section, fg_color="transparent")
        self.counter_strip.pack(fill="x", padx=12, pady=(10, 6))
        self._counter_placeholder = ctk.CTkLabel(
            self.counter_strip,
            text="No images classified yet — running counts will appear here.",
            font=ctk.CTkFont(size=11),
            text_color=("gray55", "gray55"),
        )
        self._counter_placeholder.pack(side="left", padx=4)

        # Hint label above the body — changes based on view
        self.view_hint = ctk.CTkLabel(
            section, text="Click a card to preview the image",
            font=ctk.CTkFont(size=10),
            text_color=("gray55", "gray55"),
            anchor="w",
        )
        self.view_hint.pack(fill="x", padx=14, pady=(0, 4))

        # Body container — holds either the Feed view or the Gallery view.
        # We swap by pack_forget on one and pack on the other.
        self._body = ctk.CTkFrame(section, fg_color="transparent")
        self._body.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        # Feed view (default)
        self.log_frame = ctk.CTkScrollableFrame(self._body, corner_radius=6)

        # Gallery view (built lazily on first show, since it's heavy)
        self.gallery_frame: Optional[ctk.CTkScrollableFrame] = None

        self._show_view("feed")

    def _tab_button(self, parent, label: str, view_id: str, tooltip: str = ""):
        """A small CTkButton styled as a view-tab toggle."""
        btn = ctk.CTkButton(
            parent, text=label,
            width=78, height=26,
            command=lambda: self._show_view(view_id),
            font=ctk.CTkFont(size=11, weight="bold"),
            corner_radius=4,
        )
        return btn

    def _show_view(self, view_id: str):
        """Swap between Feed and Gallery in the log section body."""
        self._view_var.set(view_id)
        # Style the tabs
        for tab, vid in [(self.feed_tab, "feed"), (self.gallery_tab, "gallery")]:
            if vid == view_id:
                tab.configure(
                    fg_color=ACCENT_COLOR, hover_color=ACCENT_HOVER,
                    text_color="white",
                )
            else:
                tab.configure(
                    fg_color="transparent",
                    hover_color=GHOST_HOVER,
                    text_color=GHOST_TEXT,
                )

        # Swap the body
        if view_id == "feed":
            if self.gallery_frame is not None:
                self.gallery_frame.pack_forget()
            self.log_frame.pack(fill="both", expand=True)
            self.view_hint.configure(text="Click a card to preview the image")
        else:
            self.log_frame.pack_forget()
            if self.gallery_frame is None:
                self.gallery_frame = ctk.CTkScrollableFrame(
                    self._body, corner_radius=6,
                )
            self.gallery_frame.pack(fill="both", expand=True)
            self.view_hint.configure(
                text="Click a thumbnail to preview · Right-click to move to a different category"
            )
            self._rebuild_gallery()

    # ---- Log feed helpers ----

    def _compute_category_colors(self, categories):
        """Assign a stable color from CATEGORY_PALETTE to each category."""
        colors: dict[str, tuple] = {}
        for i, cat in enumerate(categories):
            colors[cat[0]] = CATEGORY_PALETTE[i % len(CATEGORY_PALETTE)]
        colors["Unsorted"] = UNSORTED_COLOR
        self._category_colors = colors

    def _build_counter_strip(self, categories):
        """Render one count chip per category (plus Unsorted)."""
        # Wipe the strip clean
        for child in self.counter_strip.winfo_children():
            child.destroy()
        self._counter_labels = {}
        self._counts = {}

        # Total chip first (special, neutral styling)
        total_chip = ctk.CTkFrame(
            self.counter_strip, corner_radius=4,
            fg_color=("#e5e7eb", "#1f2937"),
        )
        total_chip.pack(side="left", padx=(0, 6))
        total_label = ctk.CTkLabel(
            total_chip, text="Total  0",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=("gray25", "gray85"),
        )
        total_label.pack(padx=8, pady=3)
        self._counter_labels["__total__"] = total_label
        self._counts["__total__"] = 0

        # One chip per category
        for cat in categories:
            self._add_counter_chip(cat[0])
        self._add_counter_chip("Unsorted")

    def _add_counter_chip(self, name: str):
        bg = self._category_colors.get(name, UNSORTED_COLOR)
        light_bg, dark_bg, light_fg, dark_fg = bg
        chip = ctk.CTkFrame(
            self.counter_strip, corner_radius=4,
            fg_color=(light_bg, dark_bg),
        )
        chip.pack(side="left", padx=2)
        label = ctk.CTkLabel(
            chip, text=f"{name}  0",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=(light_fg, dark_fg),
        )
        label.pack(padx=8, pady=3)
        self._counter_labels[name] = label
        self._counts[name] = 0

    def _bump_counter(self, name: str):
        self._counts["__total__"] = self._counts.get("__total__", 0) + 1
        self._counts[name] = self._counts.get(name, 0) + 1
        if "__total__" in self._counter_labels:
            self._counter_labels["__total__"].configure(
                text=f"Total  {self._counts['__total__']}"
            )
        if name in self._counter_labels:
            self._counter_labels[name].configure(
                text=f"{name}  {self._counts[name]}"
            )

    def _refresh_filter_options(self, categories):
        options = ["All"] + [cat[0] for cat in categories] + ["Unsorted"]
        self.filter_combo.configure(values=options)
        self.filter_var.set("All")

    def _apply_filter(self):
        """Show/hide log rows based on the current filter selection."""
        target = self.filter_var.get()
        # Rebuild visible rows in order to preserve scroll layout.
        for row in self._log_rows:
            try:
                row["widget"].pack_forget()
            except Exception:
                pass
        for row in self._log_rows:
            # Plain text lines (headers, footers, errors) always show.
            visible = row["is_text"] or target == "All" or row["category"] == target
            if visible:
                try:
                    row["widget"].pack(fill="x", padx=4, pady=1)
                except Exception:
                    pass

        # Gallery sections: show/hide whole category if filtered
        for cat, section in self._gallery_sections.items():
            if target == "All" or cat == target:
                section.show()
            else:
                section.hide()

    # ---- Gallery management ----

    def _clear_gallery(self):
        """Destroy all gallery sections and thumb refs. Called on new run."""
        for section in self._gallery_sections.values():
            try:
                section.destroy()
            except Exception:
                pass
        self._gallery_sections.clear()
        self._thumb_widgets.clear()

    def _rebuild_gallery(self):
        """Tear down and rebuild gallery sections from the current plan.
        Called when switching to Gallery view, and when classification ends."""
        if self.gallery_frame is None:
            return
        # Wipe existing sections
        self._clear_gallery()

        if not self.plan:
            ctk.CTkLabel(
                self.gallery_frame,
                text="No classification plan yet. Click Organize to classify a source folder.",
                font=ctk.CTkFont(size=12),
                text_color=("gray45", "gray60"),
            ).pack(pady=40)
            return

        # Group plan items by category, preserving the order of self._current_categories
        by_cat: dict[str, list[dict]] = {}
        for item in self.plan.values():
            cat = item.get("category", "Unsorted")
            by_cat.setdefault(cat, []).append(item)

        category_order = [cat[0] for cat in self._current_categories] + ["Unsorted"]
        for cat in category_order:
            if cat in by_cat:
                self._add_gallery_section(cat, by_cat[cat])
        # Catch any straggler categories not in the predefined order
        for cat, items in by_cat.items():
            if cat not in self._gallery_sections:
                self._add_gallery_section(cat, items)

    def _add_gallery_section(self, category: str, items: list[dict]):
        """Add a collapsible category section to the gallery."""
        bg_pair = self._category_colors.get(category, UNSORTED_COLOR)
        section = GallerySection(
            self.gallery_frame, category, items, bg_pair,
            on_thumb_click=self._on_thumb_click,
            on_move_request=self._on_thumb_move,
            move_targets=self._all_categories_for_move(),
            is_dark_fn=self._is_dark,
            register_thumb=lambda sp, tw: self._thumb_widgets.__setitem__(sp, tw),
        )
        section.pack(fill="x", padx=4, pady=4)
        self._gallery_sections[category] = section

    def _all_categories_for_move(self) -> list[str]:
        """List of category names available as Move-to targets."""
        return [cat[0] for cat in self._current_categories] + ["Unsorted"]

    def _on_thumb_click(self, item: dict):
        """Thumbnail clicked — open preview."""
        src = item.get("src_path", "")
        current = self.plan.get(src, item)
        self._open_preview({
            "path":     src,
            "filename": current.get("filename"),
            "category": current.get("category"),
            "ranked":   current.get("ranked", []),
        })

    def _on_thumb_move(self, src_path: str, new_category: str):
        """User chose to move a thumbnail to a different category."""
        item = self.plan.get(src_path)
        if not item:
            return
        old_category = item.get("category", "Unsorted")
        if old_category == new_category:
            return

        # Update plan
        item["category"] = new_category
        item["manually_moved"] = True

        # Remove the old thumb from its current section, register the new one.
        old_thumb = self._thumb_widgets.get(src_path)
        if old_thumb:
            old_section = self._gallery_sections.get(old_category)
            if old_section:
                old_section.remove_thumb(old_thumb)
            # The old widget is destroyed by remove_thumb
            self._thumb_widgets.pop(src_path, None)

        # Make sure a section exists for the target category
        new_section = self._gallery_sections.get(new_category)
        if new_section is None:
            self._add_gallery_section(new_category, [])
            new_section = self._gallery_sections[new_category]

        # Add a fresh thumb to the new section. If the section's body is not
        # loaded yet, the thumb will get built (and registered) when the user
        # expands it. If loaded, add_thumb_from_item registers it for us.
        new_thumb = new_section.add_thumb_from_item(item)
        if new_thumb is not None:
            self._thumb_widgets[src_path] = new_thumb

        # Update the matching row in the Feed view
        self._update_feed_row(src_path, new_category)

        # Update counter strip
        self._counts[old_category] = max(0, self._counts.get(old_category, 0) - 1)
        self._counts[new_category] = self._counts.get(new_category, 0) + 1
        if old_category in self._counter_labels:
            self._counter_labels[old_category].configure(
                text=f"{old_category}  {self._counts[old_category]}"
            )
        if new_category in self._counter_labels:
            self._counter_labels[new_category].configure(
                text=f"{new_category}  {self._counts[new_category]}"
            )

        self.status_var.set(
            f"Moved {item.get('filename')} → {new_category} (will apply on next Apply Changes)"
        )

    def _update_feed_row(self, src_path: str, new_category: str):
        """Restyle the feed row when its category changes due to a manual move."""
        for row in self._log_rows:
            if row.get("src_path") != src_path:
                continue
            w = row.get("tk_widgets")
            if not w:
                return
            row["category"] = new_category
            new_bg_pair = self._category_colors.get(new_category, UNSORTED_COLOR)
            w["badge_bg_pair"] = new_bg_pair
            light_bg, dark_bg, light_fg, dark_fg = new_bg_pair
            badge_bg = dark_bg if self._is_dark() else light_bg
            badge_fg = dark_fg if self._is_dark() else light_fg
            try:
                w["badge"].configure(bg=badge_bg)
                w["badge_label"].configure(
                    text=f"{new_category} (manual)",
                    bg=badge_bg, fg=badge_fg,
                )
            except Exception:
                pass
            return

    # ---- UI helpers ----

    def _section_frame(self, parent, label: str) -> ctk.CTkFrame:
        """Make a labeled card-style section, returning the inner frame."""
        wrap = ctk.CTkFrame(parent, fg_color="transparent")
        wrap.pack(fill="x", pady=(0, 10))
        ctk.CTkLabel(
            wrap, text=label,
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=("gray35", "gray70"),
            anchor="w",
        ).pack(fill="x", padx=4, pady=(0, 4))
        section = ctk.CTkFrame(wrap, corner_radius=8)
        section.pack(fill="both", expand=True)
        return section

    def _ghost_button(self, parent, text, command, width=100, height=38):
        return ctk.CTkButton(
            parent, text=text, width=width, height=height,
            command=command,
            fg_color="transparent",
            border_width=1,
            border_color=GHOST_BORDER,
            text_color=GHOST_TEXT,
            hover_color=GHOST_HOVER,
        )

    def _setup_treeview_style(self):
        """Style ttk.Treeview to fit dark/light mode."""
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        mode = ctk.get_appearance_mode().lower()
        if mode == "dark":
            bg, fg = "#212121", "#e5e7eb"
            heading_bg, heading_fg = "#1a1a1a", "#d1d5db"
            sel_bg = "#3b82f6"
        else:
            bg, fg = "#fafafa", "#1f2937"
            heading_bg, heading_fg = "#e5e7eb", "#1f2937"
            sel_bg = "#2563eb"
        style.configure(
            "WO.Treeview",
            background=bg, foreground=fg,
            fieldbackground=bg, borderwidth=0, rowheight=26,
        )
        style.configure(
            "WO.Treeview.Heading",
            background=heading_bg, foreground=heading_fg,
            relief="flat", font=("Segoe UI", 9, "bold"),
        )
        style.map(
            "WO.Treeview",
            background=[("selected", sel_bg)],
            foreground=[("selected", "#ffffff")],
        )
        style.map("WO.Treeview.Heading", background=[("active", heading_bg)])

    def _update_mode_btn(self):
        mode = ctk.get_appearance_mode().lower()
        # Icon shows what you'd switch TO
        self.mode_btn.configure(text="☀" if mode == "dark" else "🌙")

    def _toggle_mode(self):
        """Toggle dark/light mode. Note: customtkinter has to recompute
        appearance for every widget on screen, which scales with widget
        count. With hundreds of log rows and gallery thumbs, expect a brief
        freeze. We give the user feedback so it doesn't feel broken."""
        new_mode = "light" if ctk.get_appearance_mode().lower() == "dark" else "dark"

        # Tell the user something's happening before we begin the freeze.
        # The status bar message renders before customtkinter starts churning.
        n_widgets = len(self._log_rows) + len(self._thumb_widgets)
        if n_widgets > 100:
            self.status_var.set(f"Switching to {new_mode} mode…")
            self.root.update_idletasks()

        ctk.set_appearance_mode(new_mode)
        self._update_mode_btn()
        self._setup_treeview_style()

        # Restyle the Feed rows so their raw-tk widgets match the new theme.
        # This is the expensive part for big runs; we batch with update_idletasks
        # at the end rather than per-row so the UI doesn't redraw 400 times.
        self._restyle_log_rows()

        # Gallery thumbnails are intentionally NOT restyled — their badge
        # colors and frame backgrounds were rendered for the old theme but
        # the image content (the actual picture) doesn't change. Restyling
        # 200+ thumbnail tk widgets would freeze the UI for several seconds.
        # The next Organize run rebuilds them fresh in the current theme.
        if self._gallery_sections:
            self.status_var.set(
                f"{new_mode.title()} mode — gallery thumbs keep their original "
                f"styling until the next Organize run."
            )
        else:
            self.status_var.set("Ready")

    # ---- Categories ----

    def _add_cat(self):
        self._cat_dialog(None)

    def _edit_cat(self):
        sel = self.cat_tree.selection()
        if sel:
            self._cat_dialog(sel[0])

    def _remove_cat(self):
        for item in self.cat_tree.selection():
            self.cat_tree.delete(item)

    def _reset_cats(self):
        if not messagebox.askyesno("Reset", "Restore default categories?"):
            return
        for item in self.cat_tree.get_children():
            self.cat_tree.delete(item)
        for name, prompt, threshold in DEFAULT_CATEGORIES:
            self.cat_tree.insert(
                "", "end",
                values=(name, prompt, f"{threshold:.2f}"),
            )

    def _cat_dialog(self, item_id):
        d = ctk.CTkToplevel(self.root)
        d.title("Category")
        d.geometry("560x400")
        d.transient(self.root)
        d.resizable(False, False)
        d.after(80, d.lift)
        d.after(120, d.focus_force)
        d.after(140, d.grab_set)

        body = ctk.CTkFrame(d, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=20, pady=20)

        # Folder name
        ctk.CTkLabel(
            body, text="Folder name", anchor="w",
            font=ctk.CTkFont(size=12, weight="bold"),
        ).pack(fill="x")
        name_var = tk.StringVar()
        ctk.CTkEntry(body, textvariable=name_var, height=34)\
            .pack(fill="x", pady=(4, 12))

        # CLIP prompt
        ctk.CTkLabel(
            body, text="CLIP prompt  (describe what should land here)", anchor="w",
            font=ctk.CTkFont(size=12, weight="bold"),
        ).pack(fill="x")
        prompt_text = ctk.CTkTextbox(body, height=90)
        prompt_text.pack(fill="x", pady=(4, 12))

        # Per-category threshold
        ctk.CTkLabel(
            body, text="Threshold  (below this, images go to Unsorted)", anchor="w",
            font=ctk.CTkFont(size=12, weight="bold"),
        ).pack(fill="x")
        thresh_row = ctk.CTkFrame(body, fg_color="transparent")
        thresh_row.pack(fill="x", pady=(4, 12))

        thresh_var = tk.DoubleVar(value=self.threshold_var.get())
        thresh_entry_var = tk.StringVar(value=f"{thresh_var.get():.2f}")

        thresh_slider = ctk.CTkSlider(
            thresh_row, from_=0.0, to=0.9,
            variable=thresh_var, width=320,
        )
        thresh_slider.pack(side="left")
        thresh_entry = ctk.CTkEntry(
            thresh_row, textvariable=thresh_entry_var,
            width=60, height=28, justify="center",
            font=ctk.CTkFont(size=13, weight="bold"),
        )
        thresh_entry.pack(side="left", padx=(10, 0))

        # Two-way sync between slider and entry
        def _sync_entry(*_):
            try:
                thresh_entry_var.set(f"{thresh_var.get():.2f}")
            except Exception:
                pass
        thresh_var.trace_add("write", _sync_entry)

        def _sync_slider(*_):
            try:
                v = float(thresh_entry_var.get())
                v = max(0.0, min(0.9, v))
                thresh_var.set(v)
            except (ValueError, tk.TclError):
                thresh_entry_var.set(f"{thresh_var.get():.2f}")
        thresh_entry.bind("<Return>", _sync_slider)
        thresh_entry.bind("<FocusOut>", _sync_slider)

        # Populate fields if editing
        if item_id is not None:
            vals = self.cat_tree.item(item_id, "values")
            name_var.set(vals[0])
            prompt_text.insert("1.0", vals[1])
            if len(vals) > 2:
                try:
                    thresh_var.set(float(vals[2]))
                except (ValueError, IndexError):
                    pass

        def save():
            n = name_var.get().strip()
            p = prompt_text.get("1.0", "end").strip()
            if not n or not p:
                messagebox.showerror("Missing fields", "Both fields are required.", parent=d)
                return
            n = "".join(c for c in n if c.isalnum() or c in " _-").strip()
            t = f"{thresh_var.get():.2f}"
            if item_id is None:
                self.cat_tree.insert("", "end", values=(n, p, t))
            else:
                self.cat_tree.item(item_id, values=(n, p, t))
            d.destroy()

        btns = ctk.CTkFrame(body, fg_color="transparent")
        btns.pack(fill="x")
        ctk.CTkButton(
            btns, text="Save", width=100, height=34, command=save,
            fg_color=ACCENT_COLOR, hover_color=ACCENT_HOVER,
        ).pack(side="right", padx=(8, 0))
        self._ghost_button(btns, "Cancel", d.destroy, width=100, height=34)\
            .pack(side="right")

    def _get_categories(self) -> list[tuple[str, str, float]]:
        """Return categories as (name, prompt, threshold) tuples.
        Threshold is parsed from the tree's string representation."""
        result = []
        default = self.threshold_var.get()
        for i in self.cat_tree.get_children():
            vals = self.cat_tree.item(i, "values")
            name = vals[0] if len(vals) > 0 else ""
            prompt = vals[1] if len(vals) > 1 else ""
            try:
                threshold = float(vals[2]) if len(vals) > 2 else default
            except (ValueError, TypeError):
                threshold = default
            result.append((name, prompt, threshold))
        return result

    # ---- Folder pickers ----

    def _pick_source(self):
        d = filedialog.askdirectory(title="Select source folder")
        if d:
            self.source_var.set(d)
            if not self.dest_var.get():
                self.dest_var.set(str(Path(d) / "Organized"))

    def _pick_dest(self):
        d = filedialog.askdirectory(title="Select destination folder")
        if d:
            self.dest_var.set(d)

    # ---- Run ----

    def _start(self):
        """Start classification — phase 1 of the new workflow.

        Walks the source folder, hashes each image, skips ones already in
        the destination's manifest, classifies the rest with CLIP. Output
        is an in-memory plan, not file moves. The user then reviews and
        clicks Apply Changes to commit."""
        source = self.source_var.get().strip()
        dest = self.dest_var.get().strip()
        if not source or not Path(source).is_dir():
            messagebox.showerror("Invalid source", "Pick a valid source folder.")
            return
        if not dest:
            messagebox.showerror("Invalid destination", "Pick a destination folder.")
            return
        cats = self._get_categories()
        if not cats:
            messagebox.showerror("No categories", "Add at least one category.")
            return

        self._save_config()
        Path(dest).mkdir(parents=True, exist_ok=True)

        # Fresh manifest each run — reads on disk from dest.
        self.manifest = ImageManifest(Path(dest))

        # Reset state for the new run
        self.plan.clear()
        self._thumb_widgets.clear()
        # Clear the thumbnail cache — different source images, possibly
        # different paths, no point holding onto old PhotoImages.
        _THUMB_CACHE.clear()
        self.cancel_event.clear()
        self._phase = "classifying"
        self.run_btn.configure(state="disabled")
        self.apply_btn.configure(state="disabled")
        self.cancel_btn.configure(state="normal")
        self.progress.set(0)
        self.progress_label.configure(text="")
        self._current_categories = list(cats)

        # Set up colors + counter strip + filter dropdown for this run's categories.
        self._compute_category_colors(cats)
        self._clear_log()
        self._clear_gallery()
        self._build_counter_strip(cats)
        self._refresh_filter_options(cats)

        self.status_var.set("Starting…")

        job = ClassifyJob(
            source=Path(source),
            dest=Path(dest),
            categories=cats,
            manifest=self.manifest,
            log_queue=self.log_queue,
            cancel_event=self.cancel_event,
        )
        self.worker = threading.Thread(target=job.run, daemon=True)
        self.worker.start()

    def _start_apply(self):
        """Phase 2: commit the in-memory plan to disk."""
        if not self.plan:
            messagebox.showinfo(
                "Nothing to apply",
                "No classification plan in memory yet. Click Organize first.",
            )
            return
        if self.manifest is None:
            messagebox.showerror("Internal error",
                                  "Manifest wasn't initialized — re-run Organize.")
            return

        # Confirm with a quick summary
        by_cat: dict[str, int] = {}
        manual_count = 0
        for item in self.plan.values():
            cat = item.get("category", "Unsorted")
            by_cat[cat] = by_cat.get(cat, 0) + 1
            if item.get("manually_moved"):
                manual_count += 1
        action = "Copy" if self.copy_var.get() else "Move"
        summary_lines = [f"{action} {len(self.plan)} image(s) to:"]
        for cat, count in sorted(by_cat.items()):
            summary_lines.append(f"  • {cat}: {count}")
        if manual_count:
            summary_lines.append(f"\nIncludes {manual_count} manually moved.")
        summary_lines.append("\nContinue?")
        if not messagebox.askyesno("Apply changes", "\n".join(summary_lines)):
            return

        self.cancel_event.clear()
        self._phase = "applying"
        self.run_btn.configure(state="disabled")
        self.apply_btn.configure(state="disabled")
        self.cancel_btn.configure(state="normal")
        self.progress.set(0)
        self.progress_label.configure(text="")
        self.status_var.set("Applying…")

        # Snapshot plan into a list so the worker has a stable iteration order
        plan_items = list(self.plan.values())

        job = ApplyJob(
            dest=Path(self.dest_var.get().strip()),
            plan_items=plan_items,
            copy_mode=self.copy_var.get(),
            manifest=self.manifest,
            log_queue=self.log_queue,
            cancel_event=self.cancel_event,
        )
        self.worker = threading.Thread(target=job.run, daemon=True)
        self.worker.start()

    def _start_undo(self):
        """Reverse the most recent Apply on this destination."""
        dest = self.dest_var.get().strip()
        if not dest:
            messagebox.showerror("Undo",
                                  "Set a destination folder first.")
            return

        # Load manifest fresh from disk — the in-memory one may not match
        # if the user opened the app, never ran Organize, but a prior session
        # left a last_apply entry.
        manifest = ImageManifest(Path(dest))
        if not manifest.last_apply:
            messagebox.showinfo(
                "Undo",
                "No undoable apply found.\n\n"
                "Undo only reverses the most recent Apply for this destination. "
                "Older applies are committed.",
            )
            self.manifest = manifest
            self._refresh_undo_button()
            return

        # Confirm
        n = len(manifest.last_apply.get("operations", []))
        copy_mode = manifest.last_apply.get("copy_mode", True)
        when = manifest.last_apply.get("timestamp", "an unknown time")
        action_desc = ("delete the copied files" if copy_mode
                       else "move the files back to their original locations")
        if not messagebox.askyesno(
            "Undo last apply",
            f"This will {action_desc} from the most recent Apply "
            f"({n} file(s), {when}).\n\n"
            "Files you've modified since the apply will be skipped to protect "
            "your edits.\n\n"
            "Proceed?",
        ):
            return

        self.manifest = manifest
        self.cancel_event.clear()
        self._phase = "undoing"
        self.run_btn.configure(state="disabled")
        self.apply_btn.configure(state="disabled")
        self.undo_btn.configure(state="disabled")
        self.cancel_btn.configure(state="normal")
        self.progress.set(0)
        self.progress_label.configure(text="")
        self.status_var.set("Undoing…")

        job = UndoJob(
            dest=Path(dest),
            manifest=manifest,
            log_queue=self.log_queue,
            cancel_event=self.cancel_event,
        )
        self.worker = threading.Thread(target=job.run, daemon=True)
        self.worker.start()

    def _refresh_undo_button(self):
        """Enable/disable the Undo button based on manifest state."""
        if self._phase in ("classifying", "applying", "undoing"):
            self.undo_btn.configure(state="disabled")
            return
        # Try the in-memory manifest first; fall back to peeking at disk
        # so the button works even before the user clicks Organize.
        has_undo = False
        if self.manifest is not None and self.manifest.last_apply:
            has_undo = True
        else:
            dest = self.dest_var.get().strip()
            if dest:
                p = Path(dest) / MANIFEST_FILENAME
                if p.exists():
                    try:
                        data = json.loads(p.read_text())
                        if data.get("last_apply"):
                            has_undo = True
                    except Exception:
                        pass
        self.undo_btn.configure(state="normal" if has_undo else "disabled")

    def _cancel(self):
        self.cancel_event.set()
        self.cancel_btn.configure(state="disabled")
        self.status_var.set("Cancelling…")

    # ---- Log feed: rows + queue ----

    def _clear_log(self):
        """Destroy all log rows. Called when a fresh run starts."""
        for row in self._log_rows:
            try:
                row["widget"].destroy()
            except Exception:
                pass
        self._log_rows.clear()

    def _append_log_text(self, msg: str, kind: str = "info"):
        """Append a plain text row (header, footer, error). Always visible."""
        # Style by kind
        if kind == "error":
            color = ("#b91c1c", "#fca5a5")
            weight = "bold"
        elif kind == "header":
            color = ("gray30", "gray80")
            weight = "bold"
        else:
            color = ("gray45", "gray60")
            weight = "normal"

        label = ctk.CTkLabel(
            self.log_frame,
            text=msg,
            font=ctk.CTkFont(family="Consolas", size=11, weight=weight),
            text_color=color,
            anchor="w", justify="left",
        )
        label.pack(fill="x", padx=4, pady=1)
        self._log_rows.append({
            "widget": label, "is_text": True, "category": None, "meta": None,
        })
        self._scroll_to_bottom()

    def _append_log_row(self, payload: dict):
        """Append a classified-image card row to the feed."""
        # Pull placeholder out the first time a real row arrives, so the
        # counter strip doesn't show "no images yet" alongside actual counts.
        if (self._counter_placeholder is not None and
                self._counter_placeholder.winfo_exists()):
            try:
                self._counter_placeholder.destroy()
            except Exception:
                pass
            self._counter_placeholder = None

        category = payload.get("category", "?")
        filename = payload.get("filename", "(unknown)")
        confidence = float(payload.get("confidence", 0))
        ranked = payload.get("ranked", [])
        runner_up = ranked[1] if len(ranked) > 1 else None

        # Determine confidence dot color from margin (winner - runner-up).
        # Unsorted is always red. Wide margin = green, narrow = yellow.
        if category == "Unsorted":
            dot_color = DOT_LOW
        elif runner_up is not None:
            margin = confidence - runner_up[1]
            if margin >= 0.15:
                dot_color = DOT_HIGH
            elif margin >= 0.05:
                dot_color = DOT_MEDIUM
            else:
                dot_color = DOT_LOW
        else:
            # Only one category — go on absolute confidence
            dot_color = DOT_HIGH if confidence >= 0.5 else DOT_MEDIUM

        # Resolve category badge colors
        bg_pair = self._category_colors.get(category, UNSORTED_COLOR)
        light_bg, dark_bg, light_fg, dark_fg = bg_pair

        # Build the row using raw tk widgets for speed (CTkFrame children get
        # expensive when there are hundreds of them).
        row_bg = self._resolve_bg()
        row = tk.Frame(self.log_frame, bg=row_bg, cursor="hand2", height=30)
        # Pack defer: we'll pack via _apply_filter or here directly
        row.pack(fill="x", padx=4, pady=1)

        # Confidence dot — tiny canvas with a colored oval
        dot_canvas = tk.Canvas(
            row, width=14, height=14, bg=row_bg,
            highlightthickness=0, bd=0, cursor="hand2",
        )
        dot_canvas.create_oval(2, 2, 12, 12, fill=dot_color, outline="")
        dot_canvas.pack(side="left", padx=(8, 8), pady=4)

        # Category badge — a small frame with a colored background and label
        badge = tk.Frame(
            row, bg=dark_bg if self._is_dark() else light_bg, cursor="hand2",
        )
        badge.pack(side="right", padx=(6, 10), pady=4)
        badge_label = tk.Label(
            badge, text=category,
            font=("Segoe UI", 9, "bold"),
            fg=dark_fg if self._is_dark() else light_fg,
            bg=dark_bg if self._is_dark() else light_bg,
            padx=8, pady=2, cursor="hand2",
        )
        badge_label.pack()

        # Confidence score (monospace)
        score_label = tk.Label(
            row, text=f"{confidence:.2f}",
            font=("Consolas", 11, "bold"),
            fg=self._fg_strong(), bg=row_bg, cursor="hand2",
        )
        score_label.pack(side="right", padx=(0, 6))

        # Runner-up text (small, faded)
        runner_text = ""
        if runner_up is not None:
            runner_text = f"vs {runner_up[0]} {runner_up[1]:.2f}"
        runner_label = tk.Label(
            row, text=runner_text,
            font=("Segoe UI", 9),
            fg=self._fg_muted(), bg=row_bg, cursor="hand2",
        )
        runner_label.pack(side="right", padx=(0, 10))

        # Filename — takes remaining space
        filename_label = tk.Label(
            row, text=filename, anchor="w",
            font=("Segoe UI", 10),
            fg=self._fg_strong(), bg=row_bg, cursor="hand2",
        )
        filename_label.pack(side="left", padx=(2, 8), fill="x", expand=True)

        # Bind click to all children — tkinter doesn't bubble events.
        # Build preview meta from the plan item, since the file may not yet
        # exist at any destination path (we haven't applied changes yet).
        src_path = payload.get("src_path", "")

        def on_click(_event):
            # Always pull the current category from the live plan, since
            # the user may have moved this image since it was first rendered.
            current = self.plan.get(src_path, payload)
            self._open_preview({
                "path":     src_path,
                "filename": current.get("filename", filename),
                "category": current.get("category", category),
                "ranked":   current.get("ranked", ranked),
            })

        for widget in (row, dot_canvas, badge, badge_label,
                       score_label, runner_label, filename_label):
            widget.bind("<Button-1>", on_click)

        row_record = {
            "widget": row, "is_text": False,
            "category": category, "meta": payload,
            "src_path": src_path,
            # Keep refs to per-row tk widgets for theme refresh + relabeling
            "tk_widgets": {
                "row": row, "dot": dot_canvas, "badge": badge,
                "badge_label": badge_label, "score": score_label,
                "runner": runner_label, "filename": filename_label,
                "badge_bg_pair": bg_pair,
            },
        }
        self._log_rows.append(row_record)

        # Honor the active filter — hide if it doesn't match
        if self.filter_var.get() != "All" and self.filter_var.get() != category:
            try:
                row.pack_forget()
            except Exception:
                pass

        self._bump_counter(category)
        self._scroll_to_bottom()

    def _open_preview(self, meta: dict):
        """Open (or replace) the preview dialog for a clicked row."""
        if self._preview is not None:
            try:
                self._preview.close()
            except Exception:
                pass
        self._preview = PreviewDialog(self.root, meta)

    def _scroll_to_bottom(self):
        """Snap the scrollable feed to the newest row."""
        try:
            # CTkScrollableFrame's underlying canvas is `_parent_canvas`
            self.log_frame._parent_canvas.yview_moveto(1.0)
        except Exception:
            pass

    # ---- Theme helpers used by log row rendering ----

    def _is_dark(self) -> bool:
        return ctk.get_appearance_mode().lower() == "dark"

    def _resolve_bg(self) -> str:
        """Background color for log row tk.Frames (matches CTkScrollableFrame).
        Tries to read the actual frame color first, falls back to theme defaults."""
        try:
            fg = self.log_frame.cget("fg_color")
            if isinstance(fg, (tuple, list)):
                color = fg[1 if self._is_dark() else 0]
            else:
                color = fg
            if color != "transparent":
                return color
        except Exception:
            pass
        # Default customtkinter "blue" theme CTkFrame fg_color resolved values.
        return "#242424" if self._is_dark() else "#ebebeb"

    def _fg_strong(self) -> str:
        return "#e5e7eb" if self._is_dark() else "#1f2937"

    def _fg_muted(self) -> str:
        return "#9ca3af" if self._is_dark() else "#6b7280"

    def _restyle_log_rows(self):
        """Re-apply colors to all rendered rows after a theme toggle."""
        bg = self._resolve_bg()
        fg_strong = self._fg_strong()
        fg_muted = self._fg_muted()
        for row in self._log_rows:
            if row.get("is_text"):
                # CTkLabel handles its own theming via the (light, dark) pair
                continue
            w = row.get("tk_widgets")
            if not w:
                continue
            try:
                w["row"].configure(bg=bg)
                w["dot"].configure(bg=bg)
                light_bg, dark_bg, light_fg, dark_fg = w["badge_bg_pair"]
                badge_bg = dark_bg if self._is_dark() else light_bg
                badge_fg = dark_fg if self._is_dark() else light_fg
                w["badge"].configure(bg=badge_bg)
                w["badge_label"].configure(bg=badge_bg, fg=badge_fg)
                w["score"].configure(bg=bg, fg=fg_strong)
                w["runner"].configure(bg=bg, fg=fg_muted)
                w["filename"].configure(bg=bg, fg=fg_strong)
            except Exception:
                pass

    def _poll_queue(self):
        try:
            while True:
                kind, payload = self.log_queue.get_nowait()
                if kind == "log":
                    self._append_log_text(payload)
                elif kind == "plan_item":
                    # Record into the plan, add a row to the Feed.
                    src = payload.get("src_path", "")
                    if src:
                        self.plan[src] = dict(payload)
                    self._append_log_row(payload)
                elif kind == "progress":
                    cur, total = payload
                    self.progress.set(cur / total if total else 0)
                    self.progress_label.configure(text=f"{cur}/{total}")
                elif kind == "status":
                    self.status_var.set(payload)
                elif kind == "done":
                    self._on_classify_done(payload)
                elif kind == "apply_done":
                    self._on_apply_done(payload)
                elif kind == "undo_done":
                    self._on_undo_done(payload)
                elif kind == "error":
                    self._append_log_text(f"FATAL: {payload}", kind="error")
                    self.status_var.set(f"Error: {payload}")
                    messagebox.showerror("Error", payload)
                    self._reset_buttons()
                    self._phase = "idle"
        except Empty:
            pass
        self.root.after(100, self._poll_queue)

    def _on_classify_done(self, summary: dict):
        """Called when ClassifyJob finishes. Plan is now populated."""
        self._append_log_text("")
        self._append_log_text("=== Classification done ===", kind="header")
        elapsed = summary.get("elapsed", 0)
        errors = summary.get("errors", 0)
        skipped = summary.get("skipped", 0)

        # Render per-category totals from the plan itself (more accurate than
        # passing them through the worker — plan is the source of truth).
        by_cat: dict[str, int] = {}
        for item in self.plan.values():
            cat = item.get("category", "Unsorted")
            by_cat[cat] = by_cat.get(cat, 0) + 1

        non_empty = 0
        for name, count in sorted(by_cat.items(), key=lambda x: -x[1]):
            if count:
                self._append_log_text(f"  {name}: {count}")
                non_empty += 1
        if errors:
            self._append_log_text(f"  Errors: {errors}", kind="error")
        if skipped:
            self._append_log_text(f"  (Skipped {skipped} already-organized)")

        if not self.plan:
            self.status_var.set("Done — nothing new to classify")
            self._phase = "idle"
            self._reset_buttons()
            return

        # Plan exists — enable Apply.
        self._phase = "ready"
        self._append_log_text("")
        self._append_log_text(
            f"→ Review the results, then click \"Apply Changes\" to commit.",
            kind="header",
        )
        self.status_var.set(
            f"Ready to apply — {len(self.plan)} image(s) in "
            f"{_format_elapsed(elapsed)}. Review and apply, or re-Organize."
        )
        self.run_btn.configure(state="normal")
        self.apply_btn.configure(state="normal")
        self.cancel_btn.configure(state="disabled")

        # If gallery view is active, rebuild it now that we have a plan.
        if self._view_var.get() == "gallery":
            self._rebuild_gallery()

    def _on_apply_done(self, summary: dict):
        """Called when ApplyJob finishes."""
        self._append_log_text("")
        self._append_log_text("=== Apply complete ===", kind="header")
        action = summary.get("action", "Applied")
        for name, count in sorted(summary.get("by_cat", {}).items(),
                                   key=lambda x: -x[1]):
            if count:
                self._append_log_text(f"  {name}: {count}")
        if summary.get("errors"):
            self._append_log_text(
                f"  Errors: {summary['errors']}", kind="error")
        elapsed = summary.get("elapsed", 0)
        done = summary.get("done", 0)
        undoable = summary.get("undoable", False)
        suffix = "  (Undo available)" if undoable else ""
        self.status_var.set(
            f"{action} done — {done} file(s) in {_format_elapsed(elapsed)}{suffix}"
        )
        # Plan is consumed; clear so user can't double-apply.
        self.plan.clear()
        self._phase = "idle"
        self._reset_buttons()

    def _on_undo_done(self, summary: dict):
        """Called when UndoJob finishes."""
        self._append_log_text("")
        self._append_log_text("=== Undo complete ===", kind="header")
        done = summary.get("done", 0)
        skipped = summary.get("skipped", 0)
        errors = summary.get("errors", 0)
        elapsed = summary.get("elapsed", 0)
        self._append_log_text(f"  Reversed: {done}")
        if skipped:
            self._append_log_text(
                f"  Skipped (modified or missing): {skipped}",
                kind="header",
            )
        if errors:
            self._append_log_text(f"  Errors: {errors}", kind="error")
        self.status_var.set(
            f"Undo done — reversed {done} file(s) in {_format_elapsed(elapsed)}"
        )
        self._phase = "idle"
        self._reset_buttons()

    def _reset_buttons(self):
        """Restore button states based on current phase."""
        self.run_btn.configure(state="normal")
        # Apply only enabled when we have a plan ready
        if self._phase == "ready" and self.plan:
            self.apply_btn.configure(state="normal")
        else:
            self.apply_btn.configure(state="disabled")
        self.cancel_btn.configure(state="disabled")
        # Undo state depends on manifest, not phase — refresh it always.
        self._refresh_undo_button()

    # ---- Config ----

    def _load_config(self):
        if not CONFIG_FILE.exists():
            return
        try:
            cfg = json.loads(CONFIG_FILE.read_text())
            self.source_var.set(cfg.get("source", ""))
            self.dest_var.set(cfg.get("dest", ""))
            self.copy_var.set(cfg.get("copy", True))
            default_threshold = cfg.get("threshold", DEFAULT_THRESHOLD)
            self.threshold_var.set(default_threshold)
            cats = cfg.get("categories")
            if cats:
                # Migrate 2-tuple categories from older configs to 3-tuples.
                # Pre-v1.1.0 configs stored (name, prompt) and used the global
                # threshold for everything. Newer configs store (name, prompt,
                # threshold) per category. On load, missing thresholds inherit
                # the global default so nothing changes for existing users.
                migrated = []
                for c in cats:
                    if len(c) == 2:
                        migrated.append((c[0], c[1], default_threshold))
                    elif len(c) >= 3:
                        try:
                            t = float(c[2])
                        except (ValueError, TypeError):
                            t = default_threshold
                        migrated.append((c[0], c[1], t))
                self._loaded_cats = migrated
            # Restore window geometry if saved. Defend against the
            # second-monitor-now-unplugged case by clamping to the current
            # screen — keep only the size, drop the position, if x or y
            # are off-screen.
            geom = cfg.get("geometry")
            if geom:
                self._apply_saved_geometry(geom)
        except Exception:
            pass

    def _apply_saved_geometry(self, geom: str):
        """Apply a 'WxH+X+Y' string, falling back to just size if off-screen."""
        try:
            # Parse "940x820+200+150" or "940x820-100+150" etc.
            import re
            m = re.match(r"(\d+)x(\d+)([+-]\d+)([+-]\d+)", geom)
            if not m:
                # Just size, no position
                m2 = re.match(r"(\d+)x(\d+)", geom)
                if m2:
                    self.root.geometry(f"{m2.group(1)}x{m2.group(2)}")
                return
            w, h, x, y = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
            sw = self.root.winfo_screenwidth()
            sh = self.root.winfo_screenheight()
            # If the window would land off-screen, drop the position
            if x < -50 or y < -50 or x > sw - 100 or y > sh - 100:
                self.root.geometry(f"{w}x{h}")
            else:
                self.root.geometry(f"{w}x{h}+{x}+{y}")
        except Exception:
            pass

    def _save_config(self):
        try:
            CONFIG_FILE.write_text(json.dumps({
                "source": self.source_var.get(),
                "dest": self.dest_var.get(),
                "copy": self.copy_var.get(),
                "threshold": self.threshold_var.get(),
                "categories": self._get_categories(),
                "geometry": self.root.geometry(),
            }, indent=2))
        except Exception:
            pass

    def _on_close(self):
        """Save geometry + settings on close so they persist across launches."""
        try:
            self._save_config()
        except Exception:
            pass
        try:
            if self._preview is not None:
                self._preview.close()
        except Exception:
            pass
        self.root.destroy()

    # ---- Open destination ----

    def _open_dest(self):
        dest = self.dest_var.get().strip()
        if not dest:
            messagebox.showinfo("Open destination", "No destination folder set yet.")
            return
        p = Path(dest)
        if not p.exists():
            messagebox.showinfo(
                "Open destination",
                f"That folder doesn't exist yet:\n{p}\n\n"
                f"It'll be created the first time you run a real (non-dry) sort.",
            )
            return
        import os as _os
        import sys as _sys
        import subprocess as _sub
        try:
            if _sys.platform == "win32":
                _os.startfile(str(p))  # type: ignore[attr-defined]
            elif _sys.platform == "darwin":
                _sub.run(["open", str(p)], check=False)
            else:
                _sub.run(["xdg-open", str(p)], check=False)
        except Exception as e:
            messagebox.showerror("Open destination", f"Couldn't open folder:\n{e}")

    # ---- Help ----

    def _show_help(self):
        HelpDialog(self.root)


# ---------------------------------------------------------------------------
# Gallery view — thumbnail grid grouped by category
# ---------------------------------------------------------------------------

THUMB_SIZE = 120
THUMBS_PER_ROW = 5

# Module-level cache of rendered PhotoImage objects, keyed by src_path.
# Survives gallery rebuilds within a session — re-expanding a section
# after toggling tabs is instant. Cleared on new Organize run.
_THUMB_CACHE: dict[str, "object"] = {}


def _render_thumb_image(src_path: str):
    """Open an image and produce a center-cropped square PhotoImage.

    Pure compute — no tk widget access — so safe to call from worker threads.
    Caller must create the ImageTk.PhotoImage on the main thread (Tk only
    accepts PhotoImages built on its own thread). So this returns a PIL
    Image, the main thread wraps it via ImageTk.PhotoImage when applying.
    """
    from PIL import Image
    img = Image.open(src_path).convert("RGB")
    w, h = img.size
    # Center-crop to a square first
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    img = img.crop((left, top, left + side, top + side))
    # Then resize down to THUMB_SIZE. LANCZOS gives best quality for downscale.
    img = img.resize((THUMB_SIZE, THUMB_SIZE), Image.LANCZOS)
    return img


def _widget_alive(widget) -> bool:
    """True if a tk widget still exists (hasn't been destroyed)."""
    try:
        return bool(widget.winfo_exists())
    except Exception:
        return False


def _apply_if_alive(thumb: "ThumbnailWidget", pil_image):
    """Apply a loaded image to a thumbnail, if it hasn't been destroyed."""
    if _widget_alive(thumb.frame):
        thumb.set_photo(pil_image)


def _error_if_alive(thumb: "ThumbnailWidget"):
    """Show the error state on a thumbnail, if it hasn't been destroyed."""
    if _widget_alive(thumb.frame):
        thumb.show_error()


class ThumbnailWidget:
    """A single thumbnail card in the Gallery: image + filename + badge.

    Renders synchronously with a placeholder image. The actual image is
    loaded asynchronously by a ThumbnailLoader worker, then applied via
    set_photo() on the main thread. This keeps the UI responsive when
    expanding sections with hundreds of images.
    """

    def __init__(
        self, parent, item: dict, bg_pair: tuple,
        on_click, on_move_request, move_targets: list[str], is_dark_fn,
    ):
        self.item = item
        self.bg_pair = bg_pair
        self.on_click = on_click
        self.on_move_request = on_move_request
        self.move_targets = move_targets
        self.is_dark_fn = is_dark_fn
        self.photo = None
        self._image_item_id = None  # canvas item id for the image
        self._build(parent)

    def _build(self, parent):
        # Outer frame — sized to fit the thumb + label + badge
        self.frame = tk.Frame(
            parent, bd=0, relief="flat",
            highlightthickness=1,
            highlightbackground=self._border_color(),
            bg=self._frame_bg(),
        )

        # Image canvas — initially shows a placeholder, swapped to real image when loaded
        self.canvas = tk.Canvas(
            self.frame,
            width=THUMB_SIZE, height=THUMB_SIZE,
            bg=self._frame_bg(),
            highlightthickness=0, cursor="hand2",
        )
        self.canvas.pack(padx=2, pady=(2, 0))

        # Placeholder: dim gray square with a small "loading" dot pattern.
        # Painted immediately so the layout doesn't shift when real thumbs arrive.
        self._draw_placeholder()

        # If a cached PhotoImage exists, apply it right away.
        src = self.item.get("src_path", "")
        cached = _THUMB_CACHE.get(src)
        if cached is not None:
            self._apply_photo(cached)

        # Manual-move indicator (re-stamped after image loads)
        self._draw_manual_marker_if_needed()

        # Filename — truncate if too long
        fn = self.item.get("filename", "?")
        display = fn if len(fn) <= 16 else fn[:14] + "…"
        self.filename_label = tk.Label(
            self.frame, text=display,
            font=("Segoe UI", 8),
            fg=self._fg_color(),
            bg=self._frame_bg(),
            cursor="hand2",
        )
        self.filename_label.pack(padx=4)

        # Category badge below filename
        light_bg, dark_bg, light_fg, dark_fg = self.bg_pair
        badge_bg = dark_bg if self.is_dark_fn() else light_bg
        badge_fg = dark_fg if self.is_dark_fn() else light_fg
        category = self.item.get("category", "?")
        self.badge_frame = tk.Frame(self.frame, bg=badge_bg, cursor="hand2")
        self.badge_frame.pack(padx=4, pady=(0, 4))
        self.badge_label = tk.Label(
            self.badge_frame, text=category,
            font=("Segoe UI", 7, "bold"),
            fg=badge_fg, bg=badge_bg,
            padx=6, pady=1, cursor="hand2",
        )
        self.badge_label.pack()

        # Bind events to all children — tkinter doesn't bubble
        for widget in (self.frame, self.canvas, self.filename_label,
                       self.badge_frame, self.badge_label):
            widget.bind("<Button-1>", self._handle_click)
            widget.bind("<Button-3>", self._handle_right_click)
            widget.bind("<Enter>", self._on_enter)
            widget.bind("<Leave>", self._on_leave)

    def _draw_placeholder(self):
        """Paint a subtle gray placeholder square so the layout looks intentional
        while real thumbs load in the background."""
        bg = "#2a2a2a" if self.is_dark_fn() else "#e5e5e5"
        self.canvas.delete("placeholder")
        self.canvas.create_rectangle(
            0, 0, THUMB_SIZE, THUMB_SIZE,
            fill=bg, outline="", tags=("placeholder",),
        )

    def _draw_manual_marker_if_needed(self):
        """Stamp the ✱ marker if this image was manually moved."""
        if self.item.get("manually_moved"):
            self.canvas.delete("manual_marker")
            self.canvas.create_text(
                THUMB_SIZE - 4, 6,
                text="✱", fill="#fbbf24",
                font=("Segoe UI", 12, "bold"),
                anchor="ne",
                tags=("manual_marker",),
            )

    def set_photo(self, pil_image):
        """Called from the main thread when the background loader has a PIL
        image ready. Wraps it in a PhotoImage and swaps it into the canvas."""
        try:
            from PIL import ImageTk
            photo = ImageTk.PhotoImage(pil_image)
            self._apply_photo(photo)
            _THUMB_CACHE[self.item.get("src_path", "")] = photo
        except Exception:
            pass

    def _apply_photo(self, photo):
        """Replace the placeholder with the real thumbnail image."""
        self.photo = photo
        self.canvas.delete("placeholder")
        if self._image_item_id is not None:
            self.canvas.delete(self._image_item_id)
        cx = THUMB_SIZE // 2
        cy = THUMB_SIZE // 2
        self._image_item_id = self.canvas.create_image(
            cx, cy, image=photo, anchor="center",
        )
        # Re-stamp the manual marker on top of the image
        self._draw_manual_marker_if_needed()

    def show_error(self):
        """Called when the background loader failed to open the image."""
        self.canvas.delete("placeholder")
        self.canvas.create_rectangle(
            0, 0, THUMB_SIZE, THUMB_SIZE,
            fill="#7f1d1d", outline="",
        )
        self.canvas.create_text(
            THUMB_SIZE // 2, THUMB_SIZE // 2,
            text="?", fill="#fca5a5",
            font=("Segoe UI", 28, "bold"),
        )

    def _frame_bg(self) -> str:
        return "#1f1f1f" if self.is_dark_fn() else "#f0f0f0"

    def _fg_color(self) -> str:
        return "#e5e7eb" if self.is_dark_fn() else "#1f2937"

    def _border_color(self) -> str:
        return "#2a2a2a" if self.is_dark_fn() else "#d1d5db"

    def _border_hover(self) -> str:
        return "#3b82f6" if self.is_dark_fn() else "#2563eb"

    def _handle_click(self, _event):
        self.on_click(self.item)

    def _handle_right_click(self, event):
        """Pop up a context menu with Move-to options."""
        menu = tk.Menu(self.frame, tearoff=0)
        current = self.item.get("category", "Unsorted")
        move_menu = tk.Menu(menu, tearoff=0)
        for target in self.move_targets:
            if target == current:
                continue
            move_menu.add_command(
                label=target,
                command=lambda t=target: self.on_move_request(
                    self.item.get("src_path", ""), t),
            )
        menu.add_cascade(label="Move to", menu=move_menu)
        menu.add_separator()
        menu.add_command(
            label="Preview",
            command=lambda: self.on_click(self.item),
        )
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()

    def _on_enter(self, _event):
        self.frame.configure(highlightbackground=self._border_hover())

    def _on_leave(self, _event):
        self.frame.configure(highlightbackground=self._border_color())

    def set_category(self, category: str, bg_pair: tuple):
        """Update the badge to reflect a new category (e.g. after a move)."""
        self.bg_pair = bg_pair
        light_bg, dark_bg, light_fg, dark_fg = bg_pair
        badge_bg = dark_bg if self.is_dark_fn() else light_bg
        badge_fg = dark_fg if self.is_dark_fn() else light_fg
        try:
            self.badge_frame.configure(bg=badge_bg)
            self.badge_label.configure(text=category, bg=badge_bg, fg=badge_fg)
            self._draw_manual_marker_if_needed()
        except Exception:
            pass

    def pack(self, **kwargs):
        self.frame.pack(**kwargs)

    def grid(self, **kwargs):
        self.frame.grid(**kwargs)

    def pack_forget(self):
        self.frame.pack_forget()

    def grid_forget(self):
        self.frame.grid_forget()

    def destroy(self):
        try:
            self.frame.destroy()
        except Exception:
            pass


class GallerySection:
    """Collapsible category section in the gallery: header + thumbnail grid.

    Header shows: caret (▼/▶), category name, count, colored badge.
    Body holds a flow-style grid of ThumbnailWidgets. Lazy thumbnail
    loading: body widgets aren't actually built until the section expands
    for the first time (so 400-image plans don't choke the UI).
    """

    def __init__(
        self, parent, category: str, items: list[dict], bg_pair: tuple,
        on_thumb_click, on_move_request, move_targets: list[str],
        is_dark_fn, register_thumb,
    ):
        self.category = category
        self.items = list(items)
        self.bg_pair = bg_pair
        self.on_thumb_click = on_thumb_click
        self.on_move_request = on_move_request
        self.move_targets = move_targets
        self.is_dark_fn = is_dark_fn
        self.register_thumb = register_thumb
        self._expanded = False
        self._loaded = False
        self._thumbs: list[ThumbnailWidget] = []

        self.frame = ctk.CTkFrame(parent, corner_radius=6)
        self._build_header()

        # Body container holds the thumbnail grid when expanded.
        self.body = ctk.CTkFrame(self.frame, fg_color="transparent")
        # Don't pack body yet — happens on expand

    def _build_header(self):
        header = ctk.CTkFrame(self.frame, fg_color="transparent",
                                height=42, cursor="hand2")
        header.pack(fill="x", padx=10, pady=(8, 4))
        header.pack_propagate(False)

        # Make the entire header clickable
        header.bind("<Button-1>", lambda e: self.toggle())

        # Caret
        self.caret_label = ctk.CTkLabel(
            header, text="▶",
            font=ctk.CTkFont(size=12),
            text_color=("gray45", "gray70"),
            width=20, cursor="hand2",
        )
        self.caret_label.pack(side="left")
        self.caret_label.bind("<Button-1>", lambda e: self.toggle())

        # Category badge
        light_bg, dark_bg, light_fg, dark_fg = self.bg_pair
        badge_bg = dark_bg if self.is_dark_fn() else light_bg
        badge_fg = dark_fg if self.is_dark_fn() else light_fg
        self.badge_frame = ctk.CTkFrame(
            header, corner_radius=4,
            fg_color=(light_bg, dark_bg),
        )
        self.badge_frame.pack(side="left", padx=(2, 10))
        self.badge_label = ctk.CTkLabel(
            self.badge_frame, text=self.category,
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=(light_fg, dark_fg),
        )
        self.badge_label.pack(padx=10, pady=3)
        # Header click on the badge too
        self.badge_frame.bind("<Button-1>", lambda e: self.toggle())
        self.badge_label.bind("<Button-1>", lambda e: self.toggle())

        # Count
        self.count_label = ctk.CTkLabel(
            header, text=f"{len(self.items)} image(s)",
            font=ctk.CTkFont(size=12),
            text_color=("gray45", "gray60"),
            cursor="hand2",
        )
        self.count_label.pack(side="left")
        self.count_label.bind("<Button-1>", lambda e: self.toggle())

        # Hint on the right side
        hint = ctk.CTkLabel(
            header, text="click to expand",
            font=ctk.CTkFont(size=10),
            text_color=("gray60", "gray50"),
            cursor="hand2",
        )
        hint.pack(side="right")
        hint.bind("<Button-1>", lambda e: self.toggle())
        self._hint = hint

    def _refresh_count_label(self):
        self.count_label.configure(text=f"{len(self.items)} image(s)")

    def toggle(self):
        if self._expanded:
            self.collapse()
        else:
            self.expand()

    def expand(self):
        if self._expanded:
            return
        self._expanded = True
        self.caret_label.configure(text="▼")
        self._hint.configure(text="click to collapse")

        # Lazy-load thumbnails on first expand
        if not self._loaded:
            self._build_body()
            self._loaded = True
        self.body.pack(fill="x", padx=10, pady=(0, 8))

    def collapse(self):
        if not self._expanded:
            return
        self._expanded = False
        self.caret_label.configure(text="▶")
        self._hint.configure(text="click to expand")
        self.body.pack_forget()

    def _build_body(self):
        """Create thumbnail widgets immediately with placeholders, and kick
        off a background worker to fill in the real images progressively.

        This keeps the UI responsive — instead of freezing for 3+ seconds
        while PIL decodes 200 images, the section opens instantly and
        thumbnails populate as they're ready (~10-30ms per image)."""
        # Build all thumb widgets synchronously (they're cheap — no image decode).
        for item in self.items:
            thumb = ThumbnailWidget(
                self.body, item, self.bg_pair,
                on_click=self.on_thumb_click,
                on_move_request=self.on_move_request,
                move_targets=self.move_targets,
                is_dark_fn=self.is_dark_fn,
            )
            self._thumbs.append(thumb)
            self.register_thumb(item.get("src_path", ""), thumb)

        self._regrid_thumbs()

        # Find which thumbs need image loading (not already cached)
        to_load = [
            t for t in self._thumbs
            if t.item.get("src_path", "") not in _THUMB_CACHE
        ]
        if to_load:
            self._start_thumb_loader(to_load)

    def _start_thumb_loader(self, thumbs_to_load: list):
        """Start a background thread that decodes images and posts the
        results back to the main thread for application to widgets."""
        # Take a stable list of (thumb, src_path) tuples
        work = [(t, t.item.get("src_path", "")) for t in thumbs_to_load]
        # Get a reference to the tk root so we can schedule callbacks on it
        root = self.frame.winfo_toplevel()

        def worker():
            for thumb, src_path in work:
                # If widget got destroyed (section rebuild, new run), bail.
                if not _widget_alive(thumb.frame):
                    continue
                try:
                    pil_img = _render_thumb_image(src_path)
                    # Schedule the photo application on the main thread.
                    # tk.PhotoImage must be constructed on the main thread.
                    root.after(0, lambda t=thumb, img=pil_img: _apply_if_alive(t, img))
                except Exception:
                    root.after(0, lambda t=thumb: _error_if_alive(t))

        threading.Thread(target=worker, daemon=True).start()

    def _regrid_thumbs(self):
        """Arrange all thumbs in a fixed-columns grid."""
        # Forget any prior layout
        for t in self._thumbs:
            t.grid_forget()
        for i, thumb in enumerate(self._thumbs):
            r, c = divmod(i, THUMBS_PER_ROW)
            thumb.grid(row=r, column=c, padx=4, pady=4)

    # ---- Move operations ----

    def add_thumb_from_item(self, item: dict):
        """Add a fresh thumbnail for this plan item to the section.
        Registers the new widget with the app's thumb map. Returns the
        newly created ThumbnailWidget, or None if the section's body
        isn't loaded yet (the thumb will be built on next expand)."""
        self.items.append(item)
        self._refresh_count_label()
        if not self._loaded:
            return None
        new_thumb = ThumbnailWidget(
            self.body, item, self.bg_pair,
            on_click=self.on_thumb_click,
            on_move_request=self.on_move_request,
            move_targets=self.move_targets,
            is_dark_fn=self.is_dark_fn,
        )
        self._thumbs.append(new_thumb)
        self.register_thumb(item.get("src_path", ""), new_thumb)
        self._regrid_thumbs()
        return new_thumb

    def remove_thumb(self, thumb: ThumbnailWidget):
        """Remove a thumbnail from this section by widget reference."""
        target_sp = thumb.item.get("src_path", "")
        self.items = [it for it in self.items if it.get("src_path") != target_sp]
        self._thumbs = [t for t in self._thumbs if t is not thumb]
        self._refresh_count_label()
        if self._loaded:
            try:
                thumb.grid_forget()
                thumb.destroy()
                self._regrid_thumbs()
            except Exception:
                pass

    # ---- Filtering plumbing ----

    def show(self):
        try:
            self.frame.pack(fill="x", padx=4, pady=4)
        except Exception:
            pass

    def hide(self):
        try:
            self.frame.pack_forget()
        except Exception:
            pass

    def pack(self, **kwargs):
        self.frame.pack(**kwargs)

    def destroy(self):
        try:
            self.frame.destroy()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Preview dialog — pops up when the user clicks a classified log entry
# ---------------------------------------------------------------------------

class PreviewDialog:
    """Shows the image, filename, and top-K confidence scores for one classified entry."""

    # Cap rendered preview at these dimensions; we preserve aspect ratio.
    MAX_W = 720
    MAX_H = 480

    def __init__(self, parent: ctk.CTk, meta: dict):
        self.win = ctk.CTkToplevel(parent)
        self.win.title("Preview")
        self.win.transient(parent)
        self.win.attributes("-topmost", True)
        # Drop the topmost after appearing so it doesn't permanently block.
        self.win.after(300, lambda: self.win.attributes("-topmost", False))

        path = Path(meta["path"])
        ranked = meta.get("ranked", [])
        category = meta.get("category", "?")

        # Try to load and render the image. If anything fails, show error text instead.
        photo = None
        load_error = None
        try:
            from PIL import Image, ImageTk
            img = Image.open(path).convert("RGB")
            img.thumbnail((self.MAX_W, self.MAX_H), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
        except Exception as e:
            load_error = str(e)

        # ---- Layout ----
        # Filename header
        ctk.CTkLabel(
            self.win, text=path.name,
            font=ctk.CTkFont(size=14, weight="bold"),
        ).pack(padx=16, pady=(14, 4))

        # Path (small, muted, full)
        ctk.CTkLabel(
            self.win, text=str(path),
            font=ctk.CTkFont(size=10),
            text_color=("gray45", "gray60"),
            wraplength=self.MAX_W,
        ).pack(padx=16, pady=(0, 10))

        # Image (or error message)
        if photo is not None:
            label = tk.Label(self.win, image=photo, borderwidth=0, highlightthickness=0,
                             bg=self._bg_color())
            label.image = photo  # keep a reference
            label.pack(padx=16, pady=(0, 10))
        else:
            ctk.CTkLabel(
                self.win,
                text=f"Couldn't load image:\n{load_error or 'unknown error'}",
                text_color=("#b91c1c", "#fca5a5"),
                font=ctk.CTkFont(size=12),
            ).pack(padx=16, pady=(0, 10))

        # Sorted-into-category banner
        sorted_into = ctk.CTkFrame(self.win, corner_radius=6,
                                    fg_color=("#dbeafe", "#1e3a8a"))
        sorted_into.pack(fill="x", padx=16, pady=(0, 8))
        ctk.CTkLabel(
            sorted_into,
            text=f"  Sorted into:  {category}",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=("#1e3a8a", "#dbeafe"),
            anchor="w",
        ).pack(fill="x", padx=10, pady=8)

        # Top-K scores table
        if ranked:
            scores = ctk.CTkFrame(self.win, fg_color="transparent")
            scores.pack(fill="x", padx=16, pady=(0, 8))
            ctk.CTkLabel(
                scores, text="CLIP confidence scores:",
                font=ctk.CTkFont(size=11, weight="bold"),
                text_color=("gray35", "gray70"),
                anchor="w",
            ).pack(fill="x", pady=(0, 4))
            for i, (cat, score) in enumerate(ranked):
                row = ctk.CTkFrame(scores, fg_color="transparent")
                row.pack(fill="x", pady=1)
                marker = "▸ " if i == 0 else "  "
                ctk.CTkLabel(
                    row, text=f"{marker}{cat}",
                    font=ctk.CTkFont(size=12,
                                     weight="bold" if i == 0 else "normal"),
                    anchor="w", width=200,
                ).pack(side="left")
                ctk.CTkLabel(
                    row, text=f"{score:.3f}",
                    font=ctk.CTkFont(family="Consolas", size=12,
                                     weight="bold" if i == 0 else "normal"),
                    text_color=ACCENT_COLOR if i == 0 else ("gray35", "gray70"),
                    anchor="w",
                ).pack(side="left", padx=(8, 0))

        # Close button
        ctk.CTkButton(
            self.win, text="Close", width=100,
            command=self.close,
        ).pack(pady=(8, 14))

        # Keyboard: Escape closes
        self.win.bind("<Escape>", lambda e: self.close())

        # Center near the parent
        self.win.update_idletasks()
        try:
            px = parent.winfo_x() + parent.winfo_width() // 2 - self.win.winfo_width() // 2
            py = parent.winfo_y() + parent.winfo_height() // 2 - self.win.winfo_height() // 2
            self.win.geometry(f"+{max(0, px)}+{max(0, py)}")
        except Exception:
            pass

    def _bg_color(self) -> str:
        """Match the image label's background to the current theme."""
        return "#212121" if ctk.get_appearance_mode().lower() == "dark" else "#fafafa"

    def close(self):
        try:
            self.win.destroy()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Sectioned Help dialog with sidebar navigation
# ---------------------------------------------------------------------------

class HelpDialog:
    """Sectioned help dialog. Sidebar of sections on the left, designed
    content on the right rendered from typed blocks."""

    # Conservative wraplength that works at minsize. Content is fixed-width;
    # if the dialog is enlarged the content stays this wide with whitespace
    # on the right. Keeps render code simple — no resize re-layout needed.
    CONTENT_WRAP = 500

    def __init__(self, parent: ctk.CTk):
        self.win = ctk.CTkToplevel(parent)
        self.win.title("Wallpaper Organizer — Help & Tips")
        self.win.geometry("860x640")
        self.win.minsize(740, 540)
        self.win.transient(parent)
        self.win.after(80, self.win.lift)
        self.win.after(120, self.win.focus_force)

        body = ctk.CTkFrame(self.win, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=14, pady=(14, 6))

        # ---- Sidebar ----
        sidebar = ctk.CTkFrame(body, width=210, corner_radius=8)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        ctk.CTkLabel(
            sidebar, text="SECTIONS",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color=("gray45", "gray55"),
            anchor="w",
        ).pack(fill="x", padx=14, pady=(14, 8))

        self.section_buttons: list[ctk.CTkButton] = []
        for i, section in enumerate(HELP_SECTIONS):
            btn = ctk.CTkButton(
                sidebar, text=section["title"], anchor="w",
                width=190, height=34,
                fg_color="transparent",
                text_color=("gray15", "gray85"),
                hover_color=("gray80", "gray22"),
                font=ctk.CTkFont(size=13),
                command=lambda idx=i: self._show_section(idx),
            )
            btn.pack(fill="x", padx=10, pady=1)
            self.section_buttons.append(btn)

        # ---- Content area ----
        self.content_card = ctk.CTkFrame(body, corner_radius=8)
        self.content_card.pack(side="left", fill="both", expand=True, padx=(12, 0))

        # Heading + subtitle live OUTSIDE the scrollable area so they stay
        # pinned at the top while content scrolls.
        header = ctk.CTkFrame(self.content_card, fg_color="transparent")
        header.pack(fill="x", padx=22, pady=(18, 0))

        self.heading = ctk.CTkLabel(
            header, text="",
            font=ctk.CTkFont(size=20, weight="bold"),
            anchor="w",
        )
        self.heading.pack(fill="x")

        self.subtitle = ctk.CTkLabel(
            header, text="",
            font=ctk.CTkFont(size=12),
            text_color=("gray45", "gray60"),
            anchor="w",
        )
        self.subtitle.pack(fill="x", pady=(2, 12))

        # Subtle divider line under the header
        divider = ctk.CTkFrame(self.content_card, height=1, corner_radius=0,
                                fg_color=("gray80", "gray25"))
        divider.pack(fill="x", padx=22)

        # Scrollable body
        self.content_scroll = ctk.CTkScrollableFrame(
            self.content_card, corner_radius=0, fg_color="transparent",
        )
        self.content_scroll.pack(fill="both", expand=True, padx=4, pady=(10, 14))

        # ---- Footer ----
        footer = ctk.CTkFrame(self.win, fg_color="transparent")
        footer.pack(fill="x", padx=14, pady=(6, 14))
        ctk.CTkButton(
            footer, text="Close", width=100, height=34,
            command=self.win.destroy,
            fg_color=ACCENT_COLOR, hover_color=ACCENT_HOVER,
        ).pack(side="right")

        self._show_section(0)

    # ---- Section switching ----

    def _show_section(self, idx: int):
        # Highlight the chosen sidebar button
        for i, btn in enumerate(self.section_buttons):
            if i == idx:
                btn.configure(
                    fg_color=ACCENT_COLOR, text_color="white",
                    hover_color=ACCENT_HOVER,
                )
            else:
                btn.configure(
                    fg_color="transparent",
                    text_color=("gray15", "gray85"),
                    hover_color=("gray80", "gray22"),
                )

        section = HELP_SECTIONS[idx]
        self.heading.configure(text=section["title"])
        self.subtitle.configure(text=section.get("subtitle", ""))

        # Wipe and rebuild content
        for child in self.content_scroll.winfo_children():
            child.destroy()

        for block_type, content in section["blocks"]:
            self._render_block(block_type, content)

        # Scroll to top when switching sections
        try:
            self.content_scroll._parent_canvas.yview_moveto(0)
        except Exception:
            pass

    # ---- Block renderers ----

    def _render_block(self, block_type: str, content):
        renderer = {
            "p":       self._render_paragraph,
            "h":       self._render_subheading,
            "steps":   self._render_steps,
            "ul":      self._render_bullets,
            "code":    self._render_code,
            "tip":     lambda c: self._render_callout("TIP",  c, ("#dcfce7", "#14532d"), ("#166534", "#86efac")),
            "warn":    lambda c: self._render_callout("NOTE", c, ("#fef3c7", "#451a03"), ("#92400e", "#fcd34d")),
            "compare": self._render_compare,
            "kv":      self._render_kv,
        }.get(block_type)
        if renderer:
            renderer(content)

    def _render_paragraph(self, text: str):
        ctk.CTkLabel(
            self.content_scroll, text=text,
            font=ctk.CTkFont(size=13),
            text_color=("gray20", "gray85"),
            anchor="w", justify="left",
            wraplength=self.CONTENT_WRAP,
        ).pack(fill="x", padx=18, pady=(0, 12), anchor="w")

    def _render_subheading(self, text: str):
        ctk.CTkLabel(
            self.content_scroll, text=text.upper(),
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=("gray35", "gray60"),
            anchor="w",
        ).pack(fill="x", padx=18, pady=(8, 6), anchor="w")

    def _render_steps(self, items: list):
        for i, step_text in enumerate(items, 1):
            row = ctk.CTkFrame(self.content_scroll, fg_color="transparent")
            row.pack(fill="x", padx=18, pady=(0, 10), anchor="w")

            # Numbered badge — a small accent-colored circle with the number
            badge = ctk.CTkFrame(
                row, width=24, height=24, corner_radius=12,
                fg_color=ACCENT_COLOR,
            )
            badge.pack(side="left", padx=(0, 12), anchor="n")
            badge.pack_propagate(False)
            ctk.CTkLabel(
                badge, text=str(i),
                font=ctk.CTkFont(size=11, weight="bold"),
                text_color="white",
            ).pack(expand=True)

            ctk.CTkLabel(
                row, text=step_text,
                font=ctk.CTkFont(size=13),
                text_color=("gray20", "gray85"),
                anchor="w", justify="left",
                wraplength=self.CONTENT_WRAP - 40,
            ).pack(side="left", fill="x", expand=True, anchor="n", pady=(2, 0))

    def _render_bullets(self, items: list):
        for item_text in items:
            row = ctk.CTkFrame(self.content_scroll, fg_color="transparent")
            row.pack(fill="x", padx=18, pady=(0, 6), anchor="w")
            ctk.CTkLabel(
                row, text="•",
                font=ctk.CTkFont(size=15, weight="bold"),
                text_color=ACCENT_COLOR,
                width=14, anchor="n",
            ).pack(side="left", padx=(4, 10), anchor="n")
            ctk.CTkLabel(
                row, text=item_text,
                font=ctk.CTkFont(size=13),
                text_color=("gray20", "gray85"),
                anchor="w", justify="left",
                wraplength=self.CONTENT_WRAP - 36,
            ).pack(side="left", fill="x", expand=True, anchor="n")

    def _render_code(self, text: str):
        wrap = ctk.CTkFrame(
            self.content_scroll, corner_radius=6,
            fg_color=("#f3f4f6", "#1a1a1a"),
        )
        wrap.pack(fill="x", padx=18, pady=(0, 12), anchor="w")
        ctk.CTkLabel(
            wrap, text=text,
            font=ctk.CTkFont(family="Consolas", size=12),
            text_color=("#1f2937", "#e5e7eb"),
            anchor="w", justify="left",
            wraplength=self.CONTENT_WRAP - 28,
        ).pack(fill="x", padx=14, pady=10)

    def _render_callout(self, label: str, text: str,
                        bg_pair: tuple, fg_pair: tuple):
        box = ctk.CTkFrame(self.content_scroll, corner_radius=6, fg_color=bg_pair)
        box.pack(fill="x", padx=18, pady=(2, 14), anchor="w")
        ctk.CTkLabel(
            box, text=label,
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color=fg_pair,
            anchor="w",
        ).pack(fill="x", padx=14, pady=(10, 0))
        ctk.CTkLabel(
            box, text=text,
            font=ctk.CTkFont(size=12),
            text_color=fg_pair,
            anchor="w", justify="left",
            wraplength=self.CONTENT_WRAP - 28,
        ).pack(fill="x", padx=14, pady=(2, 10))

    def _render_compare(self, pairs: list):
        """A vertical list of (label, value) comparison rows.
        Bad/Good labels get red/green badges; others get accent-color."""
        for label, value in pairs:
            row = ctk.CTkFrame(self.content_scroll, fg_color="transparent")
            row.pack(fill="x", padx=18, pady=(0, 6), anchor="w")

            lower = label.lower()
            if lower == "bad":
                badge_bg = ("#fee2e2", "#7f1d1d")
                badge_fg = ("#b91c1c", "#fca5a5")
            elif lower == "good":
                badge_bg = ("#dcfce7", "#14532d")
                badge_fg = ("#166534", "#86efac")
            else:
                badge_bg = ACCENT_COLOR
                badge_fg = ("white", "white")

            badge = ctk.CTkFrame(row, corner_radius=4, fg_color=badge_bg)
            badge.pack(side="left", padx=(0, 10), anchor="n", pady=(1, 0))
            ctk.CTkLabel(
                badge, text=label.upper(),
                font=ctk.CTkFont(size=10, weight="bold"),
                text_color=badge_fg,
            ).pack(padx=8, pady=3)

            ctk.CTkLabel(
                row, text=value,
                font=ctk.CTkFont(family="Consolas", size=12),
                text_color=("gray25", "gray85"),
                anchor="w", justify="left",
                wraplength=self.CONTENT_WRAP - 100,
            ).pack(side="left", fill="x", expand=True, anchor="n", pady=(1, 0))

    def _render_kv(self, pairs: list):
        """Aligned reference table: monospace key in accent color, then value."""
        for key, value in pairs:
            row = ctk.CTkFrame(self.content_scroll, fg_color="transparent")
            row.pack(fill="x", padx=18, pady=(0, 4), anchor="w")
            ctk.CTkLabel(
                row, text=key,
                font=ctk.CTkFont(family="Consolas", size=12, weight="bold"),
                text_color=ACCENT_COLOR,
                width=120, anchor="w",
            ).pack(side="left", padx=(4, 16))
            ctk.CTkLabel(
                row, text=value,
                font=ctk.CTkFont(size=13),
                text_color=("gray20", "gray85"),
                anchor="w",
            ).pack(side="left", fill="x", expand=True)


# ---------------------------------------------------------------------------
# App icon
# ---------------------------------------------------------------------------

def _make_app_icon():
    """Generate a small app icon as a PIL Image. None on failure."""
    try:
        from PIL import Image, ImageDraw
        size = 64
        img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        colors = [(110, 180, 220, 255), (180, 110, 220, 255), (230, 160, 90, 255)]
        for i, color in enumerate(colors):
            o = i * 7
            d.rounded_rectangle(
                [6 + o, 6 + o, 42 + o, 42 + o],
                radius=7, fill=color,
                outline=(35, 35, 40, 255), width=2,
            )
        return img
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")

    root = ctk.CTk()
    try:
        icon_img = _make_app_icon()
        if icon_img is not None:
            from PIL import ImageTk
            icon_photo = ImageTk.PhotoImage(icon_img)
            root.iconphoto(True, icon_photo)
            root._icon_ref = icon_photo  # prevent GC
    except Exception:
        pass

    WallpaperOrganizerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()