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

import json
import shutil
import threading
import time
from collections import deque
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
    ("Landscapes", "a scenic landscape photograph with mountains, valleys, beaches, or open vistas"),
    ("Nature",     "a close-up nature photograph of plants, flowers, forests, or wildlife"),
    ("Cities",     "a cityscape photograph showing buildings, skylines, streets, or urban architecture"),
    ("Space",      "a photograph of stars, galaxies, planets, nebulae, or outer space"),
    ("Anime",      "anime or manga style illustrated artwork featuring a character"),
    ("Portraits",  "a portrait photograph of a real person"),
    ("Cars",       "a photograph of cars, motorcycles, or other vehicles"),
    ("Abstract",   "abstract art, geometric patterns, or digital designs"),
    ("Minimalist", "a minimalist wallpaper with simple shapes, gradients, or solid colors"),
    ("Gaming",     "a screenshot or artwork from a video game"),
]

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
        "subtitle": "Get sorting in six steps.",
        "blocks": [
            ("steps", [
                "Pick a SOURCE folder — where your wallpapers currently live.",
                "Pick a DESTINATION folder — where the sorted subfolders will go.",
                "Edit the categories. Each has a folder name and a CLIP prompt describing what should land there. Double-click any row to edit. Add creates new categories, Reset brings defaults back.",
                'Keep "Dry run" CHECKED for the first run — shows what would happen without moving files.',
                "Click Organize. Cards will stream into the log feed as images are classified.",
                'If the totals at the top of the log look right, uncheck "Dry run" and run again — this time files actually move (or copy).',
            ]),
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
        "title": "Reading the Log",
        "subtitle": "What the card feed is showing you.",
        "blocks": [
            ("p", "Each card represents one image classification:"),
            ("code", "●  sunset_city.jpg          vs Landscapes 0.31      [Cities]  0.42"),
            ("ul", [
                "● — confidence dot. Green = clear winner. Yellow = borderline. Red = below threshold or close call.",
                "Filename of the classified image.",
                '"vs Landscapes 0.31" — the runner-up category and its score. Close runners-up signal coin-flip classifications worth reviewing.',
                "0.42 — the winner's confidence score.",
                "[Cities] — the category badge, color-coded.",
            ]),
            ("h", "Confidence guide"),
            ("kv", [
                ("0.50+",  "very confident"),
                ("0.20+",  "reasonably confident"),
                ("< 0.10", "essentially a guess"),
            ]),
            ("tip", "Click any card to preview the image alongside its full score breakdown. Use the filter dropdown to focus on one category at a time."),
        ],
    },
    {
        "title": "Threshold Tuning",
        "subtitle": "Balancing precision against coverage.",
        "blocks": [
            ("p", 'Anything scoring BELOW the threshold is dumped into an "Unsorted" folder instead of being forced into a wrong category.'),
            ("h", "Rule of thumb"),
            ("p", "With N categories, set threshold roughly 1/N, then tune based on what you see in the log."),
            ("kv", [
                ("10 categories", "threshold around 0.15"),
                ("5 categories",  "threshold around 0.25"),
                ("3 categories",  "threshold around 0.10"),
            ]),
            ("tip", "3-category setups want LOWER than 1/N (~0.10) — real winners in 3-way contests still hover near 0.40, and you don't want to dump them all to Unsorted."),
            ("h", "Tuning"),
            ("ul", [
                'If too much ends up in "Unsorted" — lower the threshold.',
                "If wrong-folder mistakes are creeping in — raise it.",
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
                "Dry-run with COPY mode and a low-ish threshold.",
                "Look at the totals in the counter strip. Does the spread feel right?",
                "Spot-check cards for close calls (look at runner-up scores).",
                "Click suspicious cards to preview the images and verify CLIP's calls.",
                "Tighten any prompt that's misfiring. Run dry again.",
                "When it looks good: uncheck Dry run, click Organize, files actually move.",
                'Hit "Open Destination" to inspect the sorted folders.',
            ]),
            ("h", "Copy vs Move"),
            ("ul", [
                "COPY (default) leaves originals in place. Safer. Recommended for first real runs.",
                "MOVE relocates the files. Use this once you trust the categories.",
            ]),
            ("p", "Either way, files already inside the destination folder are skipped on re-runs, so you can run repeatedly without double-organizing."),
        ],
    },
    {
        "title": "Common Gotchas",
        "subtitle": "Things that surprise people on first use.",
        "blocks": [
            ("warn", "First run downloads ~600 MB of model weights from HuggingFace. Wait a minute. Subsequent runs use the cached version."),
            ("ul", [
                "With batching, CPU classification runs ~0.1-0.3 sec per image. A few hundred images takes well under a minute.",
                'Borderline images often have legitimate ambiguity (a city at sunset can fairly belong to either "Cities" or "Landscapes"). Don\'t try to make CLIP perfect — for the last few percent, just drag-drop manually.',
                "Settings persist in ~/.wallpaper_organizer.json. Delete that file to reset everything.",
            ]),
            ("h", "Supported file types"),
            ("code", ".jpg, .jpeg, .png, .webp, .bmp, .gif, .tiff, .jfif"),
            ("p", "Subfolders inside the source are scanned recursively."),
        ],
    },
]


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

    def set_categories(self, categories: list[tuple[str, str]]) -> None:
        self._categories = list(categories)
        prompts = [prompt for _, prompt in self._categories]
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


class OrganizerJob:
    """Walks the source dir, classifies each image, copies/moves to dest."""

    def __init__(
        self,
        source: Path,
        dest: Path,
        categories: list[tuple[str, str]],
        copy_mode: bool,
        threshold: float,
        dry_run: bool,
        log_queue: Queue,
        cancel_event: threading.Event,
    ):
        self.source = source
        self.dest = dest
        self.categories = categories
        self.copy_mode = copy_mode
        self.threshold = threshold
        self.dry_run = dry_run
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

            images = [
                p for p in self.source.rglob("*")
                if p.is_file()
                and p.suffix.lower() in SUPPORTED_EXTS
                and self.dest not in p.parents
            ]
            total = len(images)
            self._emit("log", f"Found {total} image(s) to classify.")
            if total == 0:
                self._emit("status", "Nothing to do — no images found.")
                self._emit("done", {"by_cat": {}, "errors": 0,
                                    "dry_run": self.dry_run, "elapsed": 0})
                return

            counts: dict[str, int] = {name: 0 for name, _ in self.categories}
            counts["Unsorted"] = 0
            errors = 0
            timings: deque = deque(maxlen=50)
            start = time.monotonic()
            processed = 0

            # Process in batches of 8 — on CPU this is roughly the sweet spot.
            BATCH_SIZE = 8
            t_batch_start = time.monotonic()

            for img_path, ranked, err in classifier.classify_batch(
                images, top_k=2, batch_size=BATCH_SIZE
            ):
                if self.cancel_event.is_set():
                    self._emit("log", "Cancelled.")
                    break

                processed += 1
                try:
                    if err is not None:
                        errors += 1
                        self._emit("log", f"ERROR on {img_path.name}: {err}")
                    else:
                        name, conf = ranked[0]
                        runner_up = ranked[1] if len(ranked) > 1 else None
                        if conf < self.threshold:
                            name = "Unsorted"
                        target_dir = self.dest / name
                        target_dir.mkdir(parents=True, exist_ok=True)
                        target = target_dir / img_path.name
                        j = 1
                        while target.exists() and target.resolve() != img_path.resolve():
                            target = target_dir / f"{img_path.stem}_{j}{img_path.suffix}"
                            j += 1
                        action = "copy" if self.copy_mode else "move"
                        runner_up_str = (
                            f"  [also: {runner_up[0]} {runner_up[1]:.2f}]" if runner_up else ""
                        )
                        if not self.dry_run:
                            if self.copy_mode:
                                shutil.copy2(img_path, target)
                            else:
                                shutil.move(str(img_path), str(target))
                        prefix = f"[dry-run] {action} " if self.dry_run else ""
                        msg = f"{prefix}{img_path.name} -> {name}/  ({conf:.2f}){runner_up_str}"

                        # Path that exists *after* the action — used by the
                        # click-to-preview feature. In dry-run nothing moved;
                        # in move mode the original is gone, only target exists.
                        if self.dry_run or self.copy_mode:
                            preview_path = str(img_path)
                        else:
                            preview_path = str(target)

                        self._emit("log_clickable", {
                            "filename": img_path.name,
                            "path": preview_path,
                            "category": name,
                            "confidence": conf,
                            "ranked": ranked,
                            "is_dry_run": self.dry_run,
                            "action": action,
                            # Keep "text" too for any consumer that wants a
                            # one-line fallback representation.
                            "text": msg,
                        })
                        counts[name] = counts.get(name, 0) + 1
                except Exception as e:
                    errors += 1
                    self._emit("log", f"ERROR on {img_path.name}: {e}")

                self._emit("progress", (processed, total))

                # Update timings + ETA at batch boundaries (every BATCH_SIZE
                # images) and at the end. Timing per-image during a batch is
                # noisy; per-batch averages give a much smoother ETA.
                if processed % BATCH_SIZE == 0 or processed == total:
                    batch_elapsed = time.monotonic() - t_batch_start
                    timings.append(batch_elapsed / max(1, BATCH_SIZE))
                    t_batch_start = time.monotonic()
                    avg = sum(timings) / len(timings)
                    remaining_sec = (total - processed) * avg
                    eta = _format_eta(remaining_sec)
                    self._emit("status", f"Classifying — {processed}/{total}, {eta}")

            elapsed = time.monotonic() - start
            self._emit("done", {
                "by_cat": counts,
                "errors": errors,
                "dry_run": self.dry_run,
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
        self.dry_run_var = tk.BooleanVar(value=True)
        self.threshold_var = tk.DoubleVar(value=0.15)
        self.status_var = tk.StringVar(value="Ready")
        self.cancel_event = threading.Event()
        self.log_queue: Queue = Queue()
        self.worker: Optional[threading.Thread] = None
        self._loaded_cats: Optional[list[tuple[str, str]]] = None

        # --- Log feed state ---
        # Each row is a dict: {widget, is_text, category, meta}
        # is_text=True for plain header/footer lines that always show.
        # category is the category name for filtering. meta is the click payload.
        self._log_rows: list[dict] = []
        # Color per category, computed when a run starts.
        # Maps category name -> (light_bg, dark_bg, light_fg, dark_fg) tuple.
        self._category_colors: dict[str, tuple] = {}
        # Live counter labels, one per category (including Unsorted).
        self._counter_labels: dict[str, ctk.CTkLabel] = {}
        # Live counts (updated as rows arrive).
        self._counts: dict[str, int] = {}
        # Current filter selection ("All" or a category name).
        self.filter_var = tk.StringVar(value="All")
        # Persistent ref to the preview window so opening a new one closes the old.
        self._preview: Optional["PreviewDialog"] = None

        self._load_config()
        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
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
            "Categories  (folder name → CLIP prompt — double-click to edit)",
        )

        wrap = ctk.CTkFrame(section, fg_color="transparent")
        wrap.pack(fill="both", expand=True, padx=12, pady=(8, 12))

        # Treeview + scrollbar in a card-style frame
        tree_card = ctk.CTkFrame(wrap, fg_color=("gray85", "gray17"), corner_radius=6)
        tree_card.pack(side="left", fill="both", expand=True)

        self.cat_tree = ttk.Treeview(
            tree_card, columns=("name", "prompt"), show="headings",
            height=8, style="WO.Treeview",
        )
        self.cat_tree.heading("name", text="Folder")
        self.cat_tree.heading("prompt", text="CLIP prompt")
        self.cat_tree.column("name", width=170, anchor="w")
        self.cat_tree.column("prompt", width=560, anchor="w")
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

        for name, prompt in (self._loaded_cats or DEFAULT_CATEGORIES):
            self.cat_tree.insert("", "end", values=(name, prompt))

    def _build_options(self, parent):
        section = self._section_frame(parent, "Options")

        ctk.CTkCheckBox(
            section, text="Copy files (uncheck to move)",
            variable=self.copy_var,
        ).grid(row=0, column=0, sticky="w", padx=14, pady=(10, 4))
        ctk.CTkCheckBox(
            section, text="Dry run (preview only — no file changes)",
            variable=self.dry_run_var,
        ).grid(row=0, column=1, sticky="w", padx=20, pady=(10, 4))

        ctk.CTkLabel(section, text="Confidence threshold:")\
            .grid(row=1, column=0, sticky="w", padx=14, pady=(4, 12))
        slider_wrap = ctk.CTkFrame(section, fg_color="transparent")
        slider_wrap.grid(row=1, column=1, sticky="w", padx=20, pady=(4, 12))

        self.thresh_slider = ctk.CTkSlider(
            slider_wrap, from_=0.0, to=0.9,
            variable=self.threshold_var, width=240,
        )
        self.thresh_slider.pack(side="left")
        self.thresh_label = ctk.CTkLabel(
            slider_wrap,
            text=f"{self.threshold_var.get():.2f}",
            width=44, anchor="w",
            font=ctk.CTkFont(size=13, weight="bold"),
        )
        self.thresh_label.pack(side="left", padx=(10, 0))
        self.threshold_var.trace_add(
            "write",
            lambda *a: self.thresh_label.configure(text=f"{self.threshold_var.get():.2f}"),
        )

    def _build_run_row(self, parent):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", pady=(0, 10))

        # Primary action
        self.run_btn = ctk.CTkButton(
            row, text="Organize", width=120, height=38,
            command=self._start,
            fg_color=ACCENT_COLOR, hover_color=ACCENT_HOVER,
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        self.run_btn.pack(side="left")

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
        # also hosts the filter dropdown.
        wrap = ctk.CTkFrame(parent, fg_color="transparent")
        wrap.pack(fill="both", expand=True, pady=(0, 10))

        title_row = ctk.CTkFrame(wrap, fg_color="transparent")
        title_row.pack(fill="x", padx=4, pady=(0, 4))

        ctk.CTkLabel(
            title_row, text="Log  (click a card to preview the image)",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=("gray35", "gray70"),
            anchor="w",
        ).pack(side="left")

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

        # Scrollable feed for log rows
        self.log_frame = ctk.CTkScrollableFrame(section, corner_radius=6)
        self.log_frame.pack(fill="both", expand=True, padx=12, pady=(0, 12))

    # ---- Log feed helpers ----

    def _compute_category_colors(self, categories):
        """Assign a stable color from CATEGORY_PALETTE to each category."""
        colors: dict[str, tuple] = {}
        for i, (name, _) in enumerate(categories):
            colors[name] = CATEGORY_PALETTE[i % len(CATEGORY_PALETTE)]
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
        for name, _ in categories:
            self._add_counter_chip(name)
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
        options = ["All"] + [name for name, _ in categories] + ["Unsorted"]
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
        new_mode = "light" if ctk.get_appearance_mode().lower() == "dark" else "dark"
        ctk.set_appearance_mode(new_mode)
        self._update_mode_btn()
        self._setup_treeview_style()
        self._restyle_log_rows()

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
        for name, prompt in DEFAULT_CATEGORIES:
            self.cat_tree.insert("", "end", values=(name, prompt))

    def _cat_dialog(self, item_id):
        d = ctk.CTkToplevel(self.root)
        d.title("Category")
        d.geometry("540x300")
        d.transient(self.root)
        d.resizable(False, False)
        d.after(80, d.lift)
        d.after(120, d.focus_force)
        d.after(140, d.grab_set)

        body = ctk.CTkFrame(d, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=20, pady=20)

        ctk.CTkLabel(
            body, text="Folder name", anchor="w",
            font=ctk.CTkFont(size=12, weight="bold"),
        ).pack(fill="x")
        name_var = tk.StringVar()
        ctk.CTkEntry(body, textvariable=name_var, height=34)\
            .pack(fill="x", pady=(4, 12))

        ctk.CTkLabel(
            body, text="CLIP prompt  (describe what should land here)", anchor="w",
            font=ctk.CTkFont(size=12, weight="bold"),
        ).pack(fill="x")
        prompt_text = ctk.CTkTextbox(body, height=90)
        prompt_text.pack(fill="x", pady=(4, 14))

        if item_id is not None:
            vals = self.cat_tree.item(item_id, "values")
            name_var.set(vals[0])
            prompt_text.insert("1.0", vals[1])

        def save():
            n = name_var.get().strip()
            p = prompt_text.get("1.0", "end").strip()
            if not n or not p:
                messagebox.showerror("Missing fields", "Both fields are required.", parent=d)
                return
            n = "".join(c for c in n if c.isalnum() or c in " _-").strip()
            if item_id is None:
                self.cat_tree.insert("", "end", values=(n, p))
            else:
                self.cat_tree.item(item_id, values=(n, p))
            d.destroy()

        btns = ctk.CTkFrame(body, fg_color="transparent")
        btns.pack(fill="x")
        ctk.CTkButton(
            btns, text="Save", width=100, height=34, command=save,
            fg_color=ACCENT_COLOR, hover_color=ACCENT_HOVER,
        ).pack(side="right", padx=(8, 0))
        self._ghost_button(btns, "Cancel", d.destroy, width=100, height=34)\
            .pack(side="right")

    def _get_categories(self) -> list[tuple[str, str]]:
        return [tuple(self.cat_tree.item(i, "values")) for i in self.cat_tree.get_children()]

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
        self.cancel_event.clear()
        self.run_btn.configure(state="disabled")
        self.cancel_btn.configure(state="normal")
        self.progress.set(0)
        self.progress_label.configure(text="")

        # Set up colors + counter strip + filter dropdown for this run's categories.
        # Order matters: compute colors before building the strip (chips use them),
        # before clearing the log (chips reference old colors otherwise).
        self._compute_category_colors(cats)
        self._clear_log()
        self._build_counter_strip(cats)
        self._refresh_filter_options(cats)

        self.status_var.set("Starting…")

        job = OrganizerJob(
            source=Path(source),
            dest=Path(dest),
            categories=cats,
            copy_mode=self.copy_var.get(),
            threshold=self.threshold_var.get(),
            dry_run=self.dry_run_var.get(),
            log_queue=self.log_queue,
            cancel_event=self.cancel_event,
        )
        self.worker = threading.Thread(target=job.run, daemon=True)
        self.worker.start()

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
        def on_click(_event, meta=payload):
            self._open_preview(meta)

        for widget in (row, dot_canvas, badge, badge_label,
                       score_label, runner_label, filename_label):
            widget.bind("<Button-1>", on_click)

        self._log_rows.append({
            "widget": row, "is_text": False,
            "category": category, "meta": payload,
            # Keep refs to per-row tk widgets for theme refresh
            "tk_widgets": {
                "row": row, "dot": dot_canvas, "badge": badge,
                "badge_label": badge_label, "score": score_label,
                "runner": runner_label, "filename": filename_label,
                "badge_bg_pair": bg_pair,
            },
        })

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
                elif kind == "log_clickable":
                    self._append_log_row(payload)
                elif kind == "progress":
                    cur, total = payload
                    self.progress.set(cur / total if total else 0)
                    self.progress_label.configure(text=f"{cur}/{total}")
                elif kind == "status":
                    self.status_var.set(payload)
                elif kind == "done":
                    self._on_done(payload)
                elif kind == "error":
                    self._append_log_text(f"FATAL: {payload}", kind="error")
                    self.status_var.set(f"Error: {payload}")
                    messagebox.showerror("Error", payload)
                    self._reset_buttons()
        except Empty:
            pass
        self.root.after(100, self._poll_queue)

    def _on_done(self, summary: dict):
        self._append_log_text("")
        self._append_log_text("=== Done ===", kind="header")
        non_empty = 0
        for name, count in summary.get("by_cat", {}).items():
            if count:
                self._append_log_text(f"  {name}: {count}")
                non_empty += 1
        if summary.get("errors"):
            self._append_log_text(f"  Errors: {summary['errors']}", kind="error")
        if summary.get("dry_run"):
            self._append_log_text("(dry run — no files were changed)")
        elapsed = summary.get("elapsed", 0)
        kind = "Dry run complete" if summary.get("dry_run") else "Done"
        if non_empty == 0:
            self.status_var.set(f"{kind} — nothing was sorted (check the log)")
        else:
            self.status_var.set(
                f"{kind} — {non_empty} folder(s) in {_format_elapsed(elapsed)}"
            )
        self._reset_buttons()

    def _reset_buttons(self):
        self.run_btn.configure(state="normal")
        self.cancel_btn.configure(state="disabled")

    # ---- Config ----

    def _load_config(self):
        if not CONFIG_FILE.exists():
            return
        try:
            cfg = json.loads(CONFIG_FILE.read_text())
            self.source_var.set(cfg.get("source", ""))
            self.dest_var.set(cfg.get("dest", ""))
            self.copy_var.set(cfg.get("copy", True))
            self.dry_run_var.set(cfg.get("dry_run", True))
            self.threshold_var.set(cfg.get("threshold", 0.15))
            cats = cfg.get("categories")
            if cats:
                self._loaded_cats = [tuple(c) for c in cats]
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
                "dry_run": self.dry_run_var.get(),
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