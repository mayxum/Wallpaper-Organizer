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


# ---------------------------------------------------------------------------
# Help content - sectioned for the navigable Help dialog
# ---------------------------------------------------------------------------

HELP_SECTIONS: list[tuple[str, str]] = [
    ("🚀  Quick Start", """\
QUICK START

1. Pick a SOURCE folder — where your wallpapers currently live.

2. Pick a DESTINATION folder — where the sorted subfolders will go.

3. Edit the categories. Each one has a folder name and a CLIP prompt
   that describes what should land there. Double-click any row to edit
   it. Hit Add to create new categories, Reset to bring the defaults
   back.

4. Keep "Dry run" CHECKED for the first run. It shows what WOULD happen
   without moving any files.

5. Click Organize. Watch the log scroll by.

6. If the totals look right at the bottom, uncheck "Dry run" and run
   again — this time the files actually move (or copy).
"""),

    ("🧠  How CLIP Works", """\
HOW CLIP WORKS (PLAIN-ENGLISH VERSION)

CLIP is an AI model that understands both images and text in the same
"language." For each image, it scores how well the picture matches
every one of your category prompts. The scores add up to 1.00 across
all categories. The category with the highest score wins.

What this means in practice:

   • The PROMPT matters more than the folder NAME. CLIP doesn't read
     folder names — only the prompt text.

   • Better-described prompts = better sorting.

   • Two visually similar categories will compete for ambiguous
     images. That's why contrastive language ("an anime ILLUSTRATED
     character" vs "a PHOTOGRAPH of a real person") works well.

   • CLIP isn't perfect — for the last few percent, your eye is faster
     than tweaking prompts forever.
"""),

    ("📊  Reading the Log", """\
READING THE LOG

Each line looks like:

    sunset_city.jpg -> Cities/  (0.42)  [also: Landscapes 0.31]

   • "Cities" is the winning category.
   • "0.42" is the confidence — how sure CLIP was.
   • The "[also: ...]" part is the runner-up. If the winner and
     runner-up are CLOSE (e.g. 0.36 vs 0.34), the classification was
     a coin flip. These close calls are gold for spotting prompts
     that need sharpening.

CONFIDENCE GUIDE

   • 0.50+   very confident
   • 0.20+   reasonably confident
   • <0.10   essentially a guess
"""),

    ("🎯  Threshold Tuning", """\
THE CONFIDENCE THRESHOLD

Anything scoring BELOW the threshold is dumped into an "Unsorted"
folder instead of being forced into a wrong category.

RULE OF THUMB

With N categories, set threshold roughly 1/N, then tune based on what
you see in the log:

   • 10 categories  →  threshold around 0.15
   •  5 categories  →  threshold around 0.25
   •  3 categories  →  threshold around 0.10  (lower than 1/N, because
                       real winners in 3-way contests still hover near
                       0.40 and you don't want to dump them all to
                       Unsorted)

TUNING

   • If too much ends up in "Unsorted"  →  lower the threshold
   • If wrong-folder mistakes are creeping in  →  raise it
"""),

    ("✏️  Writing Prompts", """\
WRITING GOOD CATEGORY PROMPTS

The prompt is the text CLIP compares each image against. Think of it
like an image caption.

TIPS

   • Be descriptive, not just keywords.
        BAD:  "landscapes"
        GOOD: "a scenic landscape photograph with mountains, valleys,
               and open vistas"

   • Mention what makes the category visually distinct.

   • If a category is OVER-firing (grabbing too much), make the
     prompt more specific. Add visual qualifiers to narrow it.

   • If a category is UNDER-firing, broaden the description.

   • To distinguish two similar categories (e.g. anime vs photo
     portraits), CONTRAST them in their prompts:
        "an anime ILLUSTRATED character"
        vs
        "a PHOTOGRAPH of a real person"

   • Include "no people" in scenery prompts if you want to exclude
     human subjects from that bucket.
"""),

    ("📋  Workflow", """\
TYPICAL WORKFLOW

   1.  Set up your categories with descriptive prompts.

   2.  Dry-run with COPY mode and a low-ish threshold.

   3.  Look at the totals at the bottom of the log. Does the spread
       feel right?

   4.  Spot-check the log for close calls (look at the runner-up
       scores).

   5.  Tighten any prompt that's misfiring. Run dry again.

   6.  Once it looks good: uncheck Dry run, click Organize, files
       actually move.

   7.  Hit "Open Destination" to inspect the sorted folders.

COPY VS MOVE

   • COPY (default) leaves originals in place. Safer. Recommended for
     first real runs.

   • MOVE relocates the files. Use this once you trust the
     categories.

Either way, files already inside the destination folder are skipped
on re-runs, so you can run the tool repeatedly without
double-organizing.
"""),

    ("⚠️  Common Gotchas", """\
COMMON GOTCHAS

   • First run downloads ~600 MB of model weights. Wait a minute.
     Subsequent runs use the cached version.

   • On CPU, classification takes ~1–2 seconds per image. A few
     hundred images is a coffee break.

   • Borderline images often have legitimate ambiguity (e.g. a city
     at sunset can fairly belong to either "Cities" or "Landscapes").
     Don't try to make CLIP perfect — for the last few percent, just
     drag-drop manually.

   • Settings persist in ~/.wallpaper_organizer.json. Delete that
     file to reset everything.

SUPPORTED FILE TYPES

   .jpg, .jpeg, .png, .webp, .bmp, .gif, .tiff, .jfif

Subfolders inside the source are scanned recursively.
"""),
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
            timings: deque = deque(maxlen=20)
            start = time.monotonic()

            for i, img_path in enumerate(images, 1):
                if self.cancel_event.is_set():
                    self._emit("log", "Cancelled.")
                    break
                t0 = time.monotonic()
                try:
                    ranked = classifier.classify(img_path, top_k=2)
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
                    self._emit(
                        "log",
                        f"{prefix}{img_path.name} -> {name}/  ({conf:.2f}){runner_up_str}",
                    )
                    counts[name] = counts.get(name, 0) + 1
                except Exception as e:
                    errors += 1
                    self._emit("log", f"ERROR on {img_path.name}: {e}")

                timings.append(time.monotonic() - t0)
                self._emit("progress", (i, total))

                # Update status with rolling-average ETA
                if i == 1 or i % 3 == 0 or i == total:
                    avg = sum(timings) / len(timings)
                    remaining_sec = (total - i) * avg
                    eta = _format_eta(remaining_sec)
                    self._emit("status", f"Classifying — {i}/{total}, {eta}")

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

        self._load_config()
        self._build_ui()
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
        section = self._section_frame(parent, "Log")
        self.log = ctk.CTkTextbox(
            section, wrap="word",
            font=ctk.CTkFont(family="Consolas", size=12),
            corner_radius=6,
        )
        self.log.pack(fill="both", expand=True, padx=12, pady=(8, 12))
        self.log.configure(state="disabled")

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
        self._clear_log()
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

    # ---- Log/queue ----

    def _clear_log(self):
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    def _append_log(self, msg: str):
        self.log.configure(state="normal")
        self.log.insert("end", msg + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _poll_queue(self):
        try:
            while True:
                kind, payload = self.log_queue.get_nowait()
                if kind == "log":
                    self._append_log(payload)
                elif kind == "progress":
                    cur, total = payload
                    self.progress.set(cur / total if total else 0)
                    self.progress_label.configure(text=f"{cur}/{total}")
                elif kind == "status":
                    self.status_var.set(payload)
                elif kind == "done":
                    self._on_done(payload)
                elif kind == "error":
                    self._append_log(f"FATAL: {payload}")
                    self.status_var.set(f"Error: {payload}")
                    messagebox.showerror("Error", payload)
                    self._reset_buttons()
        except Empty:
            pass
        self.root.after(100, self._poll_queue)

    def _on_done(self, summary: dict):
        self._append_log("")
        self._append_log("=== Done ===")
        non_empty = 0
        for name, count in summary.get("by_cat", {}).items():
            if count:
                self._append_log(f"  {name}: {count}")
                non_empty += 1
        if summary.get("errors"):
            self._append_log(f"  Errors: {summary['errors']}")
        if summary.get("dry_run"):
            self._append_log("(dry run — no files were changed)")
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
            }, indent=2))
        except Exception:
            pass

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
# Sectioned Help dialog with sidebar navigation
# ---------------------------------------------------------------------------

class HelpDialog:
    def __init__(self, parent: ctk.CTk):
        self.win = ctk.CTkToplevel(parent)
        self.win.title("Wallpaper Organizer — Help & Tips")
        self.win.geometry("860x620")
        self.win.minsize(740, 520)
        self.win.transient(parent)
        self.win.after(80, self.win.lift)
        self.win.after(120, self.win.focus_force)

        body = ctk.CTkFrame(self.win, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=14, pady=(14, 6))

        # Left sidebar: section list
        sidebar = ctk.CTkFrame(body, width=210, corner_radius=8)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        ctk.CTkLabel(
            sidebar, text="SECTIONS",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=("gray45", "gray55"),
            anchor="w",
        ).pack(fill="x", padx=14, pady=(14, 6))

        self.section_buttons: list[ctk.CTkButton] = []
        for i, (title, _content) in enumerate(HELP_SECTIONS):
            btn = ctk.CTkButton(
                sidebar, text=title, anchor="w",
                width=190, height=36,
                fg_color="transparent",
                text_color=("gray15", "gray85"),
                hover_color=("gray80", "gray22"),
                font=ctk.CTkFont(size=13),
                command=lambda idx=i: self._show_section(idx),
            )
            btn.pack(fill="x", padx=10, pady=2)
            self.section_buttons.append(btn)

        # Right: content area
        content_card = ctk.CTkFrame(body, corner_radius=8)
        content_card.pack(side="left", fill="both", expand=True, padx=(12, 0))

        self.heading = ctk.CTkLabel(
            content_card, text="",
            font=ctk.CTkFont(size=18, weight="bold"),
            anchor="w",
        )
        self.heading.pack(fill="x", padx=18, pady=(16, 4))

        self.content = ctk.CTkTextbox(
            content_card, wrap="word",
            font=ctk.CTkFont(size=13),
            corner_radius=6,
            fg_color="transparent",
        )
        self.content.pack(fill="both", expand=True, padx=14, pady=(0, 14))

        # Footer
        footer = ctk.CTkFrame(self.win, fg_color="transparent")
        footer.pack(fill="x", padx=14, pady=(6, 14))
        ctk.CTkButton(
            footer, text="Close", width=100, height=34,
            command=self.win.destroy,
            fg_color=ACCENT_COLOR, hover_color=ACCENT_HOVER,
        ).pack(side="right")

        self._show_section(0)

    def _show_section(self, idx: int):
        # Highlight chosen sidebar button
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
        title, body = HELP_SECTIONS[idx]
        plain = title.split(" ", 1)[-1].strip() if " " in title else title
        self.heading.configure(text=plain)
        self.content.configure(state="normal")
        self.content.delete("1.0", "end")
        self.content.insert("1.0", body)
        self.content.configure(state="disabled")


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
