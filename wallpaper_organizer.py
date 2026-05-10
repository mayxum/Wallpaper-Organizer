#!/usr/bin/env python3
"""
Wallpaper Organizer
-------------------
Auto-sorts a folder of wallpapers into category subfolders using CLIP
zero-shot image classification. You define the categories with text
prompts; CLIP figures out which images belong where.

Setup:
    pip install torch transformers pillow

Run:
    python wallpaper_organizer.py
"""
from __future__ import annotations

import json
import shutil
import threading
from pathlib import Path
from queue import Queue, Empty
from typing import Optional

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext


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


HELP_TEXT = """\
WALLPAPER ORGANIZER — HELP & TIPS

Auto-sorts a folder of wallpapers into category subfolders using AI image
recognition (CLIP). You describe each category in plain English; CLIP scores
every image against every category and picks the best match.


QUICK START

1. Pick a SOURCE folder (where your wallpapers currently live).
2. Pick a DESTINATION folder (where the sorted subfolders will go).
3. Edit the categories. Each one has a Folder name and a CLIP prompt that
   describes what should go there. Double-click any row to edit it. Hit Add
   to create new ones, Reset to bring the defaults back.
4. Keep "Dry run" CHECKED for the first run. It shows what WOULD happen
   without moving any files.
5. Click Organize. Watch the log.
6. If the totals look right, uncheck "Dry run" and run again — this time the
   files actually move (or copy).


HOW CLIP WORKS (PLAIN-ENGLISH VERSION)

CLIP is an AI model that understands both images and text in the same
"language." For each image, it scores how well the picture matches every
one of your category prompts. The scores add up to 1.00 across all
categories. The category with the highest score wins.


READING THE LOG

Each line looks like:

    sunset_city.jpg -> Cities/  (0.42)  [also: Landscapes 0.31]

   - "Cities" is the winning category.
   - "0.42" is the confidence — how sure CLIP was.
   - The "[also: ...]" part is the runner-up. If the winner and runner-up
     are CLOSE (e.g. 0.36 vs 0.34), the classification was a coin flip.
     Use these close calls to spot prompts that need sharpening.

Confidence guide:
   - 0.50+   very confident
   - 0.20+   reasonably confident
   - <0.10   essentially a guess


THE THRESHOLD

Anything scoring BELOW the threshold is dumped into an "Unsorted" folder
instead of being forced into a wrong category.

Rule of thumb: with N categories, set threshold roughly equal to 1/N, then
tune from the log.

   - 10 categories -> threshold around 0.15
   -  5 categories -> threshold around 0.25
   -  3 categories -> threshold around 0.10 (lower than 1/N, because real
                       winners in 3-way contests still hover near 0.40 and
                       you don't want to dump them all to Unsorted)

If too much ends up in "Unsorted", lower the threshold.
If wrong-folder mistakes are creeping in, raise it.


WRITING GOOD CATEGORY PROMPTS

The PROMPT matters more than the folder NAME. CLIP doesn't read your folder
names — it only compares the image to the prompt text.

Tips:
   - Be descriptive, not just keywords.
       BAD:  "landscapes"
       GOOD: "a scenic landscape photograph with mountains, valleys, or
              open vistas"
   - Mention what makes the category visually distinct.
   - If a category is OVER-firing (grabbing too much), make the prompt
     more specific. Add visual qualifiers.
   - If a category is UNDER-firing, broaden the description.
   - To distinguish two similar categories (e.g. anime vs photo portraits),
     contrast them in their prompts: "an anime ILLUSTRATED character" vs
     "a PHOTOGRAPH of a real person."


COPY VS MOVE

   - COPY (default) leaves originals in place. Safer. Recommended for first
     real runs.
   - MOVE relocates the files. Use this once you trust the categories.

Either way, files already inside the destination folder are skipped on
re-runs, so you can run the tool repeatedly without double-organizing.


TYPICAL WORKFLOW

   1. Set up your categories with descriptive prompts.
   2. Dry run with COPY mode and a low-ish threshold.
   3. Look at the totals at the bottom of the log. Does the spread feel
      right?
   4. Spot-check the log for close calls (look for tight runner-up scores).
   5. Tighten any prompt that's misfiring. Run dry again.
   6. Once it looks good: uncheck Dry run, click Organize, files actually
      move.
   7. Hit "Open Destination" to inspect the sorted folders in Explorer.


COMMON GOTCHAS

   - First run downloads ~600 MB of model weights. Wait a minute.
   - On CPU, classification runs ~1-2 seconds per image. A few hundred
     images is a coffee break.
   - Borderline images often have legitimate ambiguity (e.g. a city at
     sunset can fairly belong to either "Cities" or "Landscapes"). Don't
     try to make CLIP perfect — for the last few percent, just drag-drop
     manually.
   - Settings persist in ~/.wallpaper_organizer.json. Delete that file to
     reset everything.


SUPPORTED FILE TYPES

.jpg, .jpeg, .png, .webp, .bmp, .gif, .tiff, .jfif

Subfolders inside the source are scanned recursively.
"""


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
        """
        Get a feature tensor out of get_text_features / get_image_features.
        transformers 4.x returns a Tensor directly; transformers 5.x returns a
        BaseModelOutputWithPooling whose `pooler_output` is the already-
        projected feature (matches the old tensor behavior, per the v5
        migration guide). Handle both.
        """
        if isinstance(output, self.torch.Tensor):
            return output
        if hasattr(output, "pooler_output") and output.pooler_output is not None:
            return output.pooler_output
        for attr in ("text_embeds", "image_embeds"):
            if hasattr(output, attr) and getattr(output, attr) is not None:
                return getattr(output, attr)
        raise RuntimeError(f"Unexpected CLIP {kind} output type: {type(output)}")

    def set_categories(self, categories: list[tuple[str, str]]) -> None:
        """Pre-compute text embeddings for the category prompts."""
        self._categories = list(categories)
        prompts = [prompt for _, prompt in self._categories]
        inputs = self.processor(text=prompts, return_tensors="pt", padding=True).to(self.device)
        with self.torch.no_grad():
            feats = self._extract_features(self.model.get_text_features(**inputs), "text")
            feats = feats / feats.norm(dim=-1, keepdim=True)
        self._text_features = feats

    def classify(self, image_path: Path, top_k: int = 2) -> list[tuple[str, float]]:
        """Return top-k (category_name, confidence) tuples, highest score first."""
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
# Background job
# ---------------------------------------------------------------------------

class OrganizerJob:
    """Background job that walks the source dir and sorts images."""

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
            self._emit("log", "Loading CLIP model (first run downloads ~600 MB)...")
            classifier = CLIPClassifier()
            self._emit("log", f"Model loaded on {classifier.device.upper()}.")
            classifier.set_categories(self.categories)

            images = [
                p for p in self.source.rglob("*")
                if p.is_file()
                and p.suffix.lower() in SUPPORTED_EXTS
                and self.dest not in p.parents  # skip already-organized files
            ]
            total = len(images)
            self._emit("log", f"Found {total} image(s) to classify.")
            if total == 0:
                self._emit("done", {"by_cat": {}, "errors": 0, "dry_run": self.dry_run})
                return

            counts: dict[str, int] = {name: 0 for name, _ in self.categories}
            counts["Unsorted"] = 0
            errors = 0

            for i, img_path in enumerate(images, 1):
                if self.cancel_event.is_set():
                    self._emit("log", "Cancelled.")
                    break
                try:
                    ranked = classifier.classify(img_path, top_k=2)
                    name, conf = ranked[0]
                    runner_up = ranked[1] if len(ranked) > 1 else None
                    if conf < self.threshold:
                        name = "Unsorted"
                    target_dir = self.dest / name
                    target_dir.mkdir(parents=True, exist_ok=True)
                    target = target_dir / img_path.name
                    # Avoid overwriting existing files.
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
                self._emit("progress", (i, total))

            self._emit("done", {"by_cat": counts, "errors": errors, "dry_run": self.dry_run})
        except Exception as e:
            self._emit("error", str(e))


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

class WallpaperOrganizerApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Wallpaper Organizer")
        self.root.geometry("820x740")

        self.source_var = tk.StringVar()
        self.dest_var = tk.StringVar()
        self.copy_var = tk.BooleanVar(value=True)
        self.dry_run_var = tk.BooleanVar(value=True)
        self.threshold_var = tk.DoubleVar(value=0.15)
        self.cancel_event = threading.Event()
        self.log_queue: Queue = Queue()
        self.worker: Optional[threading.Thread] = None
        self._loaded_cats: Optional[list[tuple[str, str]]] = None

        self._load_config()
        self._build_ui()
        self._poll_queue()

    # ---- UI build ----

    def _build_ui(self):
        # Folders
        folders = ttk.LabelFrame(self.root, text="Folders", padding=10)
        folders.pack(fill="x", padx=10, pady=(10, 5))
        folders.columnconfigure(1, weight=1)

        ttk.Label(folders, text="Source folder:").grid(row=0, column=0, sticky="w")
        ttk.Entry(folders, textvariable=self.source_var).grid(row=0, column=1, sticky="ew", padx=5)
        ttk.Button(folders, text="Browse...", command=self._pick_source).grid(row=0, column=2)

        ttk.Label(folders, text="Destination:").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(folders, textvariable=self.dest_var).grid(row=1, column=1, sticky="ew", padx=5, pady=(6, 0))
        ttk.Button(folders, text="Browse...", command=self._pick_dest).grid(row=1, column=2, pady=(6, 0))

        # Categories editor
        cats = ttk.LabelFrame(self.root, text="Categories  (folder name → CLIP prompt)", padding=10)
        cats.pack(fill="both", expand=False, padx=10, pady=5)

        cats_inner = ttk.Frame(cats)
        cats_inner.pack(side="left", fill="both", expand=True)

        self.cat_tree = ttk.Treeview(cats_inner, columns=("name", "prompt"), show="headings", height=9)
        self.cat_tree.heading("name", text="Folder")
        self.cat_tree.heading("prompt", text="CLIP prompt")
        self.cat_tree.column("name", width=140, anchor="w")
        self.cat_tree.column("prompt", width=560, anchor="w")
        self.cat_tree.pack(side="left", fill="both", expand=True)
        self.cat_tree.bind("<Double-1>", lambda e: self._edit_cat())

        scroll = ttk.Scrollbar(cats_inner, orient="vertical", command=self.cat_tree.yview)
        scroll.pack(side="left", fill="y")
        self.cat_tree.configure(yscrollcommand=scroll.set)

        cat_btns = ttk.Frame(cats)
        cat_btns.pack(side="left", fill="y", padx=(8, 0))
        ttk.Button(cat_btns, text="Add",    command=self._add_cat).pack(fill="x", pady=2)
        ttk.Button(cat_btns, text="Edit",   command=self._edit_cat).pack(fill="x", pady=2)
        ttk.Button(cat_btns, text="Remove", command=self._remove_cat).pack(fill="x", pady=2)
        ttk.Button(cat_btns, text="Reset",  command=self._reset_cats).pack(fill="x", pady=2)

        for name, prompt in (self._loaded_cats or DEFAULT_CATEGORIES):
            self.cat_tree.insert("", "end", values=(name, prompt))

        # Options
        opts = ttk.LabelFrame(self.root, text="Options", padding=10)
        opts.pack(fill="x", padx=10, pady=5)
        ttk.Checkbutton(opts, text="Copy files (uncheck to move)", variable=self.copy_var)\
            .grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(opts, text="Dry run (preview only, no file changes)", variable=self.dry_run_var)\
            .grid(row=0, column=1, sticky="w", padx=20)

        ttk.Label(opts, text="Confidence threshold:").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Scale(opts, variable=self.threshold_var, from_=0.0, to=0.9,
                  orient="horizontal", length=260)\
            .grid(row=1, column=1, sticky="w", padx=10, pady=(8, 0))
        self.thresh_label = ttk.Label(opts, text=f"{self.threshold_var.get():.2f}")
        self.thresh_label.grid(row=1, column=2, sticky="w")
        self.threshold_var.trace_add(
            "write", lambda *a: self.thresh_label.config(text=f"{self.threshold_var.get():.2f}")
        )

        # Run row
        run_frame = ttk.Frame(self.root)
        run_frame.pack(fill="x", padx=10, pady=5)
        self.run_btn = ttk.Button(run_frame, text="Organize", command=self._start)
        self.run_btn.pack(side="left")
        self.cancel_btn = ttk.Button(run_frame, text="Cancel", command=self._cancel, state="disabled")
        self.cancel_btn.pack(side="left", padx=8)
        self.progress = ttk.Progressbar(run_frame, mode="determinate", length=320)
        self.progress.pack(side="left", padx=10)
        self.progress_label = ttk.Label(run_frame, text="")
        self.progress_label.pack(side="left")
        # Right-side buttons
        ttk.Button(run_frame, text="Help / Tips", command=self._show_help).pack(side="right", padx=4)
        ttk.Button(run_frame, text="Open Destination", command=self._open_dest).pack(side="right", padx=4)

        # Log
        log_frame = ttk.LabelFrame(self.root, text="Log", padding=6)
        log_frame.pack(fill="both", expand=True, padx=10, pady=(5, 10))
        self.log = scrolledtext.ScrolledText(log_frame, height=10, wrap="word", state="disabled")
        self.log.pack(fill="both", expand=True)

    # ---- categories ----

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
        d = tk.Toplevel(self.root)
        d.title("Category")
        d.transient(self.root)
        d.grab_set()
        d.resizable(False, False)

        ttk.Label(d, text="Folder name:").grid(row=0, column=0, sticky="w", padx=8, pady=6)
        name_var = tk.StringVar()
        ttk.Entry(d, textvariable=name_var, width=40).grid(row=0, column=1, padx=8, pady=6)

        ttk.Label(d, text="CLIP prompt:").grid(row=1, column=0, sticky="nw", padx=8, pady=6)
        prompt_text = tk.Text(d, width=50, height=4, wrap="word")
        prompt_text.grid(row=1, column=1, padx=8, pady=6)

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

        btns = ttk.Frame(d)
        btns.grid(row=2, column=0, columnspan=2, pady=8)
        ttk.Button(btns, text="Save", command=save).pack(side="left", padx=4)
        ttk.Button(btns, text="Cancel", command=d.destroy).pack(side="left", padx=4)

    def _get_categories(self) -> list[tuple[str, str]]:
        return [tuple(self.cat_tree.item(i, "values")) for i in self.cat_tree.get_children()]

    # ---- folder pickers ----

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

    # ---- run ----

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
        self.run_btn.config(state="disabled")
        self.cancel_btn.config(state="normal")
        self.progress.config(value=0, maximum=1)
        self.progress_label.config(text="")
        self._clear_log()

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
        self.cancel_btn.config(state="disabled")

    # ---- log/queue ----

    def _clear_log(self):
        self.log.config(state="normal")
        self.log.delete("1.0", "end")
        self.log.config(state="disabled")

    def _append_log(self, msg: str):
        self.log.config(state="normal")
        self.log.insert("end", msg + "\n")
        self.log.see("end")
        self.log.config(state="disabled")

    def _poll_queue(self):
        try:
            while True:
                kind, payload = self.log_queue.get_nowait()
                if kind == "log":
                    self._append_log(payload)
                elif kind == "progress":
                    cur, total = payload
                    self.progress.config(maximum=total, value=cur)
                    self.progress_label.config(text=f"{cur}/{total}")
                elif kind == "done":
                    self._on_done(payload)
                elif kind == "error":
                    self._append_log(f"FATAL: {payload}")
                    messagebox.showerror("Error", payload)
                    self._reset_buttons()
        except Empty:
            pass
        self.root.after(100, self._poll_queue)

    def _on_done(self, summary: dict):
        self._append_log("")
        self._append_log("=== Done ===")
        for name, count in summary.get("by_cat", {}).items():
            if count:
                self._append_log(f"  {name}: {count}")
        if summary.get("errors"):
            self._append_log(f"  Errors: {summary['errors']}")
        if summary.get("dry_run"):
            self._append_log("(dry run — no files were changed)")
        self._reset_buttons()

    def _reset_buttons(self):
        self.run_btn.config(state="normal")
        self.cancel_btn.config(state="disabled")

    # ---- config ----

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

    # ---- new helpers: open folder, help dialog ----

    def _open_dest(self):
        """Open the destination folder in the OS file manager."""
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

    def _show_help(self):
        """Pop up a Help / Tips window with a built-in guide."""
        d = tk.Toplevel(self.root)
        d.title("Wallpaper Organizer — Help & Tips")
        d.geometry("720x600")
        d.transient(self.root)

        txt = scrolledtext.ScrolledText(d, wrap="word", padx=14, pady=12, font=("Segoe UI", 10))
        txt.pack(fill="both", expand=True)
        txt.insert("end", HELP_TEXT)
        txt.configure(state="disabled")

        btns = ttk.Frame(d)
        btns.pack(fill="x", pady=(0, 10))
        ttk.Button(btns, text="Close", command=d.destroy).pack(side="right", padx=12)


def _make_app_icon():
    """Generate a small app icon as a PIL Image. Returns None if PIL isn't ready yet."""
    try:
        from PIL import Image, ImageDraw
        size = 64
        img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        # Three offset rounded squares — a stack of sorted folders/wallpapers.
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


def main():
    root = tk.Tk()
    try:
        style = ttk.Style()
        for theme in ("vista", "clam", "alt"):
            if theme in style.theme_names():
                style.theme_use(theme)
                break
    except Exception:
        pass
    # App icon (best-effort — silently skipped if PIL isn't ready)
    try:
        icon_img = _make_app_icon()
        if icon_img is not None:
            from PIL import ImageTk
            icon_photo = ImageTk.PhotoImage(icon_img)
            root.iconphoto(True, icon_photo)
            root._icon_ref = icon_photo  # keep a reference to prevent GC
    except Exception:
        pass
    WallpaperOrganizerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
