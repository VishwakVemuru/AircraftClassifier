"""
Aircraft Classifier - simple desktop app.

Run with:  python predict_gui.py
(or double-click run.bat on Windows, which sets everything up for you)
"""
import os
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox

APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, APP_DIR)

try:
    import torch
    import torch.nn as nn
    from torchvision import transforms
    from PIL import Image, ImageTk
    from mobilenetv2_def import mobilenetv2
except ImportError as e:
    root = tk.Tk()
    root.withdraw()
    messagebox.showerror(
        "Missing dependencies",
        f"Could not import required packages:\n{e}\n\n"
        "Please run run.bat (Windows) or run.sh (Mac/Linux) first, "
        "or manually run:\n\npip install torch torchvision pillow"
    )
    sys.exit(1)

MODEL_PATH = os.path.join(APP_DIR, "aircraft_classifier.pt")
PREVIEW_SIZE = (320, 320)

CLASS_DISPLAY_NAMES = {
    "a320": "Airbus A320",
    "b737": "Boeing 737",
    "f-16": "F-16 Fighting Falcon",
    "su-30": "Su-30",
}


class AircraftClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Aircraft Classifier")
        self.root.geometry("480x640")
        self.root.resizable(False, False)
        self.root.configure(bg="#1e2530")

        self.model = None
        self.class_names = None
        self.transform = None
        self.current_image_path = None
        self.tk_preview_img = None

        self._build_ui()
        self._load_model_async()

    # ------------------------------------------------------------------
    def _build_ui(self):
        title = tk.Label(
            self.root, text="Aircraft Classifier", font=("Segoe UI", 20, "bold"),
            bg="#1e2530", fg="#ffffff",
        )
        title.pack(pady=(20, 4))

        subtitle = tk.Label(
            self.root, text="F-16  ·  Su-30  ·  Boeing 737  ·  Airbus A320",
            font=("Segoe UI", 10), bg="#1e2530", fg="#8a94a6",
        )
        subtitle.pack(pady=(0, 16))

        self.status_label = tk.Label(
            self.root, text="Loading model...", font=("Segoe UI", 9),
            bg="#1e2530", fg="#f0a500",
        )
        self.status_label.pack(pady=(0, 8))

        self.preview_frame = tk.Frame(
            self.root, width=PREVIEW_SIZE[0], height=PREVIEW_SIZE[1],
            bg="#2a3342", highlightbackground="#3d4759", highlightthickness=1,
        )
        self.preview_frame.pack(pady=8)
        self.preview_frame.pack_propagate(False)

        self.preview_label = tk.Label(
            self.preview_frame, text="No photo selected", bg="#2a3342", fg="#5c6779",
            font=("Segoe UI", 10),
        )
        self.preview_label.pack(expand=True)

        self.upload_btn = tk.Button(
            self.root, text="Choose Photo", command=self.choose_photo,
            font=("Segoe UI", 11, "bold"), bg="#3d7fff", fg="white",
            activebackground="#2f66d0", relief="flat", padx=20, pady=8,
            cursor="hand2", state="disabled",
        )
        self.upload_btn.pack(pady=(16, 6))

        self.classify_btn = tk.Button(
            self.root, text="Classify", command=self.classify_current,
            font=("Segoe UI", 11, "bold"), bg="#2ecc71", fg="white",
            activebackground="#25a35a", relief="flat", padx=20, pady=8,
            cursor="hand2", state="disabled",
        )
        self.classify_btn.pack(pady=(0, 16))

        self.results_frame = tk.Frame(self.root, bg="#1e2530")
        self.results_frame.pack(fill="x", padx=30)

    # ------------------------------------------------------------------
    def _load_model_async(self):
        threading.Thread(target=self._load_model, daemon=True).start()

    def _load_model(self):
        try:
            ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
            self.class_names = ckpt["class_names"]
            img_size = ckpt["img_size"]
            mean, std = ckpt["mean"], ckpt["std"]

            model = mobilenetv2(num_classes=1000)
            model.classifier = nn.Linear(model.classifier.in_features, len(self.class_names))
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            self.model = model

            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])

            self.root.after(0, self._on_model_loaded)
        except Exception as e:
            self.root.after(0, lambda: self._on_model_load_failed(e))

    def _on_model_loaded(self):
        self.status_label.config(text="Model ready", fg="#2ecc71")
        self.upload_btn.config(state="normal")

    def _on_model_load_failed(self, error):
        self.status_label.config(text="Failed to load model", fg="#e74c3c")
        messagebox.showerror("Error", f"Could not load model:\n{error}")

    # ------------------------------------------------------------------
    def choose_photo(self):
        path = filedialog.askopenfilename(
            title="Choose an aircraft photo",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All files", "*.*")],
        )
        if not path:
            return
        self.current_image_path = path
        self._show_preview(path)
        self.classify_btn.config(state="normal")
        self._clear_results()

    def _show_preview(self, path):
        img = Image.open(path).convert("RGB")
        img.thumbnail(PREVIEW_SIZE)
        self.tk_preview_img = ImageTk.PhotoImage(img)
        self.preview_label.config(image=self.tk_preview_img, text="")

    def _clear_results(self):
        for widget in self.results_frame.winfo_children():
            widget.destroy()

    # ------------------------------------------------------------------
    def classify_current(self):
        if not self.current_image_path or self.model is None:
            return
        self.classify_btn.config(state="disabled", text="Classifying...")
        threading.Thread(target=self._classify, daemon=True).start()

    def _classify(self):
        try:
            img = Image.open(self.current_image_path).convert("RGB")
            x = self.transform(img).unsqueeze(0)
            with torch.no_grad():
                logits = self.model(x)
                probs = torch.softmax(logits, dim=1)[0].tolist()
            ranked = sorted(zip(self.class_names, probs), key=lambda p: -p[1])
            self.root.after(0, lambda: self._show_results(ranked))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Prediction failed:\n{e}"))
        finally:
            self.root.after(0, lambda: self.classify_btn.config(state="normal", text="Classify"))

    def _show_results(self, ranked):
        self._clear_results()
        best_name, best_p = ranked[0]
        best_label = CLASS_DISPLAY_NAMES.get(best_name, best_name)

        best_frame = tk.Label(
            self.results_frame,
            text=f"Best guess: {best_label}  ({best_p*100:.1f}%)",
            font=("Segoe UI", 13, "bold"), bg="#1e2530", fg="#2ecc71",
            wraplength=400, justify="left",
        )
        best_frame.pack(anchor="w", pady=(4, 12))

        for name, p in ranked:
            label = CLASS_DISPLAY_NAMES.get(name, name)
            row = tk.Frame(self.results_frame, bg="#1e2530")
            row.pack(fill="x", pady=3)

            name_lbl = tk.Label(
                row, text=label, font=("Segoe UI", 10), bg="#1e2530", fg="#c8cedb",
                width=20, anchor="w",
            )
            name_lbl.pack(side="left")

            bar_bg = tk.Frame(row, bg="#2a3342", height=14, width=180)
            bar_bg.pack(side="left", padx=8)
            bar_bg.pack_propagate(False)

            fill_width = max(2, int(180 * p))
            bar_fill = tk.Frame(bar_bg, bg="#3d7fff", height=14, width=fill_width)
            bar_fill.place(x=0, y=0)

            pct_lbl = tk.Label(
                row, text=f"{p*100:.1f}%", font=("Segoe UI", 10), bg="#1e2530", fg="#8a94a6",
                width=6,
            )
            pct_lbl.pack(side="left")


if __name__ == "__main__":
    root = tk.Tk()
    app = AircraftClassifierApp(root)
    root.mainloop()
