# C:\summer\trafficlight-lite\tools\gui_trafficlight_system.py
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import subprocess, sys, json, os
from pathlib import Path

CONFIG_FILE = "gui_config.json"

class TrafficLightGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("trafficlight-lite GUI")

        self.config_data = self.load_config()

        # ------- Settings -------
        self.var_weights = tk.StringVar(value=self.config_data.get("weights", "C:/summer/trafficlight-lite/best.pt"))
        self.var_source  = tk.StringVar(value=self.config_data.get("source",  "C:/summer/trafficlight-lite/test.mp4"))
        self.var_device  = tk.StringVar(value=self.config_data.get("device",  "0"))
        self.var_conf    = tk.DoubleVar(value=self.config_data.get("conf",    0.55))
        self.var_imgsz   = tk.IntVar(   value=self.config_data.get("imgsz",   640))

        # ------- ROI (Trapezoid params -> polygon 계산) -------
        self.var_top_w     = tk.DoubleVar(value=self.config_data.get("top_w",     0.3))
        self.var_bottom_w  = tk.DoubleVar(value=self.config_data.get("bottom_w",  0.9))
        self.var_height    = tk.DoubleVar(value=self.config_data.get("height",    0.6))
        self.var_bottom_x  = tk.DoubleVar(value=self.config_data.get("bottom_x",  0.5))

        # ------- Flags -------
        self.var_draw_roi = tk.BooleanVar(value=self.config_data.get("draw_roi", True))
        self.var_captions = tk.BooleanVar(value=self.config_data.get("captions", True))
        self.var_show     = tk.BooleanVar(value=self.config_data.get("show",     True))

        # ------- Advanced -------
        self.var_min_area         = tk.IntVar(value=int(self.config_data.get("min_area", 120)))
        self.var_debounce         = tk.IntVar(value=int(self.config_data.get("debounce_ms", 800)))
        self.var_iou              = tk.DoubleVar(value=float(self.config_data.get("iou", 0.7)))
        self.var_min_conf_by_cls  = tk.StringVar(value=self.config_data.get("min_conf_by_cls", "8:0.80,7:0.70"))

        # Captions stabilization
        self.var_cap_min_frames = tk.IntVar(value=int(self.config_data.get("caption_min_frames", 6)))
        self.var_cap_min_tracks = tk.IntVar(value=int(self.config_data.get("caption_min_tracks", 1)))
        self.var_cap_min_conf   = tk.DoubleVar(value=float(self.config_data.get("caption_min_conf", 0.50)))

        self.proc = None
        self.create_widgets()

    # ---------- UI ----------
    def create_widgets(self):
        # Settings
        frm = ttk.LabelFrame(self, text="Settings")
        frm.grid(row=0, column=0, padx=8, pady=8, sticky="nsew")

        ttk.Label(frm, text="Weights (.pt):").grid(row=0, column=0, sticky="e", padx=6, pady=6)
        ttk.Entry(frm, textvariable=self.var_weights, width=60).grid(row=0, column=1, padx=6)
        ttk.Button(frm, text="Browse", command=self.browse_weights).grid(row=0, column=2, padx=6)

        ttk.Label(frm, text="Source (video):").grid(row=1, column=0, sticky="e", padx=6, pady=6)
        ttk.Entry(frm, textvariable=self.var_source, width=60).grid(row=1, column=1, padx=6)
        ttk.Button(frm, text="Browse", command=self.browse_source).grid(row=1, column=2, padx=6)

        ttk.Label(frm, text="Device:").grid(row=2, column=0, sticky="e", padx=6, pady=6)
        ttk.Entry(frm, textvariable=self.var_device, width=6).grid(row=2, column=1, sticky="w", padx=6)

        ttk.Label(frm, text="conf:").grid(row=2, column=1, sticky="e", padx=100, pady=6)
        ttk.Entry(frm, textvariable=self.var_conf, width=6).grid(row=2, column=1, sticky="w", padx=150)

        ttk.Label(frm, text="imgsz:").grid(row=2, column=2, sticky="e", padx=6, pady=6)
        ttk.Entry(frm, textvariable=self.var_imgsz, width=6).grid(row=2, column=3, sticky="w", padx=6)

        # ROI (Trapezoid)
        roi_frm = ttk.LabelFrame(self, text="ROI Trapezoid")
        roi_frm.grid(row=1, column=0, padx=8, pady=8, sticky="nsew")

        sliders = [
            ("Top width", self.var_top_w),
            ("Bottom width", self.var_bottom_w),
            ("Height", self.var_height),
            ("Bottom X center", self.var_bottom_x),
        ]
        for i, (lbl, var) in enumerate(sliders):
            ttk.Label(roi_frm, text=lbl).grid(row=i, column=0, sticky="e")
            ttk.Scale(roi_frm, from_=0.0, to=1.0, variable=var,
                      orient="horizontal", length=300,
                      command=lambda e: self.save_config()).grid(row=i, column=1, padx=6)
            ttk.Label(roi_frm, textvariable=var, width=6).grid(row=i, column=2, padx=6)

        # Flags
        flag_frm = ttk.LabelFrame(self, text="Flags")
        flag_frm.grid(row=2, column=0, padx=8, pady=8, sticky="nsew")
        ttk.Checkbutton(flag_frm, text="draw_roi", variable=self.var_draw_roi,
                        command=self.save_config).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(flag_frm, text="captions", variable=self.var_captions,
                        command=self.save_config).grid(row=0, column=1, sticky="w")
        ttk.Checkbutton(flag_frm, text="show window", variable=self.var_show,
                        command=self.save_config).grid(row=0, column=2, sticky="w")

        # Advanced
        adv_frm = ttk.LabelFrame(self, text="Advanced Settings")
        adv_frm.grid(row=3, column=0, padx=8, pady=8, sticky="nsew")

        row = 0
        ttk.Label(adv_frm, text="min_area:").grid(row=row, column=0, sticky="e", padx=6, pady=6)
        ttk.Entry(adv_frm, textvariable=self.var_min_area, width=8).grid(row=row, column=1, sticky="w")

        ttk.Label(adv_frm, text="debounce_ms:").grid(row=row, column=2, sticky="e", padx=6, pady=6)
        ttk.Entry(adv_frm, textvariable=self.var_debounce, width=8).grid(row=row, column=3, sticky="w")

        ttk.Label(adv_frm, text="iou:").grid(row=row, column=4, sticky="e", padx=6, pady=6)
        ttk.Entry(adv_frm, textvariable=self.var_iou, width=8).grid(row=row, column=5, sticky="w")

        row += 1
        ttk.Label(adv_frm, text="min_conf_by_cls:").grid(row=row, column=0, sticky="e", padx=6, pady=6)
        ttk.Entry(adv_frm, textvariable=self.var_min_conf_by_cls, width=28).grid(row=row, column=1, sticky="w", columnspan=5)

        # Captions stabilization controls
        row += 1
        ttk.Label(adv_frm, text="caption_min_frames:").grid(row=row, column=0, sticky="e", padx=6, pady=6)
        ttk.Entry(adv_frm, textvariable=self.var_cap_min_frames, width=8).grid(row=row, column=1, sticky="w")

        ttk.Label(adv_frm, text="caption_min_tracks:").grid(row=row, column=2, sticky="e", padx=6, pady=6)
        ttk.Entry(adv_frm, textvariable=self.var_cap_min_tracks, width=8).grid(row=row, column=3, sticky="w")

        ttk.Label(adv_frm, text="caption_min_conf:").grid(row=row, column=4, sticky="e", padx=6, pady=6)
        ttk.Entry(adv_frm, textvariable=self.var_cap_min_conf, width=8).grid(row=row, column=5, sticky="w")

        # Run / Stop
        btn_frm = ttk.Frame(self)
        btn_frm.grid(row=4, column=0, pady=8)
        ttk.Button(btn_frm, text="Run", command=self.run).grid(row=0, column=0, padx=10)
        ttk.Button(btn_frm, text="Stop", command=self.stop).grid(row=0, column=1, padx=10)

    # ---------- Helpers ----------
    def browse_weights(self):
        p = filedialog.askopenfilename(filetypes=[("PyTorch Weights", "*.pt")])
        if p:
            self.var_weights.set(p)
            self.save_config()

    def browse_source(self):
        p = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi;*.mov")])
        if p:
            self.var_source.set(p)
            self.save_config()

    def load_config(self):
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def save_config(self):
        data = {
            "weights": self.var_weights.get(),
            "source":  self.var_source.get(),
            "device":  self.var_device.get(),
            "conf":    self.var_conf.get(),
            "imgsz":   self.var_imgsz.get(),
            "top_w":     self.var_top_w.get(),
            "bottom_w":  self.var_bottom_w.get(),
            "height":    self.var_height.get(),
            "bottom_x":  self.var_bottom_x.get(),
            "draw_roi":  self.var_draw_roi.get(),
            "captions":  self.var_captions.get(),
            "show":      self.var_show.get(),
            "min_area":        self.var_min_area.get(),
            "debounce_ms":     self.var_debounce.get(),
            "iou":             self.var_iou.get(),
            "min_conf_by_cls": self.var_min_conf_by_cls.get(),
            "caption_min_frames": self.var_cap_min_frames.get(),
            "caption_min_tracks": self.var_cap_min_tracks.get(),
            "caption_min_conf":   self.var_cap_min_conf.get(),
        }
        try:
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print("Config save error:", e)

    # ---------- Run / Stop ----------
    def run(self):
        # 항상 최신 설정 저장
        self.save_config()

        weights = Path(self.var_weights.get())
        source  = Path(self.var_source.get())
        if not weights.exists():
            messagebox.showerror("Error", f"Weights not found: {weights}")
            return
        if not source.exists():
            messagebox.showerror("Error", f"Source not found: {source}")
            return

        # trapezoid → polygon (normalized)
        top_w = self.var_top_w.get()
        bottom_w = self.var_bottom_w.get()
        h  = self.var_height.get()
        bx = self.var_bottom_x.get()

        x1 = bx - bottom_w/2
        x2 = bx + bottom_w/2
        x3 = 0.5 - top_w/2
        x4 = 0.5 + top_w/2

        # 좌상→우상→우하→좌하
        roi_vals = [x3, 0.0, x4, 0.0, x2, h, x1, h]
        roi_str = ",".join([f"{v:.3f}" for v in roi_vals])

        script_path = Path(__file__).parent / "video_trafficlight_system.py"
        cmd = [
            sys.executable, str(script_path),
            "--weights", str(weights),
            "--source",  str(source),
            "--device",  self.var_device.get(),
            "--imgsz",   str(self.var_imgsz.get()),
            "--conf",    str(self.var_conf.get()),
            "--iou",     str(self.var_iou.get()),
            "--name",    "gui_run",
            "--debounce_ms",    str(self.var_debounce.get()),
            "--min_area",       str(self.var_min_area.get()),
            "--min_conf_by_cls", self.var_min_conf_by_cls.get(),
            "--roi", roi_str,
        ]
        if self.var_draw_roi.get(): cmd.append("--draw_roi")
        if self.var_captions.get(): cmd.append("--captions")
        if self.var_show.get():     cmd.append("--show")

        # captions stabilization 인자 전달
        cmd += [
            "--caption_min_frames", str(self.var_cap_min_frames.get()),
            "--caption_min_tracks", str(self.var_cap_min_tracks.get()),
            "--caption_min_conf",   str(self.var_cap_min_conf.get()),
        ]

        self.proc = subprocess.Popen(cmd)

    def stop(self):
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.kill()
            except Exception:
                pass
            self.proc = None

if __name__ == "__main__":
    app = TrafficLightGUI()
    app.mainloop()
