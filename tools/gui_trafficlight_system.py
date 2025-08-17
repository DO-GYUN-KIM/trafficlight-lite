# -*- coding: utf-8 -*-
"""
Traffic-Light Video System - GUI (full version, with captions)
- ROI 슬라이더 조절 + Preview ROI
- draw_roi / captions 옵션 전달
- 캡션 임계값: min tracks / min conf
- 콘솔 로그 실시간 표시
"""

import os
import sys
import queue
import threading
import subprocess
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

PROJECT_ROOT = Path(r"C:\summer\trafficlight-lite")
TOOLS_DIR    = PROJECT_ROOT / "tools"
DEFAULT_WEIGHTS = PROJECT_ROOT / r"runs\detect\train_gpu_quick2\weights\best.pt"
DEFAULT_SOURCE  = PROJECT_ROOT / "test.mp4"

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Traffic-Light Video System - GUI")
        self.geometry("940x740")
        self.minsize(900, 680)

        self.proc = None
        self.log_q = queue.Queue()

        self._build_ui()
        self.after(100, self._drain_log)

    def _build_ui(self):
        frm = ttk.LabelFrame(self, text="Settings")
        frm.pack(side=tk.TOP, fill=tk.X, padx=10, pady=8, ipady=4)

        # Weights
        ttk.Label(frm, text="Weights (.pt):").grid(row=0, column=0, sticky="e", padx=6, pady=6)
        self.var_weights = tk.StringVar(value=str(DEFAULT_WEIGHTS))
        ttk.Entry(frm, textvariable=self.var_weights, width=84).grid(row=0, column=1, sticky="we", padx=4, pady=6)
        ttk.Button(frm, text="Browse...", command=self._pick_weights).grid(row=0, column=2, padx=4, pady=6)

        # Source
        ttk.Label(frm, text="Source (video):").grid(row=1, column=0, sticky="e", padx=6, pady=6)
        self.var_source = tk.StringVar(value=str(DEFAULT_SOURCE))
        ttk.Entry(frm, textvariable=self.var_source, width=84).grid(row=1, column=1, sticky="we", padx=4, pady=6)
        ttk.Button(frm, text="Browse...", command=self._pick_source).grid(row=1, column=2, padx=4, pady=6)

        # Device / imgsz / name
        ttk.Label(frm, text="Device:").grid(row=2, column=0, sticky="e", padx=6, pady=6)
        self.var_device = tk.StringVar(value="0")  # "0" or "cpu"
        ttk.Combobox(frm, textvariable=self.var_device, values=("0", "cpu"), width=10, state="readonly").grid(row=2, column=1, sticky="w", padx=4, pady=6)

        ttk.Label(frm, text="imgsz:").grid(row=2, column=1, sticky="e", padx=210, pady=6)
        self.var_imgsz = tk.IntVar(value=512)
        ttk.Entry(frm, textvariable=self.var_imgsz, width=8).grid(row=2, column=1, sticky="w", padx=270, pady=6)

        ttk.Label(frm, text="Run name:").grid(row=2, column=1, sticky="e", padx=410, pady=6)
        self.var_name = tk.StringVar(value="video_system_gui")
        ttk.Entry(frm, textvariable=self.var_name, width=20).grid(row=2, column=1, sticky="w", padx=490, pady=6)

        # conf / debounce / min_area
        ttk.Label(frm, text="conf:").grid(row=3, column=0, sticky="e", padx=6, pady=6)
        self.var_conf = tk.DoubleVar(value=0.55)
        ttk.Scale(frm, from_=0.10, to=0.90, variable=self.var_conf, orient="horizontal", length=300).grid(row=3, column=1, sticky="w", padx=4, pady=4)
        ttk.Label(frm, textvariable=self.var_conf, width=6).grid(row=3, column=1, sticky="w", padx=310, pady=4)

        ttk.Label(frm, text="debounce_ms:").grid(row=3, column=1, sticky="e", padx=390, pady=6)
        self.var_debounce = tk.IntVar(value=800)
        ttk.Entry(frm, textvariable=self.var_debounce, width=10).grid(row=3, column=1, sticky="w", padx=490, pady=6)

        ttk.Label(frm, text="min_area:").grid(row=3, column=1, sticky="e", padx=590, pady=6)
        self.var_min_area = tk.IntVar(value=120)
        ttk.Entry(frm, textvariable=self.var_min_area, width=10).grid(row=3, column=1, sticky="w", padx=670, pady=6)

        # -------- ROI 슬라이더 --------
        ttk.Label(frm, text="ROI (nx,ny,nw,nh)").grid(row=4, column=0, sticky="e", padx=6, pady=6)
        self.var_roi_x = tk.DoubleVar(value=0.00)
        self.var_roi_y = tk.DoubleVar(value=0.00)
        self.var_roi_w = tk.DoubleVar(value=1.00)
        self.var_roi_h = tk.DoubleVar(value=0.55)

        roi_sl = ttk.Frame(frm); roi_sl.grid(row=4, column=1, sticky="we", padx=4)
        for i,(text,var) in enumerate([
            ("x", self.var_roi_x),
            ("y", self.var_roi_y),
            ("w", self.var_roi_w),
            ("h", self.var_roi_h),
        ]):
            ttk.Label(roi_sl, text=text).grid(row=i, column=0, sticky="e")
            s = ttk.Scale(roi_sl,
                          from_=0.0 if text in ("x","y") else 0.1,
                          to=1.0,
                          variable=var,
                          orient="horizontal",
                          length=340)
            s.grid(row=i, column=1, padx=6, pady=3, sticky="we")
            ttk.Label(roi_sl, textvariable=var, width=6).grid(row=i, column=2, padx=6)

        ttk.Button(frm, text="Preview ROI", command=self.preview_roi).grid(row=4, column=2, padx=4, pady=6)

        # min_conf_by_cls + draw_roi + captions
        ttk.Label(frm, text="min_conf_by_cls").grid(row=5, column=0, sticky="e", padx=6, pady=6)
        self.var_min_conf_map = tk.StringVar(value="8:0.80,7:0.70")
        ttk.Entry(frm, textvariable=self.var_min_conf_map, width=50).grid(row=5, column=1, sticky="w", padx=4, pady=6)

        self.var_draw_roi = tk.BooleanVar(value=True)
        ttk.Checkbutton(frm, text="Draw ROI", variable=self.var_draw_roi).grid(row=5, column=2, sticky="w", padx=6)

        # ----- captions 옵션 -----
        cap_frm = ttk.Frame(frm); cap_frm.grid(row=6, column=1, sticky="w", padx=4, pady=4)
        self.var_captions = tk.BooleanVar(value=True)
        ttk.Checkbutton(cap_frm, text="Show captions (Korean)", variable=self.var_captions).grid(row=0, column=0, padx=4)

        ttk.Label(cap_frm, text="min tracks").grid(row=0, column=1, padx=6)
        self.var_cap_min_tracks = tk.IntVar(value=2)
        ttk.Entry(cap_frm, textvariable=self.var_cap_min_tracks, width=6).grid(row=0, column=2)

        ttk.Label(cap_frm, text="min conf").grid(row=0, column=3, padx=6)
        self.var_cap_min_conf = tk.DoubleVar(value=0.60)
        ttk.Entry(cap_frm, textvariable=self.var_cap_min_conf, width=6).grid(row=0, column=4)

        # 버튼
        btns = ttk.Frame(frm); btns.grid(row=7, column=0, columnspan=3, sticky="we", padx=4, pady=8)
        ttk.Button(btns, text="Run", command=self.run).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="Stop", command=self.stop).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="Open Output Folder", command=self.open_output).pack(side=tk.LEFT, padx=4)

        # 콘솔
        log_frm = ttk.LabelFrame(self, text="Console")
        log_frm.pack(fill=tk.BOTH, expand=True, padx=10, pady=4)
        self.txt = tk.Text(log_frm, wrap="word", height=20)
        self.txt.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

    # ---------- 핸들러 ----------
    def _pick_weights(self):
        p = filedialog.askopenfilename(title="Select weights (.pt)",
                                       filetypes=[("PyTorch weights",".pt"),("All","*.*")])
        if p: self.var_weights.set(p)

    def _pick_source(self):
        p = filedialog.askopenfilename(title="Select video",
                                       filetypes=[("MP4","*.mp4"),("All","*.*")])
        if p: self.var_source.set(p)

    def preview_roi(self):
        import cv2
        src = Path(self.var_source.get())
        if not src.exists():
            messagebox.showerror("Error", f"Source not found:\n{src}")
            return
        cap = cv2.VideoCapture(str(src))
        ok, frame = cap.read()
        cap.release()
        if not ok:
            messagebox.showerror("Error", "영상을 읽을 수 없습니다.")
            return

        H, W = frame.shape[:2]
        win = "ROI preview (ESC to close)"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 900, int(900*H/W))

        while True:
            vis = frame.copy()
            rx, ry, rw, rh = self.var_roi_x.get(), self.var_roi_y.get(), self.var_roi_w.get(), self.var_roi_h.get()
            x1, y1, x2, y2 = int(rx*W), int(ry*H), int((rx+rw)*W), int((ry+rh)*H)
            cv2.rectangle(vis, (x1,y1), (x2,y2), (255,180,0), 2)
            cv2.putText(vis, f"ROI: ({rx:.2f},{ry:.2f},{rw:.2f},{rh:.2f})", (12,28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50,220,255), 2, cv2.LINE_AA)
            cv2.imshow(win, vis)
            if cv2.waitKey(50) & 0xFF == 27:  # ESC
                break
        cv2.destroyWindow(win)

    def run(self):
        if self.proc and self.proc.poll() is None:
            messagebox.showinfo("Running", "이미 실행 중입니다.")
            return

        weights = Path(self.var_weights.get())
        source  = Path(self.var_source.get())
        if not weights.exists():
            messagebox.showerror("Error", f"Weights not found:\n{weights}")
            return
        if not source.exists():
            messagebox.showerror("Error", f"Source not found:\n{source}")
            return

        device   = self.var_device.get()
        imgsz    = str(self.var_imgsz.get())
        name     = self.var_name.get().strip() or "video_system_gui"
        conf     = f"{self.var_conf.get():.2f}"
        debounce = str(self.var_debounce.get())
        min_area = str(self.var_min_area.get())
        roi = f"{self.var_roi_x.get():.2f},{self.var_roi_y.get():.2f},{self.var_roi_w.get():.2f},{self.var_roi_h.get():.2f}"
        min_map  = self.var_min_conf_map.get().strip()

        cmd = [
            sys.executable, str(TOOLS_DIR / "video_trafficlight_system.py"),
            "--weights", str(weights),
            "--source",  str(source),
            "--device",  device,
            "--imgsz",   imgsz,
            "--name",    name,
            "--show",
            "--conf",    conf,
            "--debounce_ms", debounce,
            "--min_area",    min_area,
            "--roi",         roi,
            "--min_conf_by_cls", min_map
        ]
        if self.var_draw_roi.get():
            cmd.append("--draw_roi")
        if self.var_captions.get():
            cmd += ["--captions",
                    "--caption_min_tracks", str(self.var_cap_min_tracks.get()),
                    "--caption_min_conf",   f"{self.var_cap_min_conf.get():.2f}"]

        self._log("\n>>> " + " ".join(cmd) + "\n")

        def runner():
            try:
                self.proc = subprocess.Popen(
                    cmd, cwd=str(TOOLS_DIR),
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
                )
                for line in self.proc.stdout:
                    self.log_q.put(line.rstrip())
            except Exception as e:
                self.log_q.put(f"[ERROR] {e}")
            finally:
                self.log_q.put("[DONE] process finished.")

        threading.Thread(target=runner, daemon=True).start()

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            self._log("[INFO] process terminated by user.\n")
        else:
            self._log("[INFO] no running process.\n")

    def open_output(self):
        out_root = PROJECT_ROOT / "runs" / "detect"
        os.makedirs(out_root, exist_ok=True)
        if sys.platform.startswith("win"):
            os.startfile(str(out_root))
        else:
            subprocess.call(["open", str(out_root)])

    def _log(self, msg: str):
        self.txt.insert(tk.END, msg + "\n")
        self.txt.see(tk.END)

    def _drain_log(self):
        try:
            while True:
                line = self.log_q.get_nowait()
                self._log(line)
        except queue.Empty:
            pass
        self.after(100, self._drain_log)

if __name__ == "__main__":
    app = App()
    app.mainloop()
