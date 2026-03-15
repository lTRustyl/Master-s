import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import cv2
import time
import threading
from ultralytics import YOLO


class Logger:
    def __init__(self, log_path="detections.txt"):
        self.log_path = log_path
        # Перезаписуємо файл і пишемо заголовок
        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write("time_s\tclass\tconf\tbbox(x1,y1,x2,y2)\n")

    def log_snapshot(self, t_rel_seconds, objects):
        """
        Записує "зріз" детекцій у момент часу t_rel_seconds.
        Якщо об'єктів немає — пишемо NO_OBJECTS.
        objects: список (cls_name, conf, (x1,y1,x2,y2))
        """
        with open(self.log_path, "a", encoding="utf-8") as f:
            if not objects:
                f.write(f"{t_rel_seconds:.2f}\tNO_OBJECTS\t-\t-\n")
            else:
                for cls_name, conf, (x1, y1, x2, y2) in objects:
                    f.write(
                        f"{t_rel_seconds:.2f}\t{cls_name}\t{conf:.3f}\t"
                        f"{x1},{y1},{x2},{y2}\n"
                    )

    def log_summary(self, avg_fps, avg_inference_ms):
        """
        Записує в кінець файлу зведені метрики:
        - середній FPS
        - середній час інференсу одного кадру, мс
        """
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write("\n=== SUMMARY ===\n")
            f.write(f"Average FPS:\t{avg_fps:.3f}\n")
            f.write(
                f"Average inference time per frame, ms:\t{avg_inference_ms:.3f}\n"
            )


class SettingsManager:
    def __init__(self, path="settings.json"):
        self.user_preferences = {}
        self.path = path

    def load_settings(self):
        pass

    def save_settings(self):
        pass


class AppGUI:
    def __init__(self):
        # Розмір, з яким працює модель
        self.MODEL_W = 640
        self.MODEL_H = 640

        # ==== GUI BASE ====
        self.root = tk.Tk()
        self.root.title("Military Recognition")

        # Стартовий розмір вікна під відео + панелі
        window_w = self.MODEL_W + 400
        window_h = self.MODEL_H + 260
        self.root.geometry(f"{window_w}x{window_h}")
        self.root.minsize(900, 700)

        # ======= ВИБІР МОДЕЛІ (Light / Heavy) =======
        self.model_var = tk.StringVar(value="Light")  # за замовчуванням легка
        self.detector = None          # модель підвантажимо перед стартом
        self.class_names = []         # заповнимо після завантаження моделі
        self.classes_ui = ["All"]     # буде оновлено пізніше


        self.class_colors = {
            "AFV": (0, 255, 0),      # зелений
            "APC": (0, 255, 255),    # жовтий
            "MEV": (255, 0, 0),      # синій
            "LAV": (255, 0, 255),    # фіолетовий
        }
        self.default_color = (0, 165, 255)  # помаранчевий

        # ==== CONTROL PANEL ====
        top = ttk.Frame(self.root)
        top.pack(fill="x", pady=4, padx=4)

        self.src_var = tk.StringVar(value="0-USB")
        self.cls_var = tk.StringVar(value="All")
        self.conf_var = tk.DoubleVar(value=0.55)

        # Source
        ttk.Label(top, text="Source").pack(side="left")
        ttk.Combobox(
            top,
            textvariable=self.src_var,
            values=("0-USB", "rtsp://YOUR_STREAM", "video_name"),
            width=22,
        ).pack(side="left", padx=4)

        # Model selector
        ttk.Label(top, text="Model").pack(side="left")
        ttk.Combobox(
            top,
            textvariable=self.model_var,
            values=("Light", "Heavy"),
            width=10,
            state="readonly",
        ).pack(side="left", padx=4)

        # Class filter (заповнимо списком класів після завантаження моделі)
        ttk.Label(top, text="Class").pack(side="left")
        self.class_combo = ttk.Combobox(
            top,
            textvariable=self.cls_var,
            values=("All",),
            width=18,
            state="readonly",
        )
        self.class_combo.pack(side="left", padx=4)

        # Confidence
        ttk.Label(top, text="Conf").pack(side="left")
        tk.Scale(
            top,
            from_=0.30,
            to=0.95,
            resolution=0.05,
            orient="horizontal",
            variable=self.conf_var,
            length=200,
        ).pack(side="left", padx=4)

        # Start/Stop button
        self.btn = ttk.Button(top, text="Start", command=self.toggle)
        self.btn.pack(side="left", padx=10)

        # ==== CENTRAL AREA: статус + відео ====
        center = ttk.Frame(self.root)
        center.pack(fill="both", expand=True, padx=4)

        status = ttk.Frame(center)
        status.pack(fill="x")
        self.status_lbl = ttk.Label(status, text="Idle")
        self.status_lbl.pack(side="left")

        video_frame = ttk.Frame(center)
        video_frame.pack(expand=True)

        self.video_lbl = tk.Label(video_frame, bg="black")
        self.video_lbl.pack()

        # Таймер під відео
        self.timer_lbl = ttk.Label(video_frame, text="Time: 0.00 s")
        self.timer_lbl.pack(pady=(4, 0))

        # ==== LOG TABLE (bottom) ====
        bottom = ttk.Frame(self.root)
        bottom.pack(fill="both", padx=4, pady=(0, 4))

        cols = ("Time_s", "Class", "Conf", "BBox")
        self.tree = ttk.Treeview(bottom, columns=cols, show="headings", height=8)
        for col in cols:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=140, anchor="center")
        self.tree.pack(fill="both", expand=False)

        self.logger = None
        self.cap = None
        self.running = False

        # Для відносного часу
        self.start_time = None
        self.last_logged_sec = -1

        # Лічильники для FPS / inference time
        self.total_frames = 0
        self.total_inference_time = 0.0

    # ───────────────────────────────────────────────────────────
    def load_selected_model(self):
        model_choice = self.model_var.get()
        if model_choice == "Light":
            weights = "lightmodel.pt"
        else:
            weights = "heavymodel.pt"

        try:
            self.detector = YOLO(weights)
        except Exception as e:
            messagebox.showerror("Model error", f"Cannot load model '{weights}': {e}")
            self.detector = None
            return False

        # Імена класів
        self.class_names = list(self.detector.model.names.values())
        self.classes_ui = ["All"] + self.class_names

        # Оновлюємо combobox класів
        self.class_combo["values"] = self.classes_ui
        self.cls_var.set("All")

        return True

    # ───────────────────────────────────────────────────────────
    def toggle(self):
        if not self.running:
            if not self.start_video():
                return
            self.btn.config(text="Stop")
        else:
            self.running = False
            self.btn.config(text="Start")

    def start_video(self):
        # 1) Завантажуємо вибрану модель
        if not self.load_selected_model():
            return False

        # 2) Налаштовуємо джерело відео
        src_text = self.src_var.get().strip()

        if src_text.startswith("0"):
            src = 0
        elif src_text.lower().startswith("rtsp://"):
            src = src_text
        else:
            if not src_text.lower().endswith(".mp4"):
                src = src_text + ".mp4"
            else:
                src = src_text

        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            messagebox.showerror("Source error", f"Cannot open: {src}")
            return False

        self.logger = Logger("detections.txt")
        self.start_time = time.time()
        self.last_logged_sec = -1

        # скидаємо лічильники
        self.total_frames = 0
        self.total_inference_time = 0.0

        self.running = True
        threading.Thread(target=self.loop, daemon=True).start()
        return True

    # ───────────────────────────────────────────────────────────
    def preprocess_frame(self, frame):
        """CLAHE по яскравості для кращого контрасту."""
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        y_eq = clahe.apply(y)
        ycrcb_eq = cv2.merge((y_eq, cr, cb))
        frame_eq = cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2BGR)
        return frame_eq

    # ───────────────────────────────────────────────────────────
    def _merge_overlapping_boxes(self, boxes, iou_thresh=0.6):
        """
        boxes: список (cls_name, conf, (x1, y1, x2, y2))
        Повертає список боксів, де для кожної зони кадру лишається
        лише один бокс з найбільшою впевненістю (class-agnostic NMS).
        """
        if not boxes:
            return []

        boxes_sorted = sorted(boxes, key=lambda b: b[1], reverse=True)
        kept = []

        def iou(box_a, box_b):
            _, _, (ax1, ay1, ax2, ay2) = box_a
            _, _, (bx1, by1, bx2, by2) = box_b

            inter_x1 = max(ax1, bx1)
            inter_y1 = max(ay1, by1)
            inter_x2 = min(ax2, bx2)
            inter_y2 = min(ay2, by2)

            if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
                return 0.0

            inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
            area_a = (ax2 - ax1) * (ay2 - ay1)
            area_b = (bx2 - bx1) * (by2 - by1)
            union = area_a + area_b - inter_area
            if union <= 0:
                return 0.0
            return inter_area / union

        for cand in boxes_sorted:
            if all(iou(cand, k) < iou_thresh for k in kept):
                kept.append(cand)

        return kept

    # ───────────────────────────────────────────────────────────
    def loop(self):
        t0, frames = time.time(), 0

        MIN_BOX_AREA_RATIO = 0.0008
        IGNORE_TOP_RATIO = 0.30  # верхні 30% кадру ігноруємо

        while self.running and self.cap.isOpened():
            ok, frame_orig = self.cap.read()
            if not ok:
                break
            frames += 1

            if frame_orig is None or frame_orig.size == 0:
                continue

            frame_resized = cv2.resize(
                frame_orig,
                (self.MODEL_W, self.MODEL_H),
                interpolation=cv2.INTER_AREA,
            )

            h, w = frame_resized.shape[:2]
            frame_area = float(h * w)

            frame_proc = self.preprocess_frame(frame_resized)

            # замір часу інференсу
            inf_start = time.time()
            results_list = self.detector.predict(
                source=frame_proc,
                imgsz=self.MODEL_W,
                conf=self.conf_var.get(),
                verbose=False,
            )
            inf_end = time.time()
            inf_time = inf_end - inf_start
            self.total_inference_time += inf_time
            self.total_frames += 1

            results = results_list[0]

            # Збираємо кандидатів
            candidates = []

            for box in results.boxes:
                cls_i = int(box.cls)
                conf = float(box.conf)
                xyxy = box.xyxy[0].tolist()
                cls_name = self.class_names[cls_i]

                x1, y1, x2, y2 = map(int, xyxy)
                x1 = max(0, min(x1, w - 1))
                x2 = max(0, min(x2, w - 1))
                y1 = max(0, min(y1, h - 1))
                y2 = max(0, min(y2, h - 1))
                if x2 <= x1 or y2 <= y1:
                    continue

                box_w = x2 - x1
                box_h = y2 - y1
                box_area = float(box_w * box_h)

                if box_area < MIN_BOX_AREA_RATIO * frame_area:
                    continue

                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0

                if cy < IGNORE_TOP_RATIO * h:
                    continue

                candidates.append((cls_name, conf, (x1, y1, x2, y2)))

            # NMS-подібне злиття
            merged = self._merge_overlapping_boxes(candidates, iou_thresh=0.6)

            # Фільтр по класу
            objects = []
            for cls_name, conf, (x1, y1, x2, y2) in merged:
                if self.cls_var.get() != "All" and cls_name != self.cls_var.get():
                    continue

                objects.append((cls_name, conf, (x1, y1, x2, y2)))

                color = self.class_colors.get(cls_name, self.default_color)

                cv2.rectangle(frame_resized, (x1, y1), (x2, y2), color, 2)

                label = f"{cls_name}:{conf:.2f}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 1

                (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)

                tx = x1
                ty = y1 - 6

                if ty - text_h < 0:
                    ty = y1 + text_h + 6

                if tx + text_w > w:
                    tx = max(0, w - text_w - 2)

                cv2.rectangle(
                    frame_resized,
                    (tx, ty - text_h),
                    (tx + text_w, ty),
                    (0, 0, 0),
                    thickness=-1,
                )
                cv2.putText(
                    frame_resized,
                    label,
                    (tx, ty - 2),
                    font,
                    font_scale,
                    (255, 255, 255),
                    thickness,
                    lineType=cv2.LINE_AA,
                )

            # ----- ВІДНОСНИЙ ЧАС -----
            if self.start_time is not None:
                rel_time = time.time() - self.start_time
            else:
                rel_time = 0.0

            self.timer_lbl.config(text=f"Time: {rel_time:.2f} s")

            current_sec = int(rel_time)

            # Лог + таблиця раз на секунду
            if self.logger is not None and current_sec != self.last_logged_sec:
                self.last_logged_sec = current_sec
                self.logger.log_snapshot(rel_time, objects)

                if not objects:
                    self.tree.insert(
                        "",
                        "end",
                        values=(f"{rel_time:.2f}", "NO_OBJECTS", "-", "-"),
                    )
                else:
                    for cls_name, conf, (x1, y1, x2, y2) in objects:
                        self.tree.insert(
                            "",
                            "end",
                            values=(
                                f"{rel_time:.2f}",
                                cls_name,
                                f"{conf:.2f}",
                                f"{x1},{y1},{x2},{y2}",
                            ),
                        )

            # FPS у статусі
            if frames % 10 == 0:
                fps = frames / (time.time() - t0)
                self.status_lbl.config(
                    text=f"FPS: {fps:.1f} | Objects (last frame): {len(objects)}"
                )

            rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(Image.fromarray(rgb))
            self.video_lbl.configure(image=img)
            self.video_lbl.image = img

        # Після завершення — лог зведення
        if self.logger is not None and self.total_frames > 0:
            total_time = time.time() - t0
            avg_fps = self.total_frames / total_time
            avg_inf_ms = (self.total_inference_time / self.total_frames) * 1000.0
            self.logger.log_summary(avg_fps=avg_fps, avg_inference_ms=avg_inf_ms)

        self.cap.release()
        self.status_lbl.config(text="Stopped")
        self.video_lbl.config(image="")
        self.timer_lbl.config(text="Time: 0.00 s")
        self.running = False
        self.btn.config(text="Start")

    # ───────────────────────────────────────────────────────────
    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    AppGUI().run()
