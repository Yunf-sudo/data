from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageOps, ImageTk

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".avif"}
STATE_FILE = Path(__file__).with_name("advanced_label_tool_state.json")
REVIEW_CACHE_FILE = Path(__file__).with_name("advanced_label_tool_review_cache.json")
MIN_BOX_SIZE = 6
HANDLE_SIZE = 8
DEFAULT_CLASSES = [
    "beam", "bed", "broad_leaf_live", "cabinet", "coffee_table", "desk", "dining_table",
    "fake_plant", "floor_window", "main_door", "mirror", "normal_window", "room_door",
    "sharp_leaf_live", "sink", "sofa", "stairs", "stove", "toilet", "water_feature",
]
ZH = {
    "beam": "横梁", "bed": "床", "broad_leaf_live": "阔叶真植物", "cabinet": "柜子",
    "coffee_table": "茶几", "desk": "书桌", "dining_table": "餐桌", "fake_plant": "假植物",
    "floor_window": "落地窗", "main_door": "入户门", "mirror": "镜子", "normal_window": "普通窗",
    "room_door": "房门", "sharp_leaf_live": "尖叶真植物", "sink": "水槽", "sofa": "沙发",
    "stairs": "楼梯", "stove": "灶台", "toilet": "马桶", "water_feature": "水景/鱼缸",
}
COLORS = ["#ff4d4f", "#1677ff", "#52c41a", "#fa8c16", "#722ed1", "#13c2c2", "#eb2f96", "#2f54eb"]
REVIEW_FILTER_LABELS = {
    "待复核高风险": "pending_risk",
    "疑似重复框": "duplicate_boxes",
    "全部高风险": "high_risk",
    "未复核全部": "unreviewed_all",
    "全部图片": "all",
}


@dataclass
class YoloBox:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float

    @classmethod
    def from_line(cls, line: str) -> "YoloBox | None":
        parts = line.strip().split()
        if len(parts) != 5:
            return None
        try:
            cid = int(float(parts[0]))
            xc, yc, w, h = [float(x) for x in parts[1:]]
        except ValueError:
            return None
        return cls(cid, max(0.0, min(1.0, xc)), max(0.0, min(1.0, yc)), max(0.0, min(1.0, w)), max(0.0, min(1.0, h)))

    @classmethod
    def from_pixels(cls, class_id: int, box: tuple[float, float, float, float], iw: int, ih: int) -> "YoloBox":
        x1, y1, x2, y2 = box
        return cls(class_id, ((x1 + x2) / 2) / iw, ((y1 + y2) / 2) / ih, (x2 - x1) / iw, (y2 - y1) / ih)

    def to_line(self) -> str:
        return f"{self.class_id} {self.x_center:.6f} {self.y_center:.6f} {self.width:.6f} {self.height:.6f}"

    def to_pixels(self, iw: int, ih: int) -> tuple[float, float, float, float]:
        hw, hh = self.width * iw / 2, self.height * ih / 2
        cx, cy = self.x_center * iw, self.y_center * ih
        return cx - hw, cy - hh, cx + hw, cy + hh


class ClassDialog(tk.Toplevel):
    def __init__(self, parent: tk.Misc, classes: list[str], current: int | None = None):
        super().__init__(parent)
        self.title("选择类别")
        self.transient(parent)
        self.grab_set()
        self.result: int | None = None
        self.classes = classes
        self.option_add("*Font", ("Microsoft YaHei UI", 10))
        self.geometry("420x520")
        frame = ttk.Frame(self, padding=12)
        frame.pack(fill=tk.BOTH, expand=True)
        ttk.Label(frame, text="双击或回车确认").pack(anchor=tk.W)
        self.keyword = tk.StringVar()
        entry = ttk.Entry(frame, textvariable=self.keyword)
        entry.pack(fill=tk.X, pady=(6, 8))
        entry.bind("<KeyRelease>", self._refresh)
        self.listbox = tk.Listbox(frame, exportselection=False)
        self.listbox.pack(fill=tk.BOTH, expand=True)
        self.listbox.bind("<Double-Button-1>", lambda _e: self._ok())
        self.listbox.bind("<Return>", lambda _e: self._ok())
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=(8, 0))
        ttk.Button(row, text="确定", command=self._ok).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(row, text="取消", command=self._cancel).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 0))
        self.visible: list[int] = []
        self._refresh()
        if current is not None and current in self.visible:
            idx = self.visible.index(current)
            self.listbox.selection_set(idx)
            self.listbox.see(idx)
        elif self.visible:
            self.listbox.selection_set(0)
        self.bind("<Escape>", lambda _e: self._cancel())
        entry.focus_set()

    def _refresh(self, _event: object | None = None) -> None:
        key = self.keyword.get().strip().lower()
        self.listbox.delete(0, tk.END)
        self.visible = []
        for idx, name in enumerate(self.classes):
            row = f"{idx:02d}. {ZH.get(name, name)} ({name})"
            if key and key not in row.lower():
                continue
            self.visible.append(idx)
            self.listbox.insert(tk.END, row)
        if self.visible and not self.listbox.curselection():
            self.listbox.selection_set(0)

    def _ok(self) -> None:
        sel = self.listbox.curselection()
        if sel:
            self.result = self.visible[int(sel[0])]
        self.destroy()

    def _cancel(self) -> None:
        self.result = None
        self.destroy()


class AdvancedLabelTool:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("中文标注修改工具 - YOLO/RT-DETR")
        self.root.geometry("1600x960")
        self.root.minsize(1200, 760)
        self.root.option_add("*Font", ("Microsoft YaHei UI", 10))
        self.images_dir: Path | None = None
        self.labels_dir: Path | None = None
        self.classes: list[str] = list(DEFAULT_CLASSES)
        self.all_image_paths: list[Path] = []
        self.image_paths: list[Path] = []
        self.current_index = 0
        self.current_image_path: Path | None = None
        self.current_image: Image.Image | None = None
        self.current_size = (1, 1)
        self.boxes: list[YoloBox] = []
        self.selected: int | None = None
        self.tk_image: ImageTk.PhotoImage | None = None
        self.canvas_rect = (0, 0, 1, 1)
        self.cache_key: tuple[str, int, int] | None = None
        self.mode: str | None = None
        self.start_pt: tuple[float, float] | None = None
        self.preview_box: tuple[float, float, float, float] | None = None
        self.anchor_box: tuple[float, float, float, float] | None = None
        self.handle_name: str | None = None
        self.move_offset = (0.0, 0.0)
        self.status = tk.StringVar(value="先选择图片目录和标签目录。")
        self.path_var = tk.StringVar(value="图片目录：未选择")
        self.label_var = tk.StringVar(value="标签目录：未选择")
        self.info_var = tk.StringVar(value="当前图片：")
        self.count_var = tk.StringVar(value="进度：0 / 0")
        self.flags_var = tk.StringVar(value="复核标记：无")
        self.jump_var = tk.StringVar()
        self.last_image_by_dir: dict[str, str] = {}
        self.manifest_flags_by_relpath: dict[str, set[str]] = {}
        self.local_flags_by_relpath: dict[str, set[str]] = {}
        self.reviewed_images: dict[str, bool] = {}
        self.review_mode_var = tk.BooleanVar(value=True)
        self.review_filter_var = tk.StringVar(value="待复核高风险")
        self.dirty = False
        self._build_ui()
        self._load_state()
        self.root.bind("<Key>", self._on_key)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        left = ttk.Frame(self.root, padding=12)
        left.pack(side=tk.LEFT, fill=tk.Y)
        right = ttk.Frame(self.root, padding=(0, 12, 12, 12))
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ttk.Button(left, text="选择图片目录", command=self._choose_images).pack(fill=tk.X, pady=(0, 6))
        ttk.Label(left, textvariable=self.path_var, wraplength=320, justify=tk.LEFT).pack(fill=tk.X, pady=(0, 8))
        ttk.Button(left, text="选择标签目录", command=self._choose_labels).pack(fill=tk.X, pady=(0, 6))
        ttk.Label(left, textvariable=self.label_var, wraplength=320, justify=tk.LEFT).pack(fill=tk.X, pady=(0, 8))
        row = ttk.Frame(left)
        row.pack(fill=tk.X, pady=(0, 8))
        ttk.Button(row, text="从 data.yaml 读取", command=self._load_classes_yaml).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(row, text="刷新", command=self._reload).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 0))
        row_jump = ttk.Frame(left)
        row_jump.pack(fill=tk.X, pady=(0, 8))
        ttk.Button(row_jump, text="选图", command=self._choose_image_file).pack(side=tk.LEFT)
        ttk.Entry(row_jump, textvariable=self.jump_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 6))
        ttk.Button(row_jump, text="跳转", command=self._jump_to_image).pack(side=tk.LEFT)
        row_review = ttk.Frame(left)
        row_review.pack(fill=tk.X, pady=(0, 8))
        ttk.Checkbutton(row_review, text="复核模式", variable=self.review_mode_var, command=self._apply_review_filter).pack(side=tk.LEFT)
        self.review_filter_box = ttk.Combobox(row_review, textvariable=self.review_filter_var, values=list(REVIEW_FILTER_LABELS.keys()), state="readonly", width=12)
        self.review_filter_box.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 6))
        self.review_filter_box.bind("<<ComboboxSelected>>", lambda _e: self._apply_review_filter())
        ttk.Button(row_review, text="通过并下一张", command=self._approve_and_next).pack(side=tk.LEFT)
        ttk.Label(left, text="当前框").pack(anchor=tk.W)
        self.box_list = tk.Listbox(left, height=14, exportselection=False)
        self.box_list.pack(fill=tk.BOTH, pady=(4, 8))
        self.box_list.bind("<<ListboxSelect>>", self._select_from_list)
        self.box_list.bind("<Double-Button-1>", lambda _e: self._change_class())
        row2 = ttk.Frame(left)
        row2.pack(fill=tk.X, pady=(0, 8))
        ttk.Button(row2, text="上一张", command=self._prev).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(row2, text="下一张", command=self._next).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)
        ttk.Button(row2, text="保存", command=self._save).pack(side=tk.LEFT, fill=tk.X, expand=True)
        row3 = ttk.Frame(left)
        row3.pack(fill=tk.X, pady=(0, 8))
        ttk.Button(row3, text="改类别", command=self._change_class).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(row3, text="删框", command=self._delete_selected).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)
        ttk.Button(row3, text="空图", command=self._mark_empty).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(left, textvariable=self.count_var).pack(anchor=tk.W)
        ttk.Label(left, textvariable=self.flags_var, wraplength=320, justify=tk.LEFT).pack(fill=tk.X, pady=(4, 0))
        ttk.Label(left, textvariable=self.info_var, wraplength=320, justify=tk.LEFT).pack(fill=tk.X, pady=(4, 8))
        ttk.Label(left, text="快捷键\nEnter/C/空格：改类别\nDelete/BackSpace：删框\nA/← 上一张，D/→ 下一张\nQ 复核通过并下一张，G 跳转\nS 保存，E 空图\n左键拖空白画框，拖框移动，拖控制点缩放", wraplength=320, justify=tk.LEFT).pack(fill=tk.X)
        self.canvas = tk.Canvas(right, bg="#1f1f1f", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<ButtonPress-1>", self._press)
        self.canvas.bind("<B1-Motion>", self._drag)
        self.canvas.bind("<ButtonRelease-1>", self._release)
        self.canvas.bind("<Configure>", lambda _e: self._render())
        self.canvas.bind("<Double-Button-1>", lambda _e: self._change_class())
        ttk.Label(right, textvariable=self.status, relief=tk.GROOVE, anchor=tk.W, padding=(10, 8)).pack(fill=tk.X, pady=(8, 0))

    def _load_state(self) -> None:
        state: dict[str, object] = {}
        if STATE_FILE.exists():
            try:
                state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
            except Exception:
                state = {}
        self.classes = [str(x) for x in (state.get("classes") or DEFAULT_CLASSES)]
        raw_last = state.get("last_image_by_dir") or {}
        if isinstance(raw_last, dict):
            self.last_image_by_dir = {str(k): str(v) for k, v in raw_last.items() if k and v}
        raw_reviewed = state.get("reviewed_images") or {}
        if isinstance(raw_reviewed, dict):
            self.reviewed_images = {str(k): bool(v) for k, v in raw_reviewed.items() if k}
        self.review_mode_var.set(bool(state.get("review_mode", True)))
        review_filter = str(state.get("review_filter") or "待复核高风险")
        if review_filter in REVIEW_FILTER_LABELS:
            self.review_filter_var.set(review_filter)
        if state.get("images_dir") and Path(str(state["images_dir"])).exists():
            self.images_dir = Path(str(state["images_dir"]))
            self.path_var.set(f"图片目录：{self.images_dir}")
        if state.get("labels_dir") and Path(str(state["labels_dir"])).exists():
            self.labels_dir = Path(str(state["labels_dir"]))
            self.label_var.set(f"标签目录：{self.labels_dir}")
        if self.images_dir:
            self._refresh_images()
            self._restore_last_image(str(state.get("last_image") or ""))
            self._load_image()
        self._save_state()

    def _save_state(self) -> None:
        if self.images_dir and self.current_image_path:
            self.last_image_by_dir[str(self.images_dir.resolve())] = str(self.current_image_path.resolve())
        payload = {
            "images_dir": str(self.images_dir) if self.images_dir else "",
            "labels_dir": str(self.labels_dir) if self.labels_dir else "",
            "classes": self.classes,
            "last_image": str(self.current_image_path) if self.current_image_path else "",
            "last_image_by_dir": self.last_image_by_dir,
            "review_mode": bool(self.review_mode_var.get()),
            "review_filter": self.review_filter_var.get(),
            "reviewed_images": self.reviewed_images,
        }
        STATE_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _class_text(self, cid: int) -> str:
        if 0 <= cid < len(self.classes):
            name = self.classes[cid]
            return f"{ZH.get(name, name)} ({name})"
        return f"未定义类别 {cid}"

    def _choose_images(self) -> None:
        path = filedialog.askdirectory(title="请选择图片目录", initialdir=str(self.images_dir or Path.cwd()))
        if not path:
            return
        self.images_dir = Path(path)
        self.path_var.set(f"图片目录：{self.images_dir}")
        if self.labels_dir is None:
            self.labels_dir = self._guess_labels_dir(self.images_dir)
            self.label_var.set(f"标签目录：{self.labels_dir}")
        self._refresh_images()
        self._restore_last_image()
        self._load_image()
        self._save_state()

    def _choose_labels(self) -> None:
        path = filedialog.askdirectory(title="请选择标签目录", initialdir=str(self.labels_dir or self.images_dir or Path.cwd()))
        if not path:
            return
        self.labels_dir = Path(path)
        self.label_var.set(f"标签目录：{self.labels_dir}")
        self._refresh_images()
        self._load_image()
        self._save_state()

    def _guess_labels_dir(self, images_dir: Path) -> Path:
        if images_dir.parent.name.lower() == "images":
            return images_dir.parent.parent / "labels_prelabel" / images_dir.name
        if images_dir.name.lower() == "images":
            return images_dir.parent / "labels_prelabel"
        return images_dir.parent / f"{images_dir.name}_labels"

    def _label_path(self, image_path: Path) -> Path | None:
        if not self.images_dir or not self.labels_dir:
            return None
        return self.labels_dir / image_path.relative_to(self.images_dir).with_suffix(".txt")

    def _image_relpath_key(self, image_path: Path) -> str:
        if self.images_dir is None:
            return image_path.name
        try:
            return image_path.relative_to(self.images_dir).as_posix()
        except ValueError:
            return image_path.name

    def _image_state_key(self, image_path: Path) -> str:
        return str(image_path.resolve())

    def _find_manifest(self) -> Path | None:
        if not self.images_dir:
            return None
        for base in [self.images_dir.resolve(), *self.images_dir.resolve().parents]:
            path = base / "review_manifest.csv"
            if path.exists():
                return path
        return None

    def _review_cache_key(self) -> str:
        images = str(self.images_dir.resolve()) if self.images_dir else ""
        labels = str(self.labels_dir.resolve()) if self.labels_dir else ""
        return f"{images}|{labels}"

    def _refresh_images(self) -> None:
        self.all_image_paths = sorted(
            p for p in self.images_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ) if self.images_dir else []
        self._load_review_metadata()
        if self.review_mode_var.get():
            mode = REVIEW_FILTER_LABELS.get(self.review_filter_var.get(), "pending_risk")
            self.image_paths = [path for path in self.all_image_paths if self._include_in_review_queue(path, mode)]
        else:
            self.image_paths = list(self.all_image_paths)
        self._update_count_var()

    def _update_count_var(self) -> None:
        current = min(self.current_index + 1, len(self.image_paths)) if self.image_paths else 0
        if self.review_mode_var.get():
            self.count_var.set(f"进度：{current} / {len(self.image_paths)} | 全量 {len(self.all_image_paths)}")
        else:
            self.count_var.set(f"进度：{current} / {len(self.image_paths)}")

    def _load_review_metadata(self) -> None:
        self._load_manifest_flags()
        self.local_flags_by_relpath = self._load_local_review_cache()

    def _load_manifest_flags(self) -> None:
        self.manifest_flags_by_relpath = {}
        manifest_path = self._find_manifest()
        if manifest_path is None:
            return
        try:
            with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    rel = str(row.get("export_image") or "").replace("\\", "/").strip()
                    if not rel:
                        continue
                    flags = {item for item in str(row.get("flags") or "").split(";") if item}
                    if flags:
                        self.manifest_flags_by_relpath[rel] = flags
        except Exception:
            self.manifest_flags_by_relpath = {}

    def _load_local_review_cache(self) -> dict[str, set[str]]:
        if not REVIEW_CACHE_FILE.exists():
            return {}
        try:
            payload = json.loads(REVIEW_CACHE_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
        datasets = payload.get("datasets") or {}
        entry = datasets.get(self._review_cache_key()) or {}
        flags_by_relpath = entry.get("flags_by_relpath") or {}
        loaded: dict[str, set[str]] = {}
        for rel, flags in flags_by_relpath.items():
            if rel and isinstance(flags, list):
                loaded[str(rel)] = {str(item) for item in flags if item}
        return loaded

    def _save_local_review_cache(self) -> None:
        try:
            payload = json.loads(REVIEW_CACHE_FILE.read_text(encoding="utf-8")) if REVIEW_CACHE_FILE.exists() else {}
        except Exception:
            payload = {}
        datasets = payload.setdefault("datasets", {})
        datasets[self._review_cache_key()] = {
            "images_dir": str(self.images_dir.resolve()) if self.images_dir else "",
            "labels_dir": str(self.labels_dir.resolve()) if self.labels_dir else "",
            "flags_by_relpath": {
                rel: sorted(flags)
                for rel, flags in sorted(self.local_flags_by_relpath.items())
                if flags
            },
        }
        REVIEW_CACHE_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _flags_for_image(self, image_path: Path) -> set[str]:
        rel = self._image_relpath_key(image_path)
        return set(self.manifest_flags_by_relpath.get(rel, set())) | set(self.local_flags_by_relpath.get(rel, set()))

    def _include_in_review_queue(self, image_path: Path, mode: str) -> bool:
        flags = self._flags_for_image(image_path)
        reviewed = self.reviewed_images.get(self._image_state_key(image_path), False)
        if mode == "all":
            return True
        if mode == "unreviewed_all":
            return not reviewed
        if mode == "high_risk":
            return bool(flags) and not reviewed
        if mode == "duplicate_boxes":
            return "same_class_overlap" in flags and not reviewed
        return bool(flags) and not reviewed

    def _read_labels_for_path(self, image_path: Path) -> list[YoloBox]:
        label_path = self._label_path(image_path)
        if label_path is None or not label_path.exists():
            return []
        return [box for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines() if (box := YoloBox.from_line(line))]

    def _box_norm_coords(self, box: YoloBox) -> tuple[float, float, float, float]:
        return (
            box.x_center - box.width / 2,
            box.y_center - box.height / 2,
            box.x_center + box.width / 2,
            box.y_center + box.height / 2,
        )

    def _is_suspicious_same_class_pair(self, first: YoloBox, second: YoloBox) -> bool:
        ax1, ay1, ax2, ay2 = self._box_norm_coords(first)
        bx1, by1, bx2, by2 = self._box_norm_coords(second)
        inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
        inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
        inter = inter_w * inter_h
        if inter <= 0:
            return False
        area_a = max(first.width * first.height, 1e-6)
        area_b = max(second.width * second.height, 1e-6)
        union = area_a + area_b - inter
        iou = inter / union if union > 0 else 0.0
        cover = inter / min(area_a, area_b)
        area_ratio = max(area_a, area_b) / max(min(area_a, area_b), 1e-6)
        center_dx = abs(first.x_center - second.x_center) / max(min(first.width, second.width), 1e-6)
        center_dy = abs(first.y_center - second.y_center) / max(min(first.height, second.height), 1e-6)
        return cover >= 0.72 or iou >= 0.45 or (area_ratio <= 1.8 and center_dx <= 0.35 and center_dy <= 0.35)

    def _detect_local_flags(self, image_path: Path) -> set[str]:
        boxes = self._read_labels_for_path(image_path)
        for idx, first in enumerate(boxes):
            for second in boxes[idx + 1:]:
                if first.class_id == second.class_id and self._is_suspicious_same_class_pair(first, second):
                    return {"same_class_overlap"}
        return set()

    def _refresh_current_local_flags(self) -> None:
        if self.current_image_path is None:
            return
        rel = self._image_relpath_key(self.current_image_path)
        flags = self._detect_local_flags(self.current_image_path)
        if flags:
            self.local_flags_by_relpath[rel] = flags
        else:
            self.local_flags_by_relpath.pop(rel, None)
        self._save_local_review_cache()
        self._update_flag_text()

    def _update_flag_text(self) -> None:
        if self.current_image_path is None:
            self.flags_var.set("复核标记：无")
            return
        current_flags = sorted(self._flags_for_image(self.current_image_path))
        reviewed = self.reviewed_images.get(self._image_state_key(self.current_image_path), False)
        flag_text = "；".join(current_flags) if current_flags else "无"
        if reviewed:
            flag_text = f"{flag_text} | 已复核"
        self.flags_var.set(f"复核标记：{flag_text}")

    def _restore_last_image(self, fallback_last: str | None = None) -> None:
        if not self.images_dir or not self.image_paths:
            self.current_index = 0
            return
        candidates: list[Path] = []
        remembered = self.last_image_by_dir.get(str(self.images_dir.resolve()))
        if remembered:
            candidates.append(Path(remembered))
        if fallback_last:
            candidates.append(Path(fallback_last))
        for candidate in candidates:
            try:
                self.current_index = self.image_paths.index(candidate)
                return
            except ValueError:
                continue
        self.current_index = max(0, min(self.current_index, len(self.image_paths) - 1))

    def _load_image(self) -> None:
        if not self.image_paths:
            self.current_image = None
            self.current_image_path = None
            self.boxes = []
            self.selected = None
            self.dirty = False
            self.flags_var.set("复核标记：当前筛选没有图片")
            self._update_count_var()
            self._render()
            return
        self.current_index = max(0, min(self.current_index, len(self.image_paths) - 1))
        self.current_image_path = self.image_paths[self.current_index]
        try:
            with Image.open(self.current_image_path) as img:
                self.current_image = ImageOps.exif_transpose(img).convert("RGB")
        except Exception as exc:
            messagebox.showerror("打开图片失败", f"{self.current_image_path}\n\n{exc}")
            return
        self.current_size = self.current_image.size
        self.boxes = self._read_labels()
        self.selected = None
        self.mode = None
        self.preview_box = None
        self.anchor_box = None
        self.handle_name = None
        self.cache_key = None
        self.dirty = False
        self._update_count_var()
        self.info_var.set(f"当前图片：{self.current_image_path.name}")
        self._update_flag_text()
        self._refresh_box_list()
        self._render()
        self._save_state()

    def _read_labels(self) -> list[YoloBox]:
        if not self.current_image_path:
            return []
        return self._read_labels_for_path(self.current_image_path)

    def _save_quietly(self) -> None:
        if not self.current_image_path or not self.labels_dir:
            self._save_state()
            return
        label_path = self._label_path(self.current_image_path)
        if label_path is None:
            self._save_state()
            return
        try:
            label_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.write_text("\n".join(box.to_line() for box in self.boxes), encoding="utf-8")
            if self.dirty:
                self.reviewed_images[self._image_state_key(self.current_image_path)] = True
                self.dirty = False
            self._refresh_current_local_flags()
            self._save_state()
        except Exception:
            self._save_state()

    def _prune_current_from_review_queue(self, image_path: Path | None) -> None:
        if not self.review_mode_var.get() or image_path is None:
            return
        mode = REVIEW_FILTER_LABELS.get(self.review_filter_var.get(), "pending_risk")
        if self._include_in_review_queue(image_path, mode):
            return
        try:
            idx = self.image_paths.index(image_path)
        except ValueError:
            return
        self.image_paths.pop(idx)
        if idx < self.current_index:
            self.current_index -= 1
        elif idx == self.current_index:
            self.current_index = min(self.current_index, max(0, len(self.image_paths) - 1))
        self._update_count_var()

    def _save(self) -> None:
        if not self.current_image_path:
            return
        label_path = self._label_path(self.current_image_path)
        if label_path is None:
            messagebox.showwarning("缺少标签目录", "请先选择标签目录。")
            return
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text("\n".join(box.to_line() for box in self.boxes), encoding="utf-8")
        if self.dirty:
            self.reviewed_images[self._image_state_key(self.current_image_path)] = True
            self.dirty = False
        self._refresh_current_local_flags()
        self.status.set(f"已保存：{label_path.name}")
        self._save_state()

    def _autosave(self) -> None:
        self._save()
        self._refresh_box_list()
        self._render()

    def _refresh_box_list(self) -> None:
        self.box_list.delete(0, tk.END)
        for idx, box in enumerate(self.boxes):
            self.box_list.insert(tk.END, f"{idx + 1}. {self._class_text(box.class_id)}")
        if self.selected is not None and self.selected < len(self.boxes):
            self.box_list.selection_set(self.selected)

    def _select_from_list(self, _event: object) -> None:
        sel = self.box_list.curselection()
        if sel:
            self.selected = int(sel[0])
            self._render()
            self.canvas.focus_set()

    def _prev(self) -> None:
        if self.current_index > 0:
            current_path = self.current_image_path
            self._save_quietly()
            self._prune_current_from_review_queue(current_path)
            self.current_index -= 1
            self._load_image()

    def _next(self) -> None:
        if self.current_index < len(self.image_paths) - 1:
            current_path = self.current_image_path
            self._save_quietly()
            self._prune_current_from_review_queue(current_path)
            self.current_index += 1
            self._load_image()

    def _mark_empty(self) -> None:
        self.boxes = []
        self.selected = None
        self.dirty = True
        self._autosave()
        self.status.set("当前图片已标记为空图。")

    def _ask_class(self, current: int | None = None) -> int | None:
        dlg = ClassDialog(self.root, self.classes, current)
        self.root.wait_window(dlg)
        return dlg.result

    def _change_class(self) -> None:
        if self.selected is None or self.selected >= len(self.boxes):
            self.status.set("请先选中一个框。")
            return
        cid = self._ask_class(self.boxes[self.selected].class_id)
        if cid is None:
            return
        self.boxes[self.selected].class_id = cid
        self.dirty = True
        self._autosave()
        self.status.set(f"已修改类别：{self._class_text(cid)}")

    def _delete_selected(self) -> None:
        if self.selected is None or self.selected >= len(self.boxes):
            self.status.set("没有选中的框可删除。")
            return
        removed = self.boxes.pop(self.selected)
        self.selected = None
        self.dirty = True
        self._autosave()
        self.status.set(f"已删除框：{self._class_text(removed.class_id)}")

    def _load_classes_yaml(self) -> None:
        path = self._find_yaml()
        if path is None:
            path_str = filedialog.askopenfilename(title="请选择 data.yaml", filetypes=[("YAML", "*.yaml *.yml"), ("所有文件", "*.*")], initialdir=str(Path.cwd()))
            if not path_str:
                return
            path = Path(path_str)
        classes = self._parse_yaml(path)
        if not classes:
            messagebox.showwarning("读取失败", "没有找到 names 列表。")
            return
        self.classes = classes
        self._refresh_box_list()
        self._save_state()
        self.status.set(f"已读取 {len(classes)} 个类别。")

    def _find_yaml(self) -> Path | None:
        if not self.images_dir:
            return None
        for base in [self.images_dir.resolve(), *self.images_dir.resolve().parents]:
            for name in ("data.yaml", "data.yml"):
                path = base / name
                if path.exists():
                    return path
        return None

    def _parse_yaml(self, yaml_path: Path) -> list[str]:
        lines = yaml_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        started, names_map, names_list = False, {}, []
        for line in lines:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if s == "names:":
                started = True
                continue
            if not started:
                continue
            if ":" in s:
                left, right = s.split(":", 1)
                if left.strip().isdigit():
                    names_map[int(left.strip())] = right.strip().strip("\"'")
                    continue
            if s.startswith("- "):
                names_list.append(s[2:].strip().strip("\"'"))
                continue
            if not line.startswith((" ", "\t")):
                break
        if names_map:
            return [names_map.get(i, f"class_{i}") for i in range(max(names_map) + 1)]
        return names_list

    def _reload(self) -> None:
        current_path = self.current_image_path
        self._save_quietly()
        self._prune_current_from_review_queue(current_path)
        self._refresh_images()
        self._restore_last_image()
        self._load_image()

    def _apply_review_filter(self) -> None:
        current_path = self.current_image_path
        self._save_quietly()
        self._prune_current_from_review_queue(current_path)
        current_index = self.current_index
        self._refresh_images()
        if current_path and current_path in self.image_paths:
            self.current_index = self.image_paths.index(current_path)
        else:
            self.current_index = min(current_index, max(0, len(self.image_paths) - 1))
        self._load_image()
        self._save_state()

    def _approve_and_next(self) -> None:
        if self.current_image_path is None:
            return
        current_path = self.current_image_path
        self._save_quietly()
        approved_name = self.current_image_path.name
        self.reviewed_images[self._image_state_key(self.current_image_path)] = True
        if self.review_mode_var.get():
            mode = REVIEW_FILTER_LABELS.get(self.review_filter_var.get(), "pending_risk")
            if not self._include_in_review_queue(current_path, mode):
                self.image_paths.pop(self.current_index)
                if self.current_index >= len(self.image_paths):
                    self.current_index = max(0, len(self.image_paths) - 1)
                self._load_image()
            else:
                self._next()
        else:
            self._next()
        self.status.set(f"已标记复核完成：{approved_name}")
        self._save_state()

    def _choose_image_file(self) -> None:
        if not self.images_dir:
            messagebox.showwarning("未选择图片目录", "请先选择图片目录。")
            return
        current_path = self.current_image_path
        self._save_quietly()
        self._prune_current_from_review_queue(current_path)
        path = filedialog.askopenfilename(title="选择要跳转的图片", initialdir=str(self.images_dir), filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp *.avif"), ("All files", "*.*")])
        if not path:
            return
        target = Path(path)
        if target not in self.image_paths:
            if target in self.all_image_paths:
                self.review_mode_var.set(False)
                self._apply_review_filter()
            else:
                messagebox.showwarning("不在当前目录", f"{target}\n\n不在当前图片列表里。")
                return
        self.current_index = self.image_paths.index(target)
        self.jump_var.set(target.name)
        self._load_image()

    def _jump_to_image(self) -> None:
        if not self.all_image_paths:
            return
        raw = self.jump_var.get().strip()
        if not raw:
            return
        current_path = self.current_image_path
        self._save_quietly()
        self._prune_current_from_review_queue(current_path)
        target: Path | None = None
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(self.image_paths):
                target = self.image_paths[idx]
            elif 0 <= idx < len(self.all_image_paths):
                target = self.all_image_paths[idx]
        else:
            raw_lower = raw.lower()
            for path in self.image_paths or self.all_image_paths:
                rel = self._image_relpath_key(path).lower()
                if raw_lower in path.name.lower() or raw_lower in rel:
                    target = path
                    break
            if target is None:
                for path in self.all_image_paths:
                    rel = self._image_relpath_key(path).lower()
                    if raw_lower in path.name.lower() or raw_lower in rel:
                        target = path
                        break
        if target is None:
            messagebox.showwarning("未找到图片", f"没有找到：{raw}")
            return
        if target not in self.image_paths:
            self.review_mode_var.set(False)
            self._apply_review_filter()
        self.current_index = self.image_paths.index(target)
        self._load_image()

    def _to_image(self, cx: int, cy: int, clamp: bool) -> tuple[float, float] | None:
        x0, y0, dw, dh = self.canvas_rect
        if dw <= 1 or dh <= 1 or self.current_image is None:
            return None
        rx, ry = cx - x0, cy - y0
        if clamp:
            rx, ry = max(0, min(dw, rx)), max(0, min(dh, ry))
        elif rx < 0 or ry < 0 or rx > dw or ry > dh:
            return None
        iw, ih = self.current_size
        return rx / dw * iw, ry / dh * ih

    def _to_canvas(self, ix: float, iy: float) -> tuple[float, float]:
        x0, y0, dw, dh = self.canvas_rect
        iw, ih = self.current_size
        return x0 + (ix / iw) * dw, y0 + (iy / ih) * dh

    def _norm_box(self, box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        iw, ih = self.current_size
        x1, y1, x2, y2 = box
        x1, x2 = sorted((max(0.0, min(iw - 1, x1)), max(0.0, min(iw - 1, x2))))
        y1, y2 = sorted((max(0.0, min(ih - 1, y1)), max(0.0, min(ih - 1, y2))))
        return x1, y1, x2, y2

    def _box_at(self, ix: float, iy: float) -> int | None:
        for idx in range(len(self.boxes) - 1, -1, -1):
            x1, y1, x2, y2 = self.boxes[idx].to_pixels(*self.current_size)
            if x1 <= ix <= x2 and y1 <= iy <= y2:
                return idx
        return None

    def _handles(self, box: tuple[float, float, float, float]) -> dict[str, tuple[float, float]]:
        x1, y1, x2, y2 = box
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        return {"nw": (x1, y1), "n": (mx, y1), "ne": (x2, y1), "e": (x2, my), "se": (x2, y2), "s": (mx, y2), "sw": (x1, y2), "w": (x1, my)}

    def _handle_at(self, ix: float, iy: float) -> str | None:
        if self.selected is None or self.selected >= len(self.boxes):
            return None
        for name, (hx, hy) in self._handles(self.boxes[self.selected].to_pixels(*self.current_size)).items():
            if abs(ix - hx) <= HANDLE_SIZE and abs(iy - hy) <= HANDLE_SIZE:
                return name
        return None

    def _update_selected_box(self, box: tuple[float, float, float, float]) -> None:
        if self.selected is None:
            return
        x1, y1, x2, y2 = self._norm_box(box)
        if x2 - x1 < MIN_BOX_SIZE or y2 - y1 < MIN_BOX_SIZE:
            return
        cid = self.boxes[self.selected].class_id
        self.boxes[self.selected] = YoloBox.from_pixels(cid, (x1, y1, x2, y2), *self.current_size)

    def _press(self, event: tk.Event) -> None:
        point = self._to_image(event.x, event.y, clamp=False)
        if point is None or self.current_image is None:
            return
        handle = self._handle_at(*point)
        if handle:
            self.mode, self.handle_name = "resize", handle
            self.anchor_box = self.boxes[self.selected].to_pixels(*self.current_size) if self.selected is not None else None
            return
        hit = self._box_at(*point)
        if hit is not None:
            self.selected = hit
            self.anchor_box = self.boxes[hit].to_pixels(*self.current_size)
            x1, y1, _, _ = self.anchor_box
            self.move_offset = (point[0] - x1, point[1] - y1)
            self.mode = "move"
            self._refresh_box_list()
            self._render()
            return
        self.selected = None
        self._refresh_box_list()
        self.mode, self.start_pt = "draw", point
        self.preview_box = (point[0], point[1], point[0], point[1])
        self._render()

    def _drag(self, event: tk.Event) -> None:
        point = self._to_image(event.x, event.y, clamp=True)
        if point is None or self.mode is None:
            return
        if self.mode == "draw" and self.start_pt is not None:
            self.preview_box = (self.start_pt[0], self.start_pt[1], point[0], point[1])
            self._render()
            return
        if self.mode == "move" and self.anchor_box is not None and self.selected is not None:
            x1, y1, x2, y2 = self.anchor_box
            bw, bh = x2 - x1, y2 - y1
            nx1, ny1 = point[0] - self.move_offset[0], point[1] - self.move_offset[1]
            self._update_selected_box((nx1, ny1, nx1 + bw, ny1 + bh))
            self._render()
            return
        if self.mode == "resize" and self.anchor_box is not None and self.handle_name is not None:
            x1, y1, x2, y2 = self.anchor_box
            px, py = point
            if "w" in self.handle_name:
                x1 = px
            if "e" in self.handle_name:
                x2 = px
            if "n" in self.handle_name:
                y1 = py
            if "s" in self.handle_name:
                y2 = py
            self._update_selected_box((x1, y1, x2, y2))
            self._render()

    def _release(self, event: tk.Event) -> None:
        point = self._to_image(event.x, event.y, clamp=True)
        old_mode = self.mode
        self.mode, self.anchor_box, self.handle_name = None, None, None
        if old_mode == "draw" and self.start_pt is not None and point is not None:
            box = self._norm_box((self.start_pt[0], self.start_pt[1], point[0], point[1]))
            self.start_pt, self.preview_box = None, None
            if box[2] - box[0] >= MIN_BOX_SIZE and box[3] - box[1] >= MIN_BOX_SIZE:
                cid = self._ask_class()
                if cid is not None:
                    self.boxes.append(YoloBox.from_pixels(cid, box, *self.current_size))
                    self.selected = len(self.boxes) - 1
                    self.dirty = True
                    self._autosave()
                    self.status.set(f"已添加框：{self._class_text(cid)}")
                    return
        self.start_pt, self.preview_box = None, None
        if old_mode in {"move", "resize"}:
            self.dirty = True
            self._autosave()
        self._render()

    def _on_key(self, event: tk.Event) -> None:
        if isinstance(self.root.focus_get(), tk.Entry):
            return
        key = event.keysym
        if key in {"Delete", "BackSpace"}:
            self._delete_selected()
        elif key in {"Return", "KP_Enter", "space", "c", "C"}:
            self._change_class()
        elif key in {"q", "Q"}:
            self._approve_and_next()
        elif key in {"Left", "a", "A", "Prior"}:
            self._prev()
        elif key in {"Right", "d", "D", "Next"}:
            self._next()
        elif key in {"g", "G"}:
            self._jump_to_image()
        elif key in {"s", "S"}:
            self._save()
        elif key in {"e", "E"}:
            self._mark_empty()
        elif key == "Escape":
            self.mode, self.start_pt, self.preview_box, self.anchor_box, self.handle_name = None, None, None, None, None
            self._render()
            self.status.set("已取消当前操作。")

    def _on_close(self) -> None:
        self._save_quietly()
        self.root.destroy()

    def _render(self) -> None:
        self.canvas.delete("all")
        if self.current_image is None:
            self.canvas.create_text(self.canvas.winfo_width() // 2, self.canvas.winfo_height() // 2, text="请选择图片目录开始标注", fill="#fff", font=("Microsoft YaHei UI", 18))
            return
        cw, ch = max(self.canvas.winfo_width(), 50), max(self.canvas.winfo_height(), 50)
        iw, ih = self.current_size
        scale = min((cw - 20) / iw, (ch - 20) / ih)
        dw, dh = max(1, int(iw * scale)), max(1, int(ih * scale))
        ox, oy = max(0, (cw - dw) // 2), max(0, (ch - dh) // 2)
        self.canvas_rect = (ox, oy, dw, dh)
        key = (str(self.current_image_path), dw, dh)
        if key != self.cache_key:
            self.tk_image = ImageTk.PhotoImage(self.current_image.resize((dw, dh), Image.LANCZOS))
            self.cache_key = key
        self.canvas.create_image(ox, oy, image=self.tk_image, anchor=tk.NW)
        for idx, box in enumerate(self.boxes):
            x1, y1, x2, y2 = box.to_pixels(iw, ih)
            cx1, cy1 = self._to_canvas(x1, y1)
            cx2, cy2 = self._to_canvas(x2, y2)
            color = COLORS[box.class_id % len(COLORS)]
            self.canvas.create_rectangle(cx1, cy1, cx2, cy2, outline=color, width=3 if idx == self.selected else 2)
            label = self._class_text(box.class_id)
            lw = max(120, len(label) * 9)
            self.canvas.create_rectangle(cx1, cy1 - 24, cx1 + lw, cy1, fill=color, outline="")
            self.canvas.create_text(cx1 + 6, cy1 - 12, text=label, fill="#fff", anchor=tk.W, font=("Microsoft YaHei UI", 10, "bold"))
            if idx == self.selected:
                for hx, hy in self._handles((x1, y1, x2, y2)).values():
                    chx, chy = self._to_canvas(hx, hy)
                    self.canvas.create_rectangle(chx - HANDLE_SIZE, chy - HANDLE_SIZE, chx + HANDLE_SIZE, chy + HANDLE_SIZE, fill="#fff", outline=color, width=2)
        if self.preview_box is not None:
            x1, y1, x2, y2 = self._norm_box(self.preview_box)
            cx1, cy1 = self._to_canvas(x1, y1)
            cx2, cy2 = self._to_canvas(x2, y2)
            self.canvas.create_rectangle(cx1, cy1, cx2, cy2, outline="#ffd666", width=2, dash=(6, 4))


def main() -> None:
    root = tk.Tk()
    app = AdvancedLabelTool(root)
    app.status.set("工具已启动。左键画框，选中框后可改类、拖动、缩放。")
    root.mainloop()


if __name__ == "__main__":
    main()
