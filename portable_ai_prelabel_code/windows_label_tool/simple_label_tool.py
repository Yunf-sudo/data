from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
STATE_FILE = Path(__file__).with_name("label_tool_state.json")
MIN_BOX_SIZE = 6
DEFAULT_CLASSES = [
    "beam",
    "bed",
    "broad_leaf_live",
    "cabinet",
    "coffee_table",
    "desk",
    "dining_table",
    "fake_plant",
    "floor_window",
    "main_door",
    "mirror",
    "normal_window",
    "room_door",
    "sharp_leaf_live",
    "sink",
    "sofa",
    "stairs",
    "stove",
    "toilet",
    "water_feature",
]
BOX_COLORS = [
    "#ff4d4f",
    "#1677ff",
    "#52c41a",
    "#fa8c16",
    "#722ed1",
    "#13c2c2",
    "#eb2f96",
    "#2f54eb",
    "#a0d911",
    "#faad14",
]


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
            class_id = int(float(parts[0]))
            x_center, y_center, width, height = [float(item) for item in parts[1:]]
        except ValueError:
            return None
        return cls(
            class_id=class_id,
            x_center=max(0.0, min(1.0, x_center)),
            y_center=max(0.0, min(1.0, y_center)),
            width=max(0.0, min(1.0, width)),
            height=max(0.0, min(1.0, height)),
        )

    def to_line(self) -> str:
        return (
            f"{self.class_id} "
            f"{self.x_center:.6f} {self.y_center:.6f} "
            f"{self.width:.6f} {self.height:.6f}"
        )

    def to_pixel_box(self, image_width: int, image_height: int) -> tuple[float, float, float, float]:
        half_w = self.width * image_width / 2
        half_h = self.height * image_height / 2
        center_x = self.x_center * image_width
        center_y = self.y_center * image_height
        return (
            center_x - half_w,
            center_y - half_h,
            center_x + half_w,
            center_y + half_h,
        )


class SimpleLabelTool:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("简易中文标注工具 - RT-DETR/YOLO txt")
        self.root.geometry("1440x900")
        self.root.minsize(1100, 700)
        self.root.option_add("*Font", ("Microsoft YaHei UI", 10))

        self.images_dir: Path | None = None
        self.labels_dir: Path | None = None
        self.image_paths: list[Path] = []
        self.current_index = 0
        self.classes: list[str] = []
        self.boxes: list[YoloBox] = []
        self.selected_index: int | None = None
        self.pending_box: tuple[float, float, float, float] | None = None
        self.pending_digits = ""
        self.drag_start: tuple[float, float] | None = None
        self.drag_current: tuple[float, float] | None = None
        self.current_image: Image.Image | None = None
        self.current_image_path: Path | None = None
        self.current_image_size: tuple[int, int] = (1, 1)
        self.tk_image: ImageTk.PhotoImage | None = None
        self.render_cache_key: tuple[str, int, int] | None = None
        self.canvas_image_rect: tuple[int, int, int, int] = (0, 0, 1, 1)

        self.status_var = tk.StringVar(value="先选择图片文件夹，再开始标注。")
        self.path_var = tk.StringVar(value="图片文件夹：未选择")
        self.label_var = tk.StringVar(value="标签文件夹：未选择")
        self.image_info_var = tk.StringVar(value="当前图片：-")
        self.counter_var = tk.StringVar(value="进度：0 / 0")
        self.input_var = tk.StringVar(value="当前输入编号：-")

        self._build_ui()
        self._load_state()
        self.root.bind("<Key>", self._handle_keypress)

    def _read_classes_from_text(self) -> list[str]:
        lines = [
            line.strip()
            for line in self.class_text.get("1.0", tk.END).splitlines()
            if line.strip()
        ]
        return lines or list(DEFAULT_CLASSES)

    def _sync_classes_from_text(self, silent: bool = False) -> bool:
        lines = self._read_classes_from_text()
        if not lines:
            return False

        if lines == self.classes:
            return True

        self.classes = lines
        self.pending_digits = ""
        self.input_var.set("当前输入编号：-")
        self._save_state()
        self._refresh_box_listbox()
        self._render_canvas()
        if not silent:
            self._set_status(f"类别已更新，共 {len(self.classes)} 个。编号从 0 开始。")
        return True

    def _find_nearby_data_yaml(self) -> Path | None:
        if self.images_dir is None:
            return None

        candidates: list[Path] = []
        current = self.images_dir.resolve()

        for base in [current, *current.parents]:
            candidates.append(base / "data.yaml")
            candidates.append(base / "data.yml")

        images_parent = current.parent
        if images_parent.name.lower() == "images":
            dataset_root = images_parent.parent
            candidates.append(dataset_root / "data.yaml")
            candidates.append(dataset_root / "data.yml")

        seen: set[Path] = set()
        for path in candidates:
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            if resolved.exists() and resolved.is_file():
                return resolved

        return None

    def _build_ui(self) -> None:
        left = ttk.Frame(self.root, padding=12)
        left.pack(side=tk.LEFT, fill=tk.Y)

        right = ttk.Frame(self.root, padding=(0, 12, 12, 12))
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        ttk.Button(left, text="1. 选择图片文件夹", command=self._choose_images_dir).pack(
            fill=tk.X, pady=(0, 6)
        )
        ttk.Label(left, textvariable=self.path_var, wraplength=300, justify=tk.LEFT).pack(
            fill=tk.X, pady=(0, 8)
        )

        ttk.Button(left, text="2. 选择标签文件夹", command=self._choose_labels_dir).pack(
            fill=tk.X, pady=(0, 6)
        )
        ttk.Label(left, textvariable=self.label_var, wraplength=300, justify=tk.LEFT).pack(
            fill=tk.X, pady=(0, 8)
        )

        row = ttk.Frame(left)
        row.pack(fill=tk.X, pady=(0, 8))
        ttk.Button(row, text="从 data.yaml 读取类别", command=self._load_classes_from_yaml).pack(
            side=tk.LEFT, fill=tk.X, expand=True
        )
        ttk.Button(row, text="应用类别", command=self._apply_classes_from_text).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 0)
        )

        ttk.Label(left, text="类别列表（每行一个，自动编号）").pack(anchor=tk.W)
        self.class_text = tk.Text(left, width=34, height=14, undo=True)
        self.class_text.pack(fill=tk.X, pady=(4, 8))
        self.class_text.bind("<FocusOut>", lambda _event: self._sync_classes_from_text(silent=True))

        ttk.Label(left, text="当前图片的框").pack(anchor=tk.W)
        self.box_listbox = tk.Listbox(left, height=8, exportselection=False)
        self.box_listbox.pack(fill=tk.BOTH, pady=(4, 8))
        self.box_listbox.bind("<<ListboxSelect>>", self._on_box_list_select)

        action_row = ttk.Frame(left)
        action_row.pack(fill=tk.X, pady=(0, 8))
        ttk.Button(action_row, text="上一张", command=self._prev_image).pack(
            side=tk.LEFT, fill=tk.X, expand=True
        )
        ttk.Button(action_row, text="下一张", command=self._next_image).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=6
        )
        ttk.Button(action_row, text="保存", command=self._save_current_labels).pack(
            side=tk.LEFT, fill=tk.X, expand=True
        )

        action_row_2 = ttk.Frame(left)
        action_row_2.pack(fill=tk.X, pady=(0, 8))
        ttk.Button(action_row_2, text="删选中框", command=self._delete_selected_box).pack(
            side=tk.LEFT, fill=tk.X, expand=True
        )
        ttk.Button(action_row_2, text="当前图无目标", command=self._mark_empty_image).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 0)
        )

        ttk.Label(left, textvariable=self.counter_var).pack(anchor=tk.W, pady=(4, 0))
        ttk.Label(left, textvariable=self.image_info_var, wraplength=300, justify=tk.LEFT).pack(
            fill=tk.X, pady=(4, 8)
        )
        ttk.Label(left, textvariable=self.input_var, foreground="#d46b08").pack(
            anchor=tk.W, pady=(0, 8)
        )

        tip = (
            "操作说明\n"
            "1. 鼠标左键拖动框住物体。\n"
            "2. 画框后输入类别编号，按 Enter 确认。\n"
            "3. 单击已有框可选中，Delete 可删除。\n"
            "4. 左右方向键切换图片。\n"
            "5. E 表示当前图片没有目标。"
        )
        ttk.Label(left, text=tip, wraplength=300, justify=tk.LEFT).pack(fill=tk.X)

        self.canvas = tk.Canvas(right, bg="#1f1f1f", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<ButtonPress-1>", self._on_canvas_press)
        self.canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_canvas_release)
        self.canvas.bind("<Configure>", self._on_canvas_resize)

        status_bar = ttk.Label(
            right,
            textvariable=self.status_var,
            relief=tk.GROOVE,
            anchor=tk.W,
            padding=(10, 8),
        )
        status_bar.pack(fill=tk.X, pady=(8, 0))

    def _load_state(self) -> None:
        if STATE_FILE.exists():
            try:
                state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
            except Exception:
                state = {}
        else:
            state = {}

        classes = state.get("classes") or []
        if not classes:
            classes = list(DEFAULT_CLASSES)

        self.class_text.delete("1.0", tk.END)
        self.class_text.insert("1.0", "\n".join(classes))
        self.classes = classes

        image_dir = state.get("images_dir")
        if image_dir and Path(image_dir).exists():
            self.images_dir = Path(image_dir)
            self.path_var.set(f"图片文件夹：{self.images_dir}")

        label_dir = state.get("labels_dir")
        if label_dir and Path(label_dir).exists():
            self.labels_dir = Path(label_dir)
            self.label_var.set(f"标签文件夹：{self.labels_dir}")

        if self.images_dir is not None:
            self._refresh_image_list()
            last_image = state.get("last_image")
            if last_image:
                try:
                    last_path = Path(last_image)
                    self.current_index = self.image_paths.index(last_path)
                except ValueError:
                    self.current_index = 0
            self._load_current_image()

        self._save_state()

    def _save_state(self) -> None:
        payload = {
            "images_dir": str(self.images_dir) if self.images_dir else "",
            "labels_dir": str(self.labels_dir) if self.labels_dir else "",
            "classes": self.classes,
            "last_image": str(self.current_image_path) if self.current_image_path else "",
        }
        STATE_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _set_status(self, message: str) -> None:
        self.status_var.set(message)

    def _choose_images_dir(self) -> None:
        selected = filedialog.askdirectory(
            title="请选择图片文件夹",
            initialdir=str(self.images_dir or Path.cwd()),
        )
        if not selected:
            return
        self.images_dir = Path(selected)
        self.path_var.set(f"图片文件夹：{self.images_dir}")

        if self.labels_dir is None:
            guessed = self._guess_labels_dir(self.images_dir)
            self.labels_dir = guessed
            self.label_var.set(f"标签文件夹：{self.labels_dir}")

        self.current_index = 0
        self._refresh_image_list()
        self._load_current_image()
        self._save_state()

    def _choose_labels_dir(self) -> None:
        selected = filedialog.askdirectory(
            title="请选择标签文件夹",
            initialdir=str(self.labels_dir or self.images_dir or Path.cwd()),
        )
        if not selected:
            return
        self.labels_dir = Path(selected)
        self.label_var.set(f"标签文件夹：{self.labels_dir}")
        self._load_current_image()
        self._save_state()

    def _load_classes_from_yaml(self) -> None:
        auto_yaml = self._find_nearby_data_yaml()
        if auto_yaml is not None:
            classes = self._parse_classes_from_yaml(auto_yaml)
            if classes:
                self.class_text.delete("1.0", tk.END)
                self.class_text.insert("1.0", "\n".join(classes))
                self.classes = classes
                self._save_state()
                self._set_status(f"已自动读取 {auto_yaml.name}，共 {len(classes)} 个类别。")
                self._refresh_box_listbox()
                self._render_canvas()
                return

        yaml_path = filedialog.askopenfilename(
            title="请选择 data.yaml",
            filetypes=[("YAML 文件", "*.yaml *.yml"), ("所有文件", "*.*")],
            initialdir=str((self.images_dir or Path.cwd()).parent if self.images_dir else Path.cwd()),
        )
        if not yaml_path:
            return

        classes = self._parse_classes_from_yaml(Path(yaml_path))
        if not classes:
            messagebox.showwarning("读取失败", "没有在这个 YAML 里找到 names 列表。")
            return

        self.class_text.delete("1.0", tk.END)
        self.class_text.insert("1.0", "\n".join(classes))
        self.classes = classes
        self._save_state()
        self._set_status(f"已从 {Path(yaml_path).name} 读取 {len(classes)} 个类别。")
        self._refresh_box_listbox()
        self._render_canvas()

    def _apply_classes_from_text(self) -> None:
        if not self._sync_classes_from_text():
            messagebox.showwarning("类别为空", "请先在左侧输入类别，每行一个。")
            return

    def _guess_labels_dir(self, images_dir: Path) -> Path:
        if images_dir.parent.name.lower() == "images":
            return images_dir.parent.parent / "labels" / images_dir.name
        if images_dir.name.lower() == "images":
            return images_dir.parent / "labels"
        return images_dir.parent / f"{images_dir.name}_labels"

    def _refresh_image_list(self) -> None:
        if self.images_dir is None:
            self.image_paths = []
            self.counter_var.set("进度：0 / 0")
            return
        self.image_paths = sorted(
            path
            for path in self.images_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        total = len(self.image_paths)
        current = min(self.current_index + 1, total) if total else 0
        self.counter_var.set(f"进度：{current} / {total}")
        if total == 0:
            self._set_status("这个文件夹里没有找到图片。")

    def _label_path_for(self, image_path: Path) -> Path | None:
        if self.images_dir is None or self.labels_dir is None:
            return None
        relative = image_path.relative_to(self.images_dir)
        return self.labels_dir / relative.with_suffix(".txt")

    def _load_current_image(self) -> None:
        if not self.image_paths:
            self.current_image = None
            self.current_image_path = None
            self.boxes = []
            self.selected_index = None
            self.pending_box = None
            self.drag_start = None
            self.drag_current = None
            self.input_var.set("当前输入编号：-")
            self.image_info_var.set("当前图片：-")
            self.box_listbox.delete(0, tk.END)
            self._render_canvas()
            return

        self.current_index = max(0, min(self.current_index, len(self.image_paths) - 1))
        self.current_image_path = self.image_paths[self.current_index]
        self.counter_var.set(f"进度：{self.current_index + 1} / {len(self.image_paths)}")

        try:
            with Image.open(self.current_image_path) as img:
                self.current_image = img.convert("RGB")
        except Exception as exc:
            messagebox.showerror("打开图片失败", f"{self.current_image_path}\n\n{exc}")
            self.current_image = None
            return

        self.current_image_size = self.current_image.size
        self.boxes = self._load_label_file(self.current_image_path)
        self.selected_index = None
        self.pending_box = None
        self.pending_digits = ""
        self.input_var.set("当前输入编号：-")
        self.render_cache_key = None

        label_path = self._label_path_for(self.current_image_path)
        label_status = "未选择标签文件夹"
        if label_path is not None:
            if label_path.exists() and label_path.stat().st_size == 0:
                label_status = "已标记为空图"
            elif label_path.exists():
                label_status = "已有标签"
            else:
                label_status = "未标注"

        self.image_info_var.set(f"当前图片：{self.current_image_path.name}\n状态：{label_status}")
        self._refresh_box_listbox()
        self._render_canvas()
        self._set_status("可以开始画框了。画完后输入类别编号，再按 Enter。")
        self._save_state()
        self.canvas.focus_set()

    def _load_label_file(self, image_path: Path) -> list[YoloBox]:
        label_path = self._label_path_for(image_path)
        if label_path is None or not label_path.exists():
            return []
        boxes: list[YoloBox] = []
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parsed = YoloBox.from_line(line)
            if parsed is not None:
                boxes.append(parsed)
        return boxes

    def _save_current_labels(self) -> None:
        if self.current_image_path is None:
            return
        label_path = self._label_path_for(self.current_image_path)
        if label_path is None:
            messagebox.showwarning("缺少标签文件夹", "请先选择标签文件夹。")
            return

        label_path.parent.mkdir(parents=True, exist_ok=True)
        text = "\n".join(box.to_line() for box in self.boxes)
        label_path.write_text(text, encoding="utf-8")

        self.image_info_var.set(f"当前图片：{self.current_image_path.name}\n状态：已有标签")
        self._set_status(f"已保存：{label_path.name}")
        self._save_state()

    def _mark_empty_image(self) -> None:
        if self.current_image_path is None:
            return
        label_path = self._label_path_for(self.current_image_path)
        if label_path is None:
            messagebox.showwarning("缺少标签文件夹", "请先选择标签文件夹。")
            return

        self.boxes = []
        self.selected_index = None
        self.pending_box = None
        self.pending_digits = ""
        self.input_var.set("当前输入编号：-")

        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text("", encoding="utf-8")
        self.image_info_var.set(f"当前图片：{self.current_image_path.name}\n状态：已标记为空图")
        self._refresh_box_listbox()
        self._render_canvas()
        self._set_status("已保存为空图标签。")
        self._save_state()

    def _prev_image(self) -> None:
        if not self.image_paths:
            return
        if self.current_index > 0:
            self.current_index -= 1
            self._load_current_image()

    def _next_image(self) -> None:
        if not self.image_paths:
            return
        if self.current_index < len(self.image_paths) - 1:
            self.current_index += 1
            self._load_current_image()

    def _refresh_box_listbox(self) -> None:
        self.box_listbox.delete(0, tk.END)
        for idx, box in enumerate(self.boxes):
            class_name = self._class_name(box.class_id)
            self.box_listbox.insert(tk.END, f"{idx + 1}. [{box.class_id}] {class_name}")
        if self.selected_index is not None and self.selected_index < len(self.boxes):
            self.box_listbox.selection_set(self.selected_index)

    def _class_name(self, class_id: int) -> str:
        draft_classes = self._read_classes_from_text()
        if not self.classes and draft_classes:
            self.classes = draft_classes

        if 0 <= class_id < len(self.classes):
            return self.classes[class_id]
        return f"未定义类别 {class_id}"

    def _on_box_list_select(self, _event: object) -> None:
        selection = self.box_listbox.curselection()
        if not selection:
            return
        self.selected_index = int(selection[0])
        self.pending_box = None
        self.pending_digits = ""
        self.input_var.set("当前输入编号：-")
        self._render_canvas()
        box = self.boxes[self.selected_index]
        self._set_status(f"已选中框：[{box.class_id}] {self._class_name(box.class_id)}")
        self.canvas.focus_set()

    def _on_canvas_resize(self, _event: tk.Event) -> None:
        self._render_canvas()

    def _on_canvas_press(self, event: tk.Event) -> None:
        if self.current_image is None:
            return
        start = self._canvas_to_image(event.x, event.y, clamp=False)
        if start is None:
            return
        self.drag_start = start
        self.drag_current = start
        self.canvas.focus_set()

    def _on_canvas_drag(self, event: tk.Event) -> None:
        if self.drag_start is None:
            return
        current = self._canvas_to_image(event.x, event.y, clamp=True)
        if current is None:
            return
        self.drag_current = current
        self._render_canvas()

    def _on_canvas_release(self, event: tk.Event) -> None:
        if self.drag_start is None:
            return

        end = self._canvas_to_image(event.x, event.y, clamp=True)
        start = self.drag_start
        self.drag_start = None
        self.drag_current = None

        if end is None:
            self._render_canvas()
            return

        x1, y1 = start
        x2, y2 = end
        if abs(x2 - x1) < MIN_BOX_SIZE or abs(y2 - y1) < MIN_BOX_SIZE:
            selected = self._find_box_at_point(end[0], end[1])
            self.selected_index = selected
            self.pending_box = None
            self.pending_digits = ""
            self.input_var.set("当前输入编号：-")
            self._refresh_box_listbox()
            self._render_canvas()
            if selected is not None:
                box = self.boxes[selected]
                self.box_listbox.selection_clear(0, tk.END)
                self.box_listbox.selection_set(selected)
                self._set_status(
                    f"已选中框：[{box.class_id}] {self._class_name(box.class_id)}。"
                    "直接输入新编号并按 Enter，可改类别。"
                )
            else:
                self._set_status("没有选中框。可以重新画框。")
            return

        self.pending_box = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
        self.selected_index = None
        self.box_listbox.selection_clear(0, tk.END)
        self.pending_digits = ""
        self.input_var.set("当前输入编号：-")
        self._render_canvas()
        self._set_status("新框已创建。请输入类别编号，然后按 Enter。")

    def _handle_keypress(self, event: tk.Event) -> None:
        if self.root.focus_get() is self.class_text:
            return

        key = event.keysym
        char = event.char

        if char.isdigit() and (self.pending_box is not None or self.selected_index is not None):
            self.pending_digits += char
            self.input_var.set(f"当前输入编号：{self.pending_digits}")
            self._set_status(f"已输入编号：{self.pending_digits}。按 Enter 确认。")
            return

        if key == "BackSpace":
            if self.pending_digits:
                self.pending_digits = self.pending_digits[:-1]
                self.input_var.set(f"当前输入编号：{self.pending_digits or '-'}")
                self._set_status(
                    f"当前编号：{self.pending_digits or '空'}。继续输入，或按 Enter 确认。"
                )
                return
            if self.selected_index is not None:
                self._delete_selected_box()
                return

        if key == "Delete":
            self._delete_selected_box()
            return

        if key in {"Return", "KP_Enter"}:
            self._confirm_number_input()
            return

        if key == "Escape":
            self.pending_box = None
            self.pending_digits = ""
            self.input_var.set("当前输入编号：-")
            self._render_canvas()
            self._set_status("已取消当前输入。")
            return

        if key in {"Right", "d", "D", "Next"}:
            self._next_image()
            return

        if key in {"Left", "a", "A", "Prior"}:
            self._prev_image()
            return

        if key in {"s", "S"}:
            self._save_current_labels()
            return

        if key in {"e", "E"}:
            self._mark_empty_image()
            return

    def _confirm_number_input(self) -> None:
        self._sync_classes_from_text(silent=True)

        if not self.pending_digits:
            self._set_status("还没有输入类别编号。")
            return

        try:
            class_id = int(self.pending_digits)
        except ValueError:
            self._set_status("类别编号不合法。")
            return

        if class_id < 0:
            self._set_status("类别编号不能小于 0。")
            return

        if self.classes and class_id >= len(self.classes):
            self._set_status(f"编号超出范围。当前最大编号是 {len(self.classes) - 1}。")
            return

        if self.pending_box is not None:
            x1, y1, x2, y2 = self.pending_box
            image_width, image_height = self.current_image_size
            box = YoloBox(
                class_id=class_id,
                x_center=((x1 + x2) / 2) / image_width,
                y_center=((y1 + y2) / 2) / image_height,
                width=(x2 - x1) / image_width,
                height=(y2 - y1) / image_height,
            )
            self.boxes.append(box)
            self.selected_index = len(self.boxes) - 1
            self.pending_box = None
            self.pending_digits = ""
            self.input_var.set("当前输入编号：-")
            self._refresh_box_listbox()
            self.box_listbox.selection_set(self.selected_index)
            self._save_current_labels()
            self._render_canvas()
            self._set_status(
                f"已添加框：[{class_id}] {self._class_name(class_id)}。继续画下一个框即可。"
            )
            return

        if self.selected_index is not None:
            self.boxes[self.selected_index].class_id = class_id
            self.pending_digits = ""
            self.input_var.set("当前输入编号：-")
            self._refresh_box_listbox()
            self.box_listbox.selection_set(self.selected_index)
            self._save_current_labels()
            self._render_canvas()
            self._set_status(f"已修改类别为：[{class_id}] {self._class_name(class_id)}。")
            return

        self._set_status("请先画框，或先选中一个已有框。")

    def _delete_selected_box(self) -> None:
        if self.selected_index is None or self.selected_index >= len(self.boxes):
            self._set_status("没有选中的框可删除。")
            return
        removed = self.boxes.pop(self.selected_index)
        self.selected_index = None
        self.pending_digits = ""
        self.input_var.set("当前输入编号：-")
        self._refresh_box_listbox()
        self._save_current_labels()
        self._render_canvas()
        self._set_status(f"已删除框：[{removed.class_id}] {self._class_name(removed.class_id)}")

    def _find_box_at_point(self, image_x: float, image_y: float) -> int | None:
        image_width, image_height = self.current_image_size
        for idx in range(len(self.boxes) - 1, -1, -1):
            x1, y1, x2, y2 = self.boxes[idx].to_pixel_box(image_width, image_height)
            if x1 <= image_x <= x2 and y1 <= image_y <= y2:
                return idx
        return None

    def _canvas_to_image(self, canvas_x: int, canvas_y: int, clamp: bool) -> tuple[float, float] | None:
        x0, y0, display_w, display_h = self.canvas_image_rect
        if display_w <= 1 or display_h <= 1 or self.current_image is None:
            return None

        rel_x = canvas_x - x0
        rel_y = canvas_y - y0
        if clamp:
            rel_x = max(0, min(display_w, rel_x))
            rel_y = max(0, min(display_h, rel_y))
        elif rel_x < 0 or rel_y < 0 or rel_x > display_w or rel_y > display_h:
            return None

        image_width, image_height = self.current_image_size
        return (
            rel_x / display_w * image_width,
            rel_y / display_h * image_height,
        )

    def _image_to_canvas(self, image_x: float, image_y: float) -> tuple[float, float]:
        x0, y0, display_w, display_h = self.canvas_image_rect
        image_width, image_height = self.current_image_size
        return (
            x0 + (image_x / image_width) * display_w,
            y0 + (image_y / image_height) * display_h,
        )

    def _render_canvas(self) -> None:
        self.canvas.delete("all")

        if self.current_image is None:
            self.canvas.create_text(
                self.canvas.winfo_width() // 2,
                self.canvas.winfo_height() // 2,
                text="请选择图片文件夹开始标注",
                fill="#ffffff",
                font=("Microsoft YaHei UI", 18),
            )
            return

        canvas_w = max(self.canvas.winfo_width(), 50)
        canvas_h = max(self.canvas.winfo_height(), 50)
        image_w, image_h = self.current_image_size
        scale = min((canvas_w - 20) / image_w, (canvas_h - 20) / image_h)
        display_w = max(1, int(image_w * scale))
        display_h = max(1, int(image_h * scale))
        offset_x = max(0, (canvas_w - display_w) // 2)
        offset_y = max(0, (canvas_h - display_h) // 2)
        self.canvas_image_rect = (offset_x, offset_y, display_w, display_h)

        cache_key = (str(self.current_image_path), display_w, display_h)
        if cache_key != self.render_cache_key:
            resized = self.current_image.resize((display_w, display_h), Image.LANCZOS)
            self.tk_image = ImageTk.PhotoImage(resized)
            self.render_cache_key = cache_key

        self.canvas.create_image(offset_x, offset_y, image=self.tk_image, anchor=tk.NW)

        for idx, box in enumerate(self.boxes):
            x1, y1, x2, y2 = box.to_pixel_box(image_w, image_h)
            cx1, cy1 = self._image_to_canvas(x1, y1)
            cx2, cy2 = self._image_to_canvas(x2, y2)
            color = BOX_COLORS[box.class_id % len(BOX_COLORS)]
            width = 3 if idx == self.selected_index else 2
            self.canvas.create_rectangle(cx1, cy1, cx2, cy2, outline=color, width=width)
            label = f"{box.class_id}: {self._class_name(box.class_id)}"
            label_w = max(96, len(label) * 10)
            self.canvas.create_rectangle(cx1, cy1 - 24, cx1 + label_w, cy1, fill=color, outline="")
            self.canvas.create_text(
                cx1 + 6,
                cy1 - 12,
                text=label,
                fill="#ffffff",
                anchor=tk.W,
                font=("Microsoft YaHei UI", 10, "bold"),
            )

        preview_box = self.pending_box
        if self.drag_start is not None and self.drag_current is not None:
            x1, y1 = self.drag_start
            x2, y2 = self.drag_current
            preview_box = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))

        if preview_box is not None:
            x1, y1, x2, y2 = preview_box
            cx1, cy1 = self._image_to_canvas(x1, y1)
            cx2, cy2 = self._image_to_canvas(x2, y2)
            self.canvas.create_rectangle(
                cx1,
                cy1,
                cx2,
                cy2,
                outline="#ffd666",
                width=2,
                dash=(6, 4),
            )

    def _parse_classes_from_yaml(self, yaml_path: Path) -> list[str]:
        lines = yaml_path.read_text(encoding="utf-8").splitlines()
        names_started = False
        names_map: dict[int, str] = {}
        names_list: list[str] = []

        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            if stripped == "names:":
                names_started = True
                continue

            if names_started:
                if ":" in stripped:
                    left, right = stripped.split(":", 1)
                    left = left.strip().strip("\"'")
                    right = right.strip().strip("\"'")
                    if left.isdigit():
                        names_map[int(left)] = right
                        continue
                if stripped.startswith("- "):
                    names_list.append(stripped[2:].strip().strip("\"'"))
                    continue
                if not line.startswith((" ", "\t")):
                    break

        if names_map:
            max_index = max(names_map)
            return [names_map.get(idx, f"class_{idx}") for idx in range(max_index + 1)]
        return names_list


def main() -> None:
    root = tk.Tk()
    app = SimpleLabelTool(root)
    app._set_status("工具已启动。先在左侧选择图片文件夹和标签文件夹。")
    root.mainloop()


if __name__ == "__main__":
    main()
