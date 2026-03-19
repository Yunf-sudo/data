from __future__ import annotations

GROUPS: dict[str, list[str]] = {
    "plant": ["broad_leaf_live", "fake_plant", "sharp_leaf_live"],
    "table": ["coffee_table", "dining_table"],
    "door": ["main_door", "room_door"],
    "window": ["floor_window", "normal_window"],
}

DEFAULT_MASKED_CLS_OUTPUT = "FengShui_GroupCls"
DEFAULT_CLASSIFIER_RUNS = "runs/group_cls"
