# Windows 简易标注工具

这个目录就是完整工具包，给 Windows 标注员直接使用。

## 目录说明

- `run_label_tool.bat`：双击启动
- `simple_label_tool.py`：主程序
- `requirements-label-tool.txt`：依赖列表
- `data.yaml`：默认 20 类模板
- `README.md`：使用说明

## 适合什么场景

- 标注员没有技术背景
- 想要中文界面
- 想要先预设类别
- 想要画框后直接输入数字编号
- 导出格式要能直接给 RT-DETR / YOLO 训练

## 默认内置类别

工具启动后会自动带上这 20 类，不需要先准备 `yaml`：

```text
beam
bed
broad_leaf_live
cabinet
coffee_table
desk
dining_table
fake_plant
floor_window
main_door
mirror
normal_window
room_door
sharp_leaf_live
sink
sofa
stairs
stove
toilet
water_feature
```

## 启动方法

直接双击：

`run_label_tool.bat`

第一次启动会自动完成下面几件事：

- 创建本地运行环境
- 安装依赖 `Pillow`
- 打开标注界面

注意：电脑里需要先安装 Python 3。

## 最简单使用流程

1. 点击 `选择图片文件夹`
2. 点击 `选择标签文件夹`
3. 在左侧输入类别，每行一个
4. 点击 `应用类别`
5. 鼠标左键拖动，框住目标
6. 输入类别编号
7. 按 `Enter` 完成保存

类别编号从 `0` 开始。

例如类别写成：

```text
door
window
plant
table
```

对应关系就是：

- `0 = door`
- `1 = window`
- `2 = plant`
- `3 = table`

## 快捷键

- `左键拖动`：画框
- `输入数字`：输入类别编号
- `Enter`：确认当前编号
- `Delete`：删除选中框
- `左方向键`：上一张
- `右方向键`：下一张
- `S`：手动保存
- `E`：当前图片标记为无目标

## 输出格式

每张图片会生成一个同名 `txt` 文件，格式是：

```text
class_id x_center y_center width height
```

这是标准 YOLO 标注格式，RT-DETR 可直接使用。

示例：

```text
2 0.532000 0.487000 0.214000 0.356000
```

## 推荐数据目录

```text
dataset/
  images/
    train/
    val/
  labels/
    train/
    val/
```

如果你选择的是 `images/train`，工具会自动猜测标签目录为 `labels/train`。

## 可直接读取 data.yaml

如果你已经有 RT-DETR / YOLO 的 `data.yaml`，可以点击：

`从 data.yaml 读取类别`

这样就不用手动重新录入类别。

如果你现在还没有 `data.yaml`，可以直接使用这个目录里的：

`windows_label_tool/data.yaml`
