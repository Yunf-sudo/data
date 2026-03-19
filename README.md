# 风水目标检测数据集采集工具（纯爬虫版）

无需任何 API Key，使用 Bing / Houzz / Flickr / Dezeen 四个来源。

## 文件说明

```
fengshui_scraper/
├── config.py       ← 参数配置 + 20类关键词
├── crawlers.py     ← 四个爬虫（Bing/Houzz/Flickr/Dezeen）
├── downloader.py   ← 下载器 + 图片质量过滤
├── main.py         ← 主采集脚本（先运行）
└── auto_label.py   ← Grounding DINO 自动预标注（后运行）
```

---

## 使用步骤

### 第一步：安装依赖

```bash
pip install requests pillow
```

### 第二步：开始采集

```bash
python main.py
```

预计耗时：8～16 小时（20类 × 1000张，视网速而定）

支持**断点续传**，中断后直接重跑，已下载的自动跳过。

输出结构：
```
FengShui_Dataset/
├── main_door/         ← ~1000张入户大门场景图
├── room_door/         ← ~1000张室内门场景图
├── floor_window/
├── normal_window/
├── stove/
├── sink/
├── stairs/
├── water_feature/
├── broad_leaf_live/
├── sharp_leaf_live/
├── fake_plant/
├── bed/
├── sofa/
├── desk/
├── dining_table/
├── coffee_table/
├── mirror/
├── beam/
├── toilet/
└── cabinet/
```

### 第三步：自动预标注

```bash
pip install torch torchvision transformers
python auto_label.py
```

会生成：
- `FengShui_Dataset/images/train/` + `labels/train/`
- `FengShui_Dataset/images/val/`   + `labels/val/`
- `FengShui_Dataset/data.yaml`（RT-DETR 直接使用）
- `FengShui_Dataset/needs_review.txt`（空标注，需人工复核）

### 第四步：人工复核空标注

```bash
pip install labelImg
labelImg FengShui_Dataset/images/train FengShui_Dataset/labels/train
```

只需处理 `needs_review.txt` 中列出的图片。

---

## 20个类别

| idx | 标签 | 中文 | 风水意义 |
|-----|------|------|----------|
| 0 | main_door | 入户大门 | 气口，权重最高 |
| 1 | room_door | 室内门 | 门冲判定 |
| 2 | floor_window | 落地窗 | 割脚煞 / 背后无靠 |
| 3 | normal_window | 普通窗 | 采光 / 靠背 |
| 4 | stove | 炉灶（火） | 水火相冲判定 |
| 5 | sink | 水槽（水） | 水火相冲判定 |
| 6 | stairs | 楼梯 | 开门见梯 / 漏财 |
| 7 | water_feature | 水景/鱼缸 | 催财核心 |
| 8 | broad_leaf_live | 阔叶真植物 | 聚财生旺 |
| 9 | sharp_leaf_live | 尖叶真植物 | 化煞 |
| 10 | fake_plant | 假植物/枯萎 | 阴气 |
| 11 | bed | 床 | 横梁压顶 / 门冲主判 |
| 12 | sofa | 沙发 | 背后无靠 |
| 13 | desk | 书桌 | 事业/学业格局 |
| 14 | dining_table | 餐桌 | 家庭和睦 |
| 15 | coffee_table | 茶几 | 辅助判定 |
| 16 | mirror | 镜子 | 镜照床/门 |
| 17 | beam | 横梁 | 压顶煞，高权重 |
| 18 | toilet | 马桶 | 厨厕相冲 |
| 19 | cabinet | 柜子 | 动线遮挡 / 玄关 |

---

## 注意

- `beam`（横梁）和 `water_feature`（鱼缸）搜索结果天然较少，最终可能只有 400～700 张，用数据增强补偿
- Bing 每次请求之间有 1～2 秒延迟，Houzz 有 2～3.5 秒，程序内已内置，不要手动调快
- 如遇到 Bing 返回空结果，通常是临时限速，等 10 分钟后重跑即可（断点续传不会重复下载）
