# FengShui Dataset Annotation Guide

## Goal

This dataset is better suited for object detection than single-label classification.
The training target should be:

- One image can contain multiple labels.
- Every visible target class in the image should be annotated.
- Uncertain samples should go to review, not directly into training.

The updated `auto_label.py` already follows this idea:

- multi-class pre-labeling
- expected-class fallback
- duplicate-aware splitting
- review queue export

Recommended first command:

```bash
pip install torch torchvision transformers pillow tqdm
python auto_label.py --clean
```

## Core Rules

1. Do not annotate only the folder's main class.
2. Annotate all visible target classes in the same image.
3. Use one stable rule per class across the whole dataset.
4. If a class boundary is unclear, move the image to review.
5. It is better to drop a noisy image than keep a wrong label.

## Box Rules

1. Boxes should tightly cover the object itself, not large background areas.
2. If the object is heavily occluded and cannot be judged reliably, do not force a box.
3. If multiple instances are clearly separable, annotate each instance.
4. Reflections are not new objects. For `mirror`, box the mirror itself, not reflected items.
5. Extremely tiny, blurred, or decorative pseudo-objects should be skipped consistently.

## Class Definitions

### `main_door`

- Positive: front entry door, main entrance door, apartment entrance door.
- Negative: bedroom door, bathroom door, cabinet door.
- Box rule: annotate the visible door body/frame as one object.

### `room_door`

- Positive: interior room door, sliding room door, bedroom door.
- Negative: building main entry door, cabinet door.
- Box rule: annotate the actual room door, not the surrounding wall opening only.

### `floor_window`

- Positive: floor-to-ceiling window, full-height glass wall.
- Negative: normal-height window.
- Box rule: annotate the full visible window structure.

### `normal_window`

- Positive: standard windows, casement windows, small windows.
- Negative: floor-to-ceiling window.
- Box rule: annotate the whole visible window, not only the glass pane.

### `stove`

- Positive: gas stove, induction stove, cooktop, cooking range.
- Negative: microwave, oven-only front panel.
- Box rule: annotate the visible cooking unit.

### `sink`

- Positive: kitchen sink, bathroom sink, wash basin.
- Negative: countertop without a sink.
- Box rule: include the sink basin and fixed sink body area, not the whole counter.

### `stairs`

- Positive: staircase, stairway, indoor steps.
- Negative: ladder, decorative step platform without staircase function.
- Box rule: annotate the visible staircase structure.

### `water_feature`

- Positive: aquarium, fish tank, indoor fountain, indoor water feature.
- Negative: plain glass cabinet, TV wall, reflective decor.
- Box rule: annotate the water-feature object itself.

### `broad_leaf_live`

- Positive: monstera, fiddle-leaf fig, broad-leaf living plants.
- Negative: fake plants, cactus, snake plant.
- Box rule: annotate the plant body, usually including leaves and pot if inseparable.

### `sharp_leaf_live`

- Positive: cactus, aloe, snake plant, visibly spiky live plants.
- Negative: fake plants, broad-leaf plants.
- Box rule: annotate the plant body consistently.

### `fake_plant`

- Positive: artificial plant, faux plant, dried decorative plant.
- Negative: living plants.
- Box rule: annotate the decorative plant object.

### `bed`

- Positive: bed, mattress-on-frame, complete sleeping bed.
- Negative: sofa bed when used visually as sofa.
- Box rule: annotate the bed body as one object.

### `sofa`

- Positive: sofa, couch, sectional sofa, loveseat.
- Negative: bench, bed.
- Box rule: annotate the full sofa body.

### `desk`

- Positive: office desk, study desk, work desk.
- Negative: dining table, coffee table.
- Box rule: annotate the desktop furniture body.

### `dining_table`

- Positive: dining room table, kitchen dining table.
- Negative: coffee table, office desk.
- Box rule: annotate the full table body.

### `coffee_table`

- Positive: coffee table, center table, low table in living room.
- Negative: dining table, TV stand.
- Box rule: annotate the table body.

### `mirror`

- Positive: wall mirror, floor mirror, vanity mirror.
- Negative: reflective TV, glossy wall panel, reflected objects.
- Box rule: annotate the mirror object only.

### `beam`

- Positive: real ceiling beam, exposed structural beam.
- Negative: ceiling trim, shadow lines, decorative painted stripes.
- Box rule: annotate the visible beam structure.

### `toilet`

- Positive: toilet bowl, wall-hung toilet, commode.
- Negative: sink, bidet if clearly separate and not a toilet.
- Box rule: annotate the toilet fixture body.

### `cabinet`

- Positive: storage cabinet, wardrobe, sideboard, bookshelf cabinet.
- Negative: plain shelf without cabinet body, door-like wall panels.
- Box rule: annotate the cabinet furniture body.

## Review Checklist

Send the sample to review if any of the following is true:

- the expected class is missing
- there is no detection at all
- the image is a duplicate under a different class
- the class boundary is ambiguous
- the object is too small or too occluded to judge reliably
- the scene is clearly irrelevant to the dataset

The script exports these to:

- `FengShui_Dataset/review/images/`
- `FengShui_Dataset/review/labels_prelabel/`
- `FengShui_Dataset/review/needs_review.txt`

## Training-Safe Workflow

1. Run `python auto_label.py --clean`.
2. Review everything under `FengShui_Dataset/review/`.
3. Fix reviewed labels in your annotation tool.
4. Move corrected files into the main split only after review is complete.
5. Train on the clean exported dataset, not on the raw crawl folders.

## Practical Notes

- If a class keeps getting confused, merge it temporarily and train a simpler baseline first.
- Keep `train`, `val`, and `test` split fixed once review starts.
- Do not mix reviewed and unreviewed labels without tracking them.
- For the first model, clean labels matter more than maximum image count.
