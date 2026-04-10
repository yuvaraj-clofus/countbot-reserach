# Run Guide

This file explains:

- which scripts are used when you run the pipeline
- which variables in `main.py` control each method
- what each variable does
- how Stage 1 and Stage 2 connect

## Main Files

These files are part of the runtime:

- `steps/main.py`
  Entry point. You run this file.
- `steps/stage1.py`
  Stage 1 detection/segmentation methods.
- `steps/stage2.py`
  Stage 2 feature matching methods.
- `steps/fastsam_backend.py`
  FastSAM TensorRT backend. Used only when Stage 1 method is `fastsam`.

These files are not the main runtime:

- `steps/plan.md`
  Design notes.
- `steps/check_dino_fps.py`
  DINOv2 speed test only.

## Simple Flow

When you run `main.py`, the flow is:

1. Open video file or webcam.
2. Start a capture thread.
3. Read frames continuously.
4. Send each frame to Stage 1.
5. Stage 1 detects object area.
6. Stage 1 checks whether object center crossed the vertical trigger line.
7. Only if the line is crossed, `main.py` calls Stage 2.
8. Stage 2 compares the Stage 1 object crop with the reference `.pt`.
9. Show result on screen.

Pipeline:

```text
main.py
  -> stage1.py
  -> if crossed line
       -> stage2.py
```

## Run Command

From project root:

```bash
source /home/vikbot/Documents/countbot/.venv-jetson/bin/activate
python /home/vikbot/Documents/countbot/steps/main.py
```

You can also override values from command line:

```bash
python /home/vikbot/Documents/countbot/steps/main.py \
  --source /home/vikbot/Documents/countbot/steps/test.mp4 \
  --refs /home/vikbot/Documents/countbot/steps/master-models/2wheel.pt \
  --stage1-method mog2 \
  --stage2-method dinov2 \
  --device cuda
```

## What `main.py` Controls

`main.py` is the control file.

You usually change behavior here first:

- source path
- reference `.pt` path
- Stage 1 method
- Stage 2 method
- display size
- MOG2 tuning
- FastSAM tuning

## Main Defaults

Current important defaults in `main.py`:

- `DEFAULT_SOURCE`
  Default video path.
- `DEFAULT_REFS`
  Default Stage 2 reference `.pt` file.
- `STAGE1_METHOD`
  Which Stage 1 method runs.
- `STAGE2_METHOD`
  Which Stage 2 method runs.
- `DISPLAY_MAX_WIDTH`
  Max shown window width.
- `DISPLAY_MAX_HEIGHT`
  Max shown window height.

## Stage 1 Method Selection

Change this in `main.py`:

```python
STAGE1_METHOD = "fastsam"
```

Possible values:

- `"mog2"`
  Use OpenCV background subtraction.
- `"fastsam"`
  Use FastSAM TensorRT backend.
- `"color"`
  Use simple HSV color threshold.
- `"nanosam"`
  Placeholder method in current code.

Important:

- Only the active Stage 1 method uses its settings.
- If `STAGE1_METHOD = "mog2"`, FastSAM variables do nothing.
- If `STAGE1_METHOD = "fastsam"`, MOG2 variables do nothing.

## Stage 2 Method Selection

Change this in `main.py`:

```python
STAGE2_METHOD = "dinov2"
```

Possible values:

- `"dinov2"`
  Real DINOv2 embedding compare with reference `.pt`.
- `"dino"`
  Simple placeholder score using mask mean.

Important:

- In normal use, use `"dinov2"`.
- `"dino"` is useful only for light testing.

## `main.py` General Variables

These variables affect all runs:

- `DEFAULT_SOURCE`
  Input source.
  Example:
  - video file: `steps/test.mp4`
  - webcam: use CLI `--source 0`

- `DEFAULT_REFS`
  Default reference embedding file used by Stage 2.
  This is the `.pt` file your object is compared against.

- `DISPLAY_MAX_WIDTH`
  Maximum width of shown window after resize.
  Only affects display size, not detection accuracy.

- `DISPLAY_MAX_HEIGHT`
  Maximum height of shown window after resize.
  Only affects display size, not detection accuracy.

## MOG2 Variables In `main.py`

These are used only when:

```python
STAGE1_METHOD = "mog2"
```

- `MOG2_BINARY_THRESHOLD`
  After background subtraction, the mask is binarized.
  Lower value:
  - more motion becomes white
  - more sensitive
  - may add noise
  Higher value:
  - less noise
  - may miss parts of object

- `MOG2_MIN_AREA`
  Minimum contour area to keep as an object.
  Lower value:
  - keeps smaller objects
  - may detect noise
  Higher value:
  - removes small noise
  - may miss small objects

- `MOG2_MIN_WIDTH`
  Minimum bounding box width.
  Small detections under this width are ignored.

- `MOG2_MIN_HEIGHT`
  Minimum bounding box height.
  Small detections under this height are ignored.

- `MOG2_BBOX_PADDING`
  Extra pixels added around detected box.
  Increase this if box is too tight and does not cover full object.

- `MOG2_INSIDE_MARGIN`
  Used when removing a child box inside a bigger parent box.
  Bigger value makes inside-box removal more aggressive.

- `MOG2_MERGE_GAP`
  Used to merge nearby boxes into one larger box.
  Increase this if one object is split into multiple nearby boxes.

- `MOG2_LINE_FRACTION`
  Position of vertical trigger line.
  `0.5` means line is at center of frame width.
  Example:
  - `0.25` = left side trigger
  - `0.50` = middle trigger
  - `0.75` = right side trigger

- `MOG2_CENTER_MATCH_DISTANCE`
  Used to decide whether an object in current frame is the same object as previous frame.
  This helps line-cross trigger tracking.
  Increase if objects move fast between frames.

## FastSAM Variables In `main.py`

These are used only when:

```python
STAGE1_METHOD = "fastsam"
```

- `FASTSAM_MODEL_PATH`
  Path to FastSAM TensorRT engine file.
  This should point to a valid `.engine` file.
  Example value:
  `/home/vikbot/Documents/countbot/steps/master-models/weights/FastSAM-s.engine`

- `FASTSAM_CONF_THRESHOLD`
  Minimum confidence for FastSAM detections.
  Lower value:
  - more detections
  - more false positives
  Higher value:
  - fewer detections
  - more strict

- `FASTSAM_NMS_IOU`
  Non-max suppression IoU threshold.
  Controls how overlapping FastSAM detections are filtered.

- `FASTSAM_IMGSZ`
  Inference image size for FastSAM.
  Larger value:
  - may improve detection quality
  - slower
  Smaller value:
  - faster
  - may reduce detail

- `FASTSAM_MOTION_DIFF_THRESH`
  Motion-difference threshold for the motion mask used with FastSAM.
  Lower value:
  - more sensitive to motion
  Higher value:
  - less sensitive

- `FASTSAM_MIN_BOX_AREA`
  Minimum allowed FastSAM box area.
  Helps remove tiny detections.

- `FASTSAM_MAX_BOX_AREA_RATIO`
  Maximum box area relative to whole frame.
  Prevents a huge box from covering most of frame.

- `FASTSAM_MAX_BOX_WIDTH_RATIO`
  Maximum allowed box width relative to frame width.

- `FASTSAM_MAX_BOX_HEIGHT_RATIO`
  Maximum allowed box height relative to frame height.

- `FASTSAM_DARK_ROI_THRESH`
  Helps reject very dark object regions.
  Useful if dark noise causes false detections.

- `FASTSAM_MOTION_OVERLAP_THRESH`
  Minimum overlap between FastSAM box and motion region.
  Increase this to keep only boxes that agree more with motion.

- `FASTSAM_MERGE_BOX_IOU_THRESH`
  Merge overlapping boxes when IoU is above this value.

- `FASTSAM_MERGE_BOX_GAP`
  Merge boxes that are very close.

- `FASTSAM_MERGE_AXIS_OVERLAP`
  Axis overlap rule used during box merging.

- `STAGE1_LINE_FRACTION`
  Vertical trigger line position for FastSAM path.
  Same idea as `MOG2_LINE_FRACTION`.

- `STAGE1_CENTER_MATCH_DISTANCE`
  Object tracking distance for trigger logic.
  Same idea as `MOG2_CENTER_MATCH_DISTANCE`.

## How Line Trigger Works

Stage 1 always detects boxes first.

But Stage 2 does not run for every box on every frame.

Current logic:

1. Stage 1 finds current objects.
2. Stage 1 compares current objects with previous-frame objects.
3. If object center crosses the vertical line, that object becomes `triggered`.
4. `main.py` sends only that triggered object to Stage 2.

If there is a box but no crossing yet:

- Stage 2 is not called
- screen shows `WAIT_LINE`

If there is no box:

- Stage 2 is not called
- screen shows `NO_OBJECT`

## Stage 2 Variables

Stage 2 is configured by `main.py` through CLI args and defaults.

Important values:

- `--refs`
  Reference `.pt` file to compare against.

- `--device`
  Usually `cuda` or `cpu`.
  Use `cuda` for GPU if available.

- `--model-name`
  DINOv2 model name.
  Current default:
  `dinov2_vits14`

- `--threshold`
  Final match threshold.
  If similarity score is greater than or equal to this threshold:
  - result = `MATCH`
  Otherwise:
  - result = `NOT_MATCH`

Important note:

- `stage2.py` has its own internal defaults
- but when you run `main.py`, `main.py` calls `stage2.configure(...)`
- so the actual runtime values come from `main.py` and command line

## Runtime Globals In `main.py`

These are internal runtime variables:

- `GLOBAL_FRAME`
  Latest frame read by capture thread.

- `GLOBAL_FRAME_ID`
  Current frame number used internally.

- `GLOBAL_RESULT`
  Latest Stage 2 result shown on screen.

- `GLOBAL_RUNNING`
  Main stop/start flag.

- `CAPTURE_FINISHED`
  Tells the loop that video file reached end.

- `FRAME_LOCK`
  Lock for safe frame sharing between thread and main loop.

- `FRAME_QUEUE`
  Queue of frames waiting to be processed.
  Important:
  this is what helps full video processing stay in order.

## Which File To Edit For Common Changes

If you want to change method:

- edit `STAGE1_METHOD` in `main.py`
- edit `STAGE2_METHOD` in `main.py`

If you want to change input video:

- edit `DEFAULT_SOURCE` in `main.py`
- or pass `--source`

If you want to change reference model:

- edit `DEFAULT_REFS` in `main.py`
- or pass `--refs`

If you want to change MOG2 behavior:

- edit MOG2 variables in `main.py`

If you want to change FastSAM behavior:

- edit FastSAM variables in `main.py`

If you want to change DINOv2 match strictness:

- edit `--threshold` default in `main.py`

If you want to change Stage 1 method logic itself:

- edit `stage1.py`

If you want to change Stage 2 comparison logic itself:

- edit `stage2.py`

## Good Example Setups

### 1. Simple MOG2 + DINOv2

In `main.py`:

```python
STAGE1_METHOD = "mog2"
STAGE2_METHOD = "dinov2"
DEFAULT_REFS = str(SCRIPT_DIR / "master-models" / "2wheel.pt")
```

Use this when:

- object is moving clearly
- background subtraction works well
- you want a simpler pipeline

### 2. FastSAM + DINOv2

In `main.py`:

```python
STAGE1_METHOD = "fastsam"
STAGE2_METHOD = "dinov2"
```

Also set:

```python
FASTSAM_MODEL_PATH = "/home/vikbot/Documents/countbot/steps/master-models/weights/FastSAM-s.engine"
```

Use this when:

- MOG2 boxes are weak
- you want stronger object segmentation
- TensorRT engine is available

### 3. Light Test Mode

In `main.py`:

```python
STAGE1_METHOD = "mog2"
STAGE2_METHOD = "dino"
```

Use this when:

- you want faster testing
- you do not need real DINOv2 compare yet

## Output Meaning

Possible screen labels:

- `MATCH`
  Stage 2 similarity score passed threshold.

- `NOT_MATCH`
  Stage 2 similarity score did not pass threshold.

- `WAIT_LINE`
  Object detected, but object center has not crossed the trigger line yet.

- `NO_OBJECT`
  No usable object was detected in current frame.

## Important Reminder

For normal use, this is the most important place to edit:

- `steps/main.py`

That file decides:

- which Stage 1 method runs
- which Stage 2 method runs
- which video is used
- which `.pt` file is used
- what thresholds are used
