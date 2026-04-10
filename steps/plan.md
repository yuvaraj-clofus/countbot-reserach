# Simple Selector Pipeline Plan

## Goal

Build the project again from scratch with a very simple structure.

Rules:

- only functions
- no classes
- `main.py` stays very small
- `main.py` decides which method is active
- `stage1.py` can contain many methods, but now use only `mog2`
- `stage2.py` can contain many methods, but now use only `dinov2`

## Files

Use only these files:

- `steps/main.py`
- `steps/stage1.py`
- `steps/stage2.py`

## Current Active Methods

Set the active methods in `main.py`:

```python
STAGE1_METHOD = "mog2"
STAGE2_METHOD = "dinov2"
```

This means:

- `stage1.py` may have `color`, `mog2`, `nanosam`
- `stage2.py` may have `dino`, `dinov2`
- but right now the pipeline uses only:
  - stage 1: `mog2`
  - stage 2: `dinov2`

## Shared Global Variables

Keep the shared data very small.

```python
GLOBAL_FRAME = None
GLOBAL_RESULT = None
GLOBAL_RUNNING = True
```

Meaning:

- `GLOBAL_FRAME`
  latest frame from video or webcam
- `GLOBAL_RESULT`
  latest output from stage 2
- `GLOBAL_RUNNING`
  stop flag for the loops

## High-Level Flow

```text
video / webcam
   ->
reader thread in main.py
   ->
GLOBAL_FRAME
   ->
stage1.run(frame, method="mog2")
   ->
stage2.run(stage1_output, method="dinov2")
   ->
GLOBAL_RESULT
   ->
draw result on frame
```

## main.py

### Responsibility

- open webcam or video
- start a thread that keeps updating `GLOBAL_FRAME`
- choose which stage 1 method to use
- choose which stage 2 method to use
- call `stage1.run()`
- call `stage2.run()`
- show output on screen

### main.py should define method selection

`main.py` must clearly show which methods are used now.

Example:

```python
STAGE1_METHOD = "mog2"
STAGE2_METHOD = "dinov2"
```

### Simple target flow

```python
GLOBAL_FRAME = None
GLOBAL_RESULT = None
GLOBAL_RUNNING = True

STAGE1_METHOD = "mog2"
STAGE2_METHOD = "dinov2"

def capture_loop(source):
    while GLOBAL_RUNNING:
        frame = read_frame(source)
        GLOBAL_FRAME = frame

def process_loop():
    bg_subtractor = cv2.createBackgroundSubtractorMOG2()

    while GLOBAL_RUNNING:
        if GLOBAL_FRAME is None:
            continue

        frame = GLOBAL_FRAME.copy()
        stage1_output = stage1.run(frame, method=STAGE1_METHOD, bg_subtractor=bg_subtractor)
        stage2_output = stage2.run(stage1_output, method=STAGE2_METHOD)
        GLOBAL_RESULT = stage2_output

        draw_result(frame, GLOBAL_RESULT)
        show_frame(frame)
```

### Keep main.py simple

- no segmentation code in `main.py`
- no DINO logic in `main.py`
- no model logic in `main.py`
- only:
  - read frame
  - choose method
  - call stage 1
  - call stage 2
  - display result

## stage1.py

### Purpose

`stage1.py` is the segmentation selector.

It can contain multiple segmentation methods.

For now, use only `mog2`.

### Methods planned in stage1.py

- `color_threshold(img)`
- `mog2(img, bg_subtractor)`
- `nanosam(img)`

### Handler map

```python
handlers = {
    "color": color_threshold,
    "mog2": mog2,
    "nanosam": nanosam,
}
```

### Simple selection

There are 2 possible styles:

1. Manual selection from `main.py`
2. Auto selection inside `stage1.py`

For now, prefer manual selection from `main.py`.

### Stage 1 run shape

```python
def run(img, method="mog2", bg_subtractor=None):
    if method == "mog2":
        return handlers[method](img, bg_subtractor)
    return handlers[method](img)
```

### Current real use

Right now:

- `stage1.run(..., method="mog2")`

### Expected output from stage1

Keep it simple.

Example:

```python
{
    "type": "mask",
    "method": "mog2",
    "data": mask
}
```

Optional later:

- bbox
- center point
- contour list

## stage2.py

### Purpose

`stage2.py` is the feature-matching selector.

It can contain multiple feature methods.

For now, use only `dinov2`.

### Methods planned in stage2.py

- `dino(stage1_output)`
- `dinov2(stage1_output)`

### Handler map

```python
handlers = {
    "dino": dino,
    "dinov2": dinov2,
}
```

### Stage 2 run shape

```python
def run(stage1_output, method="dinov2"):
    return handlers[method](stage1_output)
```

### Current real use

Right now:

- `stage2.run(stage1_output, method="dinov2")`

### Expected output from stage2

Keep it simple.

Example:

```python
{
    "type": "result",
    "method": "dinov2",
    "label": "MATCH",
    "score": 0.84
}
```

or

```python
{
    "type": "result",
    "method": "dinov2",
    "label": "NOT_MATCH",
    "score": 0.41
}
```

## Input and Output

### Input

- webcam
- or video file such as `steps/test.mp4`

### Output

- live display window
- current processed frame
- current result from stage 2

Optional later:

- save output video
- save mask image
- save logs

## Clean Architecture Summary

### main.py

- controls source
- controls thread
- controls global frame
- controls which methods are active
- calls stage 1 and stage 2

### stage1.py

- holds segmentation methods
- currently uses `mog2`

### stage2.py

- holds feature methods
- currently uses `dinov2`

## Final Target

The final simple behavior should be:

1. `main.py` opens webcam or video
2. a thread updates `GLOBAL_FRAME`
3. `main.py` sets:
   - `STAGE1_METHOD = "mog2"`
   - `STAGE2_METHOD = "dinov2"`
4. `stage1.py` runs the selected stage 1 method
5. `stage2.py` runs the selected stage 2 method
6. result is shown on the current frame

## Later Extension

Later you can add new methods without changing the pipeline shape.

Example:

- stage 1:
  - `color`
  - `mog2`
  - `nanosam`
- stage 2:
  - `dino`
  - `dinov2`

Only `main.py` changes the active method names.
