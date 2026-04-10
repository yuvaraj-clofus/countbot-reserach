# MOG2 Conveyor Detection

This folder contains a simple conveyor object detector based on OpenCV MOG2 background subtraction.

Main file:
- [detect.py](/home/vikbot/Documents/countbot/testing/MOG2/detect.py)

## What This Method Is

This is not AI class detection like YOLO or FastSAM.

It works by:
- learning the conveyor background
- finding moving/changed regions in each frame
- cleaning the foreground mask
- turning those regions into bounding boxes
- filtering weak blobs such as dust, shadow, and lighting flicker

So this method is:
- good for fast moving-object detection
- good for simple conveyor motion boxes
- not perfect for separating real product from shadow/light changes

## How It Works

The pipeline in `detect.py` is:

1. Read video or webcam.
2. Build a background model using OpenCV MOG2.
3. Compare each frame with the learned background.
4. Threshold the foreground mask into black/white.
5. Clean the mask with blur and morphology.
6. Find contours in the white foreground regions.
7. Convert contours into bounding boxes.
8. Filter bad boxes using size, shape, fill, and background difference.
9. Merge nearby boxes into one product box.
10. Keep only stable detections that persist for multiple frames.
11. Draw boxes, score, FPS, ROI, and detection count.

## What "Threshold" Means Here

This script uses several threshold ideas:

- `--var-threshold`
  MOG2 sensitivity to change.
- binary mask threshold inside the code
  Converts MOG2 output into a clean foreground mask.
- detection filters
  These decide whether a blob is strong enough to be shown.

Important detection filters:
- `--min-area`
- `--min-fill-ratio`
- `--min-solidity`
- `--min-diff`
- `--min-score`
- `--min-persist`

## Detection Score

Each box shows text like:

```text
object 0.82
```

This is a simple quality score from `0.00` to `1.00`.

It is built from:
- fill ratio
- contour solidity
- difference from the learned background

Higher score means the blob looks more like a real moving object.

This is not a neural-network confidence score.

## Why Shadows and Lighting Can Still Appear

MOG2 detects foreground change, not true object meaning.

So if:
- lighting changes strongly
- shadow moves across the belt
- dust creates a visible change

then MOG2 may still produce a box.

The filters reduce these false detections, but they cannot remove them completely.

## Main Arguments

Common useful arguments:

- `--source`
  Video path or webcam index like `0`
- `--output`
  Output video path
- `--no-save`
  Disable video saving
- `--no-show`
  Disable preview window
- `--history`
  Background memory length
- `--var-threshold`
  MOG2 change sensitivity
- `--warmup`
  Frames used to learn the background first
- `--min-area`
  Removes tiny dust/noise blobs
- `--min-fill-ratio`
  Rejects weak hollow blobs
- `--min-solidity`
  Rejects irregular shadow-like shapes
- `--min-diff`
  Rejects weak lighting-only changes
- `--min-score`
  Only show stronger detections
- `--min-persist`
  Ignore one-frame flicker
- `--max-aspect-ratio`
  Reject long glare streaks
- `--roi-left`, `--roi-right`, `--roi-top`, `--roi-bottom`
  Restrict detection to the conveyor area only
- `--box-pad`
  Add padding around the final box
- `--merge-gap`
  Merge nearby detections into one box

## Run Commands

Default video:

```bash
python3 /home/vikbot/Documents/countbot/testing/MOG2/detect.py
```

Use webcam:

```bash
python3 /home/vikbot/Documents/countbot/testing/MOG2/detect.py --source 0
```

Do not save output:

```bash
python3 /home/vikbot/Documents/countbot/testing/MOG2/detect.py --no-save
```

Stronger filtering for dust, shadow, and lighting:

```bash
python3 /home/vikbot/Documents/countbot/testing/MOG2/detect.py \
  --min-area 2200 \
  --min-fill-ratio 0.28 \
  --min-solidity 0.42 \
  --min-diff 26 \
  --min-score 0.82 \
  --min-persist 3 \
  --max-aspect-ratio 3.5 \
  --mask-blur 7
```

Use ROI for only the conveyor lane:

```bash
python3 /home/vikbot/Documents/countbot/testing/MOG2/detect.py \
  --min-area 2200 \
  --min-fill-ratio 0.28 \
  --min-solidity 0.42 \
  --min-diff 26 \
  --min-score 0.82 \
  --min-persist 3 \
  --max-aspect-ratio 3.5 \
  --mask-blur 7 \
  --roi-left 0.12 \
  --roi-right 0.88 \
  --roi-top 0.18 \
  --roi-bottom 0.92
```

## What You See On Screen

- green boxes for detected objects
- `object 0.xx` score on each box
- top text with frame number, FPS, and detection count
- orange ROI rectangle
- foreground mask preview beside the main video

## Best Use

Best use for this method:
- quick conveyor motion detection
- simple product box proposals
- fast preview with low complexity

If you need stronger product-vs-shadow understanding, combine this with:
- ROI masking
- confirmation logic
- FastSAM or YOLO style model detection
