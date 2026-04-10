# YOLOE Segmentation Test

This folder is for testing YOLOE with a segmentation head so we can get boxes and masks from a single inference pass.

## Files

- `detect.py` - runs YOLOE segmentation on an image, video, or webcam

## Expected weights

Put a YOLOE segmentation checkpoint in `testing/yoloe/weights/` or pass a full path with `--model`.

Examples:

- `yoloe-11s-seg.pt`
- `yoloe-26s-seg.pt`

Detection-only weights are not enough for mask output. Use a `-seg` checkpoint.

## Example

```bash
cd /home/vikbot/Documents/countbot/testing/yoloe
python3 detect.py \
  --model /path/to/yoloe-26s-seg.pt \
  --source /home/vikbot/Documents/countbot/testing/video/bearing_wheel.mp4 \
  --classes bearing wheel \
  --project-masks \
  --show
```

Output is saved by default to `testing/yoloe/runs/yoloe_output.mp4`.
