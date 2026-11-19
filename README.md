# Dashcam event detection

Detect speedbreaker crossings from camera motion and GPS speed, and look for triangular warning signs in the front-camera footage. The current input format is VIOFO front-camera MP4s with filenames ending in `F.MP4`.

## Run a scan

Install `uv` and put FFmpeg and FFprobe on your PATH. Pass a folder of clips or individual files. This command scans for crossings, looks for signs on each approach, and writes an offline review page.

```powershell
uv sync --locked
uv run --locked analyze.py videos/complete --detect speedbreaker,warning_sign --sign-lookback 15 --output analysis/events.csv --review visualize/events.html
```

Open `visualize/events.html` to jump to detections in the original footage. The event CSV includes the filename, clip-relative timestamp and score. The companion `events.clips.csv` lists coverage for each clip.

For crossings alone, use `--detect speedbreaker` and omit `--sign-lookback`. GPS speeds come from `telemetry/<clip stem>.telemetry.csv` alongside the footage, or from the MP4 when ExifTool is on PATH. The default scan can also detect crossings using camera motion alone.
