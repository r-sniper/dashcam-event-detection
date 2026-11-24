# Dashcam event detection

Detect speedbreaker crossings from camera motion and GPS speed, and look for triangular warning signs in the front-camera footage. The current input format is VIOFO front-camera MP4s with filenames ending in `F.MP4`.

## Run a scan

Install `uv` and put FFmpeg, FFprobe and ExifTool on your PATH. Pass a folder of clips or individual files. This command scans for crossings, looks for signs on each approach, and writes an offline review page.

```powershell
uv sync --locked
uv run --locked analyze.py videos/complete --detect speedbreaker,warning_sign --sign-lookback 15 --output analysis/events.csv --review visualize/events.html
```

Open `visualize/events.html` to jump to detections in the original footage. The event CSV includes the filename, clip-relative timestamp and score. The companion `events.clips.csv` lists coverage for each clip.

Each speedbreaker row also reports `start_s` and `end_s` in seconds from the start of its source clip. The interval covers the motion samples behind consecutive above-threshold scores around `peak_s`, bounded by the scan window and the midpoints to neighboring peaks.

After saving the CSV, the scan joins those intervals in CSV order into `events.speedbreakers.mp4` beside it. The compilation uses 720p video at 30 fps and includes the source audio when available. A run with zero crossings clears any previous compilation for that output name.

Each run prepares a `ProcessedVideo` directly from each MP4 before detection. It contains container and stream metadata, timestamped GPS coordinates, speed and heading. Speedbreaker runs also measure camera motion at 10 Hz. The speedbreaker detector requires motion samples and at least one GPS speed aligned to the clip. A GPS gap can leave an individual crossing's speed empty. Warning-sign detection only needs the video frames and duration.

Preprocessing writes one archive per clip and extraction configuration under `.speedbreaker-work/preprocessed/`. By default each run refreshes the archive. Add `--preprocess-cache` to reuse it when the source path, size, modification time, preprocessing version and extraction settings match. Use `--work-dir PATH` to choose another working directory. The existing `--motion-cache PATH` flag can still reuse optical flow independently.

## Contributing

Submit every change through a pull request. Describe what changed and how you tested it. Every pull request requires approval from [Rahul Shah (@r-sniper)](https://github.com/r-sniper) before it can be merged.

## License and citation

This project uses the [MIT License](LICENSE). If you use it in published work, please cite the software.

> Shah, Rahul (2026). *Dashcam event detection* [Software]. [GitHub repository](https://github.com/r-sniper/dashcam-event-detection).
