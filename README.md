# Dashcam event detection

Find speedbreaker crossings and triangular warning signs in dashcam footage. Review crossings in a highlight video or jump to individual events in the original clips.

## Tested cameras

- VIOFO A229 Plus (front camera)

To request support for another camera, [open a GitHub issue](https://github.com/r-sniper/dashcam-event-detection/issues/new).

## How to run

### Prerequisites

Install uv, FFmpeg (includes ffprobe), and ExifTool. On the first run, uv sets up Python 3.12 or newer and the project dependencies.

**Windows (PowerShell)**

```powershell
winget install -e --id astral-sh.uv
winget install -e --id Gyan.FFmpeg
winget install -e --id OliverBetz.ExifTool
```

**macOS** with [Homebrew](https://brew.sh/)

```sh
brew install uv ffmpeg exiftool
```

**Linux (Ubuntu/Debian)**

```sh
sudo apt update && sudo apt install -y curl ffmpeg libimage-exiftool-perl
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Scan your clips

1. Download or clone this repository. Open a new terminal in the folder containing `analyze.py`.
2. Put your original front-camera clips directly in `videos/sample/`. Keep the filenames ending in `F.MP4`. The speedbreaker detector uses embedded GPS speed and reads filename timestamps as India Standard Time (UTC+05:30).
3. Run the default speedbreaker scan.

   ```sh
   uv run analyze.py videos/sample
   ```

Results appear in `analysis/speedbreaker/`.

- `detections.csv` has one row per event. Open the clip named in `filename` and seek to `peak_s` seconds from its start.
- `detections.clips.csv` lists each scanned clip, the seconds inspected, and the event count, including zero.
- `detections.speedbreakers.mp4` plays the crossings in sequence when the scan finds any.

## Additional options

The default detector is `speedbreaker`. To also look for triangular warning signs, select both detectors.

```sh
uv run analyze.py videos/sample --detect speedbreaker,warning_sign
```

Add `--review review.html` for a clickable event list. Run `uv run analyze.py --help` for all options.

To detect vehicles and read Indian registration plates, run the scan with the vehicle extra.

```sh
uv run --extra vehicle analyze.py videos/sample --detect vehicle
```

The scan lists the selected detectors and logs when each starts and finishes on a clip. In a terminal, progress bars show completed clips and sampled frames for each active vehicle scan, with an estimated time remaining. Redirected logs report vehicle progress about every five seconds as frames finish.

## Contributing

Submit every change through a pull request. Describe what changed and how you tested it. Every pull request requires approval from [Rahul Shah (@r-sniper)](https://github.com/r-sniper) before it can be merged.

## License and citation

This project uses the [MIT License](LICENSE). If you use it in published work, please cite the software.

> Shah, Rahul (2026). *Dashcam event detection* [Software]. [GitHub repository](https://github.com/r-sniper/dashcam-event-detection).
