# Dashcam event detection

Find speedbreaker crossings and triangular warning signs in dashcam footage. Review crossings in a highlight video or jump to individual events in the original clips.

## Tested cameras

- VIOFO A229 Plus (front camera)

Speedbreaker detection uses the camera's recorded GPS speed and reads the timestamp in each filename as India Standard Time (UTC+05:30).

To request support for another camera, [open a GitHub issue](https://github.com/r-sniper/dashcam-event-detection/issues/new).

## How to run

### Prerequisites

Run the commands for your operating system. On the first run, uv sets up Python and the project's dependencies.

**Windows (PowerShell)**

```powershell
winget install -e --id Git.Git
winget install -e --id astral-sh.uv
winget install -e --id Gyan.FFmpeg
winget install -e --id OliverBetz.ExifTool
```

**macOS** with [Homebrew](https://brew.sh/)

```sh
brew install git uv ffmpeg exiftool
```

**Linux (Ubuntu/Debian)**

```sh
sudo apt update && sudo apt install -y git curl ffmpeg libimage-exiftool-perl
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Find speedbreakers in your footage

1. Open a new terminal and download the project.

   ```sh
   git clone https://github.com/r-sniper/dashcam-event-detection.git
   ```

2. Enter the project folder. Run the remaining commands here.

   ```sh
   cd dashcam-event-detection
   ```

3. Create a folder for your footage.

   ```sh
   mkdir videos
   mkdir videos/sample
   ```

4. Copy your front-camera files ending in `F.MP4` directly into `videos/sample/`. Keep their original names.

5. Find speedbreaker crossings.

   ```sh
   uv run analyze.py videos/sample
   ```

6. Open `analysis/speedbreaker/` inside the project to see your results.

   - Play `detections.speedbreakers.mp4` to watch the crossings in sequence. This video appears when the tool finds a crossing.
   - Open `detections.csv` to find a crossing in the original footage. Play the clip named in `filename` and jump to `peak_s` seconds from its start.
   - Open `detections.clips.csv` to see how many crossings each clip had, including zero.

## Additional options

To find both speedbreakers and triangular warning signs, run this command from the project folder.

```sh
uv run analyze.py videos/sample --detect speedbreaker,warning_sign
```

Add `--review review.html` for a clickable event list. Run `uv run analyze.py --help` for all options.

## Contributing

Submit every change through a pull request. Describe what changed and how you tested it. Every pull request requires approval from [Rahul Shah (@r-sniper)](https://github.com/r-sniper) before it can be merged.

## License and citation

This project uses the [MIT License](LICENSE). If you use it in published work, please cite the software.

> Shah, Rahul (2026). *Dashcam event detection* [Software]. [GitHub repository](https://github.com/r-sniper/dashcam-event-detection).
