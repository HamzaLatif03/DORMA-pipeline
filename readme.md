# DORMA Pipeline

Stream camera + mic from an iPhone to a Mac server with real-time face recognition and live transcription.

## Setup

### Option 1: Conda (recommended)

Creates the environment with all dependencies (including prebuilt dlib — no C++ compiler needed):

```bash
conda env create -f environment.yml
conda activate dorma_pipeline
```

### Option 2: pip

Requires Python 3.10, [CMake](https://cmake.org/download/), and Xcode Command Line Tools (macOS) for building dlib.

```bash
# macOS
brew install cmake
xcode-select --install

# Ubuntu/Debian
sudo apt install cmake build-essential

# Install dlib via conda first (avoids build issues), then pip for the rest
conda install -c conda-forge dlib "numpy>=1.24,<2"
pip install -r requirements.txt
```

> **Note:** numpy must stay below 2.0 — dlib and opencv are not compatible with numpy 2.x on Python 3.10.

## Run

### Start the server

```bash
uvicorn run:app --host 0.0.0.0 --port 8000
```

### Expose over HTTPS (required for iPhone camera)

```bash
ngrok http 8000
```

Open the `https://` URL from ngrok on your iPhone. iOS Safari requires HTTPS for camera access.

## Usage

1. On the **iPhone**, open the ngrok HTTPS URL and tap **Connect & start stream**.
2. On the **Mac**, open the same URL and tap **View stream** to see the live video with face IDs and transcription.
