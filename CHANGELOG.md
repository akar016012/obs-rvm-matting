# Changelog

All notable changes to this project are recorded in this file. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses [semantic versioning](https://semver.org/).

## Unreleased

### Added

- **Automatic color** corrects brightness from the scene's highlights and white balance from plain background areas, gradually, without letting skin set the white balance.
- A debug snapshot now also logs the camera's raw color (skin, bright background, dark clothing and brightness spread), for tuning OBS's color correction without saving the picture.

## 1.0.0 - 2026-11-03

### Added

- The **Person Matting (RVM)** filter cuts a person out of a camera source with the Robust Video Matting model at 1920x1080.
- Two models are available: **High accuracy** (ResNet-50) and **Standard** (MobileNetV3), and either can run on the Neural Engine or the graphics chip.
- **Line up cutout with video** holds each frame until its own cutout is ready.
- **Steady edges** holds the edge still where the picture is still, and a light blur removes camera noise from the edge.
- **Use empty-background photo** removes anything that matches a photo of the empty room.
- **Remove my red chair** always removes a red or magenta chair behind the person, together with thin leftovers of its black rim, and turns any remaining red fringe into neutral shadow.
- `build.sh` fetches the OBS headers and the model weights, converts the models to Core ML, and builds, signs and installs the plugin.
