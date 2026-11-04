# Changelog

All notable changes to this project are recorded in this file. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses [semantic versioning](https://semver.org/).

## Unreleased

### Added

- **Automatic color** corrects brightness from the scene's highlights and partly corrects white balance from plain background areas, gradually, without letting skin set the white balance and without washing out skin.
- A debug snapshot now also logs the camera's raw color (skin, bright background, dark clothing and brightness spread), for tuning OBS's color correction without saving the picture.
- `build.sh` builds the High accuracy model from other ResNet-50 weights when `RVM_RESNET50_WEIGHTS` is set.
- The cutout holds up better in dim rooms, because dim frames are brightened for the model only. The picture viewers see does not change.
- **Blend into the room** lights you to suit a chosen **Room background**, so you look like you are in that place.
- **Clean edge colors** removes the thin line of room color around the cutout. It is on by default and needs no background photo.
- Every setting now has a **?** help tooltip.
- **Use empty-background photo** now averages several frames, follows exposure changes, needs a much closer color match so it no longer cuts into necks and shoulders, and can open thin gaps such as the space beside a headband. Its capture button now counts down with sounds, and each filter now keeps its own photo. A photo saved by an earlier version goes to the first filter that loads, so other filters need a new one.

### Removed

- **Remove my red chair** is gone, because a rule for one color of chair does not help most people. Any saved setting for the old option is ignored.

### Fixed

- A speckled cutout no longer opens gaps in the protected middle of the body.
- OBS no longer crashes when you remove a filter that failed to start.
- **Steady edges** no longer holds a closed gap on screen as a square patch, because large changes now show at once.
- **Use empty-background photo** no longer cuts through dark headphones that pass in front of a dark chair.

## 1.0.0 - 2026-11-03

### Added

- The **Person Matting (RVM)** filter cuts a person out of a camera source with the Robust Video Matting model at 1920x1080.
- Two models are available: **High accuracy** (ResNet-50) and **Standard** (MobileNetV3), and either can run on the Neural Engine or the graphics chip.
- **Line up cutout with video** holds each frame until its own cutout is ready.
- **Steady edges** holds the edge still where the picture is still, and a light blur removes camera noise from the edge.
- **Use empty-background photo** removes anything that matches a photo of the empty room.
- **Remove my red chair** always removes a red or magenta chair behind the person, together with thin leftovers of its black rim, and turns any remaining red fringe into neutral shadow.
- `build.sh` fetches the OBS headers and the model weights, converts the models to Core ML, and builds, signs and installs the plugin.
