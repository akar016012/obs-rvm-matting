# obs-rvm-matting

This is an OBS Studio filter for macOS on Apple silicon that cuts a person out of a camera feed, so the background can be replaced without a green screen. It runs the [Robust Video Matting](https://github.com/PeterL1n/RobustVideoMatting) (RVM) model through Core ML at full 1920x1080 resolution. It aims for a sharp, steady edge around hair, headphones and shoulders that keeps up with you when you move.

The filter shows up in OBS as **Person Matting (RVM)**.

## Features

- The cutout is computed at full 1080p resolution on the Neural Engine or the graphics chip, at about 28 ms per frame on the author's Apple silicon Mac.
- Each camera frame can wait for its own cutout, so a waving hand or a raised mug is cut where it is, not where it was a frame earlier. This costs about one frame of delay.
- The edge is held steady where the picture is still, and camera noise along dark edges such as headphone foam is smoothed out, so the outline does not dance.
- An optional photo of the empty room removes leftovers the model is unsure about and opens thin gaps, such as the space beside a headband.
- Optional automatic color corrects brightness and white balance as the light changes, without changing your skin tone.

## Requirements

- You need an Apple silicon Mac running macOS 13 or later.
- OBS Studio must be installed in `/Applications`. The plugin was built and tested against OBS 33.0.0, and the build fetches the OBS headers that match the installed version.
- The Xcode command line tools must be installed, because the build uses `clang`, `codesign` and `xcrun coremlcompiler`. Run `xcode-select --install` if they are missing.
- `git`, `curl` and [uv](https://docs.astral.sh/uv/) must be on your `PATH`. uv runs the one-off model conversion in a throwaway Python 3.12 environment, so nothing is installed into your system Python.
- The first build downloads about 1.5 GB (mostly PyTorch for the model conversion) and takes several minutes.

## Installation

1. Clone the repository and open a terminal in it.

   ```bash
   git clone https://github.com/akar016012/obs-rvm-matting.git
   ```

   ```bash
   cd obs-rvm-matting
   ```

2. Quit OBS if it is running, then build and install the plugin. The plugin is copied into `~/Library/Application Support/obs-studio/plugins/`.

   ```bash
   ./build.sh --install
   ```

3. Start OBS. Right-click your camera source, choose **Filters**, click **+** under **Audio/Video Filters**, and add **Person Matting (RVM)**.
4. Add an image or video source for your new background, and drag it below the camera source in the scene's source list.
5. If you want to use the result in a video call, click **Start Virtual Camera** in OBS and choose **OBS Virtual Camera** as the camera in your meeting app.

`./build.sh` without `--install` builds `build/obs-rvm-matting.plugin` without installing it.

### Recommended camera settings

- Turn off **Use Buffering** in the camera source's properties. Buffering adds a noticeable delay between your voice and your picture.
- Put this filter first on the camera, ahead of any **Crop/Pad** filter, because the cutout is made from the whole camera frame.
- Set the camera to 1920x1080 at 30 fps. Both models expect 1080p frames. Other sizes are stretched to 16:9 for the model, and the cutout still lines up with the picture.

### Uninstalling

Quit OBS, then delete the plugin and its saved background photo.

```bash
rm -rf ~/Library/Application\ Support/obs-studio/plugins/obs-rvm-matting.plugin
```

```bash
rm -rf ~/Library/Application\ Support/obs-studio/plugin_config/obs-rvm-matting
```

## Settings

- **Model** chooses between the two models described below. **High accuracy** is the default. A new model loads in the background, and the current one keeps running until it is ready, so changing this setting never freezes the video.
- **Run on** picks the Neural Engine, the graphics chip or automatic placement. The Neural Engine is the default, because OBS itself keeps the graphics chip busy.
- **Hide faint edges below** and **Make solid above** set how the soft edge is drawn. Matte values below the first become transparent and values above the second become solid. The defaults of 0.25 and 0.75 give a crisper edge than the model's raw output.
- **Steady edges** holds the cutout still where the camera picture has not changed between frames, so the edge stops flickering. Where the picture moves, the cutout follows at once. Set it to 0 to turn the steadying off. The light blur that removes camera noise from the edge always applies.
- **Line up cutout with video** holds each camera frame until its own cutout is ready, so fast-moving hands and objects are cut where they are. It delays the video by about one frame (33 ms at 30 fps). If four frames are already waiting, the oldest is shown anyway with the newest earlier cutout. If no cutout has arrived for half a second, for example while the first model is still loading when OBS starts, frames are shown without waiting, so the video never freezes. Until the first cutout arrives they are shown uncut; after that, the newest available cutout is used.
- **Automatic color** corrects brightness and white balance by itself, for any camera, and follows the light through the day. It is off by default. Turn off any **Color Correction** filter on the same camera while it is on, or the two corrections add up.
  - Brightness is set from the room, not from you: the brightest 2% of the background are brought to a fixed level, and mid-tones get a gentle lift like a phone camera's. If you fill almost the whole picture, the whole picture is used instead.
  - White balance moves plain background areas, such as walls, part of the way toward neutral, so a warm wall does not turn skin cool and pale. If there are almost none in view, the last white balance is kept.
  - Skin never sets the white balance, and sets the brightness only in a close-up with almost no background in view, so every skin tone keeps its own color.
  - Changes are made gradually over a few seconds, both when the light changes and when you turn the setting on or off, so the picture never visibly pumps. The current red, green and blue gains, which include both exposure and white balance, are written to the OBS log every 30 seconds.
  - It only adjusts standard (SDR) camera video. HDR and wide-gamut sources are left unchanged.
- **Use empty-background photo** cuts out anything that closely matches a photo of your empty room, including thin gaps such as the space between headphones and your head. It is off by default. Click **Capture empty background (in 5 seconds)**, which also turns it on, and step out of view. The button counts down, with a tick each second, and capture then averages 30 frames. Stay out of view until you hear the final chime. If the camera stops during capture, you hear an error sound and the button says so. Each frame is matched to the photo's brightness before comparing, and each pixel's own camera noise is allowed for. Apart from thin gaps, it only cuts pixels the model is unsure about. A chair that moves will no longer match the photo. Each filter keeps its own photo, saved in the plugin's OBS config folder and reloaded on start. The photo gradually follows small lighting changes where the model is sure it sees background. **Forget background photo** deletes it. The setting does nothing until a photo has been taken.

The solid middle of your body, which is the fully opaque part of the cutout with small gaps filled and shrunk inward by about 16 pixels, is not cut by the background photo, except for thin gaps near the edge of the upper half of your body that match the photo almost exactly. Clothes that happen to match what is behind them therefore stay whole.

## Models

The Core ML files published with RVM use Core ML's older NeuralNetwork format and took about 60 ms per 1080p frame inside OBS. `tools/convert_model.py` re-exports the PyTorch weights as a float16 ML Program and computes only the alpha matte, not the foreground color, which makes them several times faster. `build.sh` downloads the RVM v1.0.0 weights and runs the conversion for you.

| Model setting           | Network     | Time per frame inside OBS              | Notes                                                                   |
| ----------------------- | ----------- | -------------------------------------- | ----------------------------------------------------------------------- |
| High accuracy (default) | ResNet-50   | About 28 ms on the Neural Engine       | It keeps chair mesh and gaps beside the neck out of the cutout better.  |
| Standard                | MobileNetV3 | About 17 to 22 ms on the graphics chip | It is lighter, but it often counts a mesh headrest as part of a person. |

Both models take 1920x1080 frames, run the network on a quarter-size copy, and use RVM's guided-filter refiner to return a full-size matte. The timings were measured on the author's Mac and will differ on other chips.

### Other ResNet-50 weights

To build the plugin with other ResNet-50 weights, such as ones you trained yourself, set `RVM_RESNET50_WEIGHTS`:

```bash
RVM_RESNET50_WEIGHTS=path/to/weights.pth ./build.sh --install
```

The **High accuracy** setting then runs those weights. Running `./build.sh --install` without the variable goes back to the release model.

## Troubleshooting

- **The filter is missing from the list.** Check the OBS log (**Help → Log Files → Show Log Files**) for lines starting with `[rvm-matting]`. A missing line means OBS did not load the plugin; make sure it is in `~/Library/Application Support/obs-studio/plugins/` and that you restarted OBS after installing.
- **The cutout lags or stutters.** The log prints a line like the one below every 30 seconds. If `inference avg` is above 33 ms, switch **Model** to **Standard**, or try the other **Run on** choice.
- **Something behind you keeps appearing.** Try an empty-background photo, and turn it off if it cuts into your neck or shoulders. It works best when the camera, the lighting and the chair stay exactly where they were.
- **Parts of you disappear when you hold still against a similar-colored wall.** Click **Forget background photo**, or turn off **Use empty-background photo**.

```
[rvm-matting] Matte updates: 26.8 per second, inference avg 27.9 ms, max 41.4 ms, frames skipped 96, flicker 0.433 raw / 0.302 shown, still-edge flicker 6.89 raw / 2.08 shown
```

The flicker figures are the mean change between consecutive cutouts, out of 255. The first pair is over the whole picture. The second pair covers only edge pixels where the picture is still, which is where a dancing edge shows. Each pair gives the cutout before the edge blur and the steadying first, and what is shown second. The first number already includes the background photo.

## How it works

1. OBS calls the filter's `filter_video` on its video thread for each camera frame. The filter copies the frame into a pooled `CVPixelBuffer` and hands it to a background queue. If the model is still busy, or two finished cutouts are still waiting for the second queue, the frame is not sent to the model, so the cutout always follows the newest frame and no backlog builds up.
2. The model queue converts the frame to BGRA at the model size, brightens a copy for the model when the frame is dim, which includes many ordinary indoor scenes, runs the model with its four recurrent states, and feeds the new states back in for the next frame. The states are reset after more than a second without frames.
3. A second queue finishes each matte while the model starts on the next frame. It measures where the picture moved since the last frame on a quarter-size grid, applies the background photo, blurs the matte over 5 pixels to remove camera noise, steadies it where the picture is still, and stores it with the timestamp of the frame it came from. The last four mattes are kept.
4. With **Line up cutout with video** on, `filter_video` holds incoming frames and releases the newest one that has a matte, in the same way as OBS's own Video Delay (Async) filter.
5. `video_render` uploads the matte for the frame being shown as a texture. `data/rvm-matting.effect` applies the automatic color gains in linear light when that setting is on, and multiplies the premultiplied camera image by the matte, using the same premultiplied blending as OBS's chroma key filter.

### Checking edge quality

The filter has one hidden setting, `debug_snapshot`, which is not shown in the properties dialog. Set it to a plain file name ending in `.png`, such as `matte.png`, in the scene collection JSON while OBS is closed. Other names, and the name of the saved background photo, are ignored. Ninety mattes after the model starts, the filter writes the matte as a grayscale PNG with that name into `~/Library/Application Support/obs-studio/plugin_config/obs-rvm-matting/`, and then clears the setting. Because only plain file names are accepted, the snapshot is never written anywhere else. The image does not contain the camera picture itself.

At the same moment, the filter writes a line like this to the OBS log, which describes the raw camera color before any other filter runs:

```
[rvm-matting] Camera color: skin 137 88 67, bright background 153 154 162, dark clothing 20 21 22 (R G B), brightness p1 8 p50 140 p99 173
```

If you prefer fixed settings to **Automatic color**, compare it with the same areas in a reference picture you like, such as one from another camera app, to choose settings for OBS's **Color Correction** filter. That filter works in linear light: **Gamma** lifts the mid-tones, **Contrast** acts as an overall gain, and **Color Multiply** sets the white balance.

## Project layout

- `src/rvm-matting.m` holds the whole plugin: the Core ML engine and the OBS filter.
- `data/rvm-matting.effect` is the shader that applies the matte.
- `data/locale/en-US.ini` holds the text shown in OBS.
- `tools/convert_model.py` converts the RVM weights to Core ML.
- `build.sh` fetches the dependencies, converts the models, builds, signs and installs the plugin. `RVM_RESNET50_WEIGHTS` makes it build the High accuracy model from other weights.
- `Info.plist` is the plugin bundle's property list.

## Contributing

Bug reports, ideas and pull requests are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) before you start, and follow the [code of conduct](CODE_OF_CONDUCT.md). To report a security problem, follow [SECURITY.md](SECURITY.md) instead of opening a public issue.

## License

This project is licensed under the GNU General Public License version 3 or later; see [LICENSE](LICENSE). The RVM models and weights are released under GPL-3.0 by their authors, and OBS Studio is GPL-2.0-or-later, so the plugin and the converted models it bundles are distributed under GPL-3.0.

## Acknowledgements

This plugin would not exist without Robust Video Matting by Shanchuan Lin, Linjie Yang, Imran Saleemi and Soumyadip Sengupta. If you use it in research, please cite their paper:

```bibtex
@inproceedings{lin2022robust,
  title={Robust High-Resolution Video Matting with Temporal Guidance},
  author={Lin, Shanchuan and Yang, Linjie and Saleemi, Imran and Sengupta, Soumyadip},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={238--247},
  year={2022}
}
```

The filter's rendering follows OBS Studio's own chroma key and video delay filters.
