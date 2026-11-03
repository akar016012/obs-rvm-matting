# obs-rvm-matting

This is an OBS Studio filter for macOS on Apple silicon that cuts a person out of a camera feed, so the background can be replaced without a green screen. It runs the [Robust Video Matting](https://github.com/PeterL1n/RobustVideoMatting) (RVM) model through Core ML at full 1920x1080 resolution. It aims for a sharp, steady edge around hair, headphones and shoulders that keeps up with you when you move.

The filter shows up in OBS as **Person Matting (RVM)**.

## Features

- The cutout is computed at full 1080p resolution on the Neural Engine or the graphics chip, at about 28 ms per frame on the author's Apple silicon Mac.
- Each camera frame can wait for its own cutout, so a waving hand or a raised mug is cut where it is, not where it was a frame earlier. This costs about one frame of delay.
- The edge is held steady where the picture is still, and camera noise along dark edges such as headphone foam is smoothed out, so the outline does not dance.
- An optional photo of the empty room removes anything the model mistakes for you, such as a lamp or a door.
- An optional colour rule always removes a red or magenta chair behind you, such as a mesh headrest.

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

3. Start OBS. Right-click your camera source, choose **Filters**, click **+** under **Effect Filters**, and add **Person Matting (RVM)**.
4. Add an image or video source for your new background, and drag it below the camera source in the scene's source list.
5. If you want to use the result in a video call, click **Start Virtual Camera** in OBS and choose **OBS Virtual Camera** as the camera in your meeting app.

`./build.sh` without `--install` builds `build/obs-rvm-matting.plugin` without installing it.

### Recommended camera settings

- Turn off **Use Buffering** in the camera source's properties. Buffering adds a noticeable delay between your voice and your picture.
- Set the camera to 1920x1080 at 30 fps. Both models expect 1080p frames, and other sizes are scaled to fit.

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
- **Remove my red chair** always cuts out red or magenta pixels outside the solid middle of your body, such as a red mesh headrest that the model sometimes counts as part of you. It tells the chair apart from skin and wood by its magenta tint: the mesh has more blue than green, while skin and wood have more green than blue. It is off by default, because it only suits chairs of that colour. Three details keep the edge clean:
  - Where the model counts a pixel as mostly you, the pixel is only cut if it looks like chair itself, so a black earcup in front of the chair keeps its edge.
  - Close to where the chair was cut, strips of the cutout thinner than about 9 pixels are removed, which takes away the chair's black rim.
  - Close to the cut, any remaining pixel with the chair's magenta tint has its red pulled down to its other colours, so a leftover rim looks like neutral shadow instead of a red fringe.
- **Use empty-background photo** cuts out anything that matches a photo of your empty room. Click **Capture empty background (in 5 seconds)** and step out of view, and the photo is taken five seconds later. It is saved as `background-photo.png` in the plugin's OBS config folder and reloaded on start. The photo slowly follows lighting changes where the model is sure it sees background. **Forget background photo** deletes it. The setting does nothing until a photo has been taken.

The solid middle of your body, which is the fully opaque part of the cutout shrunk inwards by about 16 pixels, is never cut by the chair rule or by the background photo. Clothes that happen to match what is behind them therefore keep no holes.

## Models

The Core ML files published with RVM use Core ML's older NeuralNetwork format and took about 60 ms per 1080p frame inside OBS. `tools/convert_model.py` re-exports the PyTorch weights as a float16 ML Program and computes only the alpha matte, not the foreground colour, which makes them several times faster. `build.sh` downloads the RVM v1.0.0 weights and runs the conversion for you.

| Model setting           | Network     | Time per frame inside OBS              | Notes                                                                   |
| ----------------------- | ----------- | -------------------------------------- | ----------------------------------------------------------------------- |
| High accuracy (default) | ResNet-50   | About 28 ms on the Neural Engine       | It keeps chair mesh and gaps beside the neck out of the cutout better.  |
| Standard                | MobileNetV3 | About 17 to 22 ms on the graphics chip | It is lighter, but it often counts a mesh headrest as part of a person. |

Both models take 1920x1080 frames, run the network on a quarter-size copy, and use RVM's guided-filter refiner to return a full-size matte. The timings were measured on the author's Mac and will differ on other chips.

## Troubleshooting

- **The filter is missing from the list.** Check the OBS log (**Help → Log Files → Show Log Files**) for lines starting with `[rvm-matting]`. A missing line means OBS did not load the plugin; make sure it is in `~/Library/Application Support/obs-studio/plugins/` and that you restarted OBS after installing.
- **The cutout lags or stutters.** The log prints a line like the one below every 30 seconds. If `inference avg` is above 33 ms, switch **Model** to **Standard**, or try the other **Run on** choice.
- **Something behind you keeps appearing.** Take an empty-background photo, or turn on **Remove my red chair** if it is a red chair.
- **Parts of you disappear when you hold still against a similar-coloured wall.** Click **Forget background photo**, or turn off **Use empty-background photo**.

```
[rvm-matting] Matte updates: 26.8 per second, inference avg 27.9 ms, max 41.4 ms, frames skipped 96, flicker 0.433 raw / 0.302 shown, still-edge flicker 6.89 raw / 2.08 shown
```

The flicker figures are the mean change between consecutive cutouts, out of 255. The first pair is over the whole picture. The second pair covers only edge pixels where the picture is still, which is where a dancing edge shows. Each pair gives the cutout before the edge blur and the steadying first, and what is shown second. The first number already includes the background photo and the chair rule.

## How it works

1. OBS calls the filter's `filter_video` on its video thread for each camera frame. The filter copies the frame into a pooled `CVPixelBuffer` and hands it to a background queue. If the model is still busy, or two finished cutouts are still waiting for the second queue, the frame is not sent to the model, so the cutout always follows the newest frame and no backlog builds up.
2. The model queue converts the frame to BGRA at the model size, runs the model with its four recurrent states, and feeds the new states back in for the next frame. The states are reset after more than a second without frames.
3. A second queue finishes each matte while the model starts on the next frame. It measures where the picture moved since the last frame on a quarter-size grid, applies the background photo and the chair rule, blurs the matte over 5 pixels to remove camera noise, steadies it where the picture is still, and stores it with the timestamp of the frame it came from. The last four mattes are kept.
4. With **Line up cutout with video** on, `filter_video` holds incoming frames and releases the newest one that has a matte, in the same way as OBS's own Video Delay (Async) filter.
5. `video_render` uploads the matte for the frame being shown as a two-channel texture, whose second channel marks where to remove the chair's colour. `data/rvm-matting.effect` removes that colour and multiplies the premultiplied camera image by the matte, using the same colour-space handling and blending as OBS's chroma key filter.

### Checking edge quality

The filter has one hidden setting, `debug_snapshot`, which is not shown in the properties dialog. Set it to a plain file name ending in `.png`, such as `matte.png`, in the scene collection JSON while OBS is closed. Other names, and the name of the saved background photo, are ignored. Ninety mattes after the model starts, the filter writes the matte as a grayscale PNG with that name into `~/Library/Application Support/obs-studio/plugin_config/obs-rvm-matting/`, and then clears the setting. Because only plain file names are accepted, the snapshot is never written anywhere else. When the chair rule is on, it also writes `-chair.png` (what was cut) and `-chairscore.png` (how chair-like each grid cell looked) next to it. None of these images contains the camera picture itself.

## Project layout

- `src/rvm-matting.m` holds the whole plugin: the Core ML engine and the OBS filter.
- `data/rvm-matting.effect` is the shader that applies the matte.
- `data/locale/en-US.ini` holds the text shown in OBS.
- `tools/convert_model.py` converts the RVM weights to Core ML.
- `build.sh` fetches the dependencies, converts the models, builds, signs and installs the plugin.
- `Info.plist` is the plugin bundle's property list.

## Contributing

Bug reports, ideas and pull requests are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) before you start, and follow the [code of conduct](CODE_OF_CONDUCT.md). To report a security problem, follow [SECURITY.md](SECURITY.md) instead of opening a public issue.

## Licence

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
