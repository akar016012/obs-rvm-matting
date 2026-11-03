# Contributing to obs-rvm-matting

Thank you for taking the time to help. This project is small and maintained in spare time, so clear reports and focused pull requests make the biggest difference.

Everyone who takes part is expected to follow the [code of conduct](CODE_OF_CONDUCT.md).

## Reporting a bug

Before opening an issue, please search the existing issues to see whether it has already been reported. When you open one, use the bug report form and include the following:

- Give your Mac's chip, your macOS version and your OBS version.
- Give the filter settings you used, in particular **Model** and **Run on**.
- Paste the `[rvm-matting]` lines from the OBS log (**Help → Log Files → Show Log Files**).
- Describe what you expected and what happened. A short screen recording or a cropped screenshot helps a lot with edge problems.

Please do not post screenshots that show other people or anything private in your room.

## Suggesting a feature

Open an issue with the feature request form and describe the problem you want solved before the solution you have in mind. That makes it easier to find an approach that fits the plugin.

## Making a change

1. Fork the repository and create a branch for your change.
2. Build and install your version with `./build.sh --install`, then restart OBS.
3. Test the change with a real camera. Check the `[rvm-matting]` log lines, and use the hidden `debug_snapshot` setting described in the README if you are changing how the edge is computed.
4. Update the README and `data/locale/en-US.ini` whenever you add, rename or change a setting.
5. Add a line to `CHANGELOG.md` under **Unreleased**.
6. Open a pull request that explains what you changed, why, and how you tested it. Include before and after log lines for changes that affect speed or flicker.

### Code style

- The Objective-C in `src/` uses OBS Studio's style: tabs for indentation, braces on their own line for functions, and lines up to 120 characters.
- Python in `tools/` is formatted with `autopep8`.
- Markdown and YAML files are formatted with Prettier.
- Comments explain why the code does something, in full sentences.

### Things that need special care

- `filter_video`, `update` and `video_render` all run on the OBS video thread. Anything slower than a memory copy belongs on the engine's background queues.
- Frames held by **Line up cutout with video** must always be released with `obs_source_release_frame`, or the camera source runs out of frames.
- The RVM models are GPL-3.0. Do not add code or models under a licence that is incompatible with GPL-3.0.

## Licence of contributions

By contributing, you agree that your contributions are licensed under the GNU General Public License version 3 or later, the same licence as the rest of the project.
