# Security policy

## Supported versions

Only the latest commit on the `main` branch receives fixes.

## Reporting a vulnerability

Please do not report security problems in public issues. Report them privately through GitHub instead: open the repository's **Security** tab, choose **Report a vulnerability**, and describe the problem and how to reproduce it.

You can expect a first reply within two weeks. If the report is confirmed, a fix will be published on `main` and the report will be credited to you unless you ask otherwise.

## Scope

The plugin runs inside OBS Studio with the same permissions as OBS. It reads camera frames, and it writes files only to its own OBS config folder: the optional background photo and any debug snapshots you request. It only loads the two models bundled with it, whatever a scene collection says. `build.sh` downloads OBS, SIMDe and Robust Video Matting source code and the model weights from GitHub over HTTPS, and checks the weights against fixed SHA-256 checksums. Weights you supply yourself through `RVM_RESNET50_WEIGHTS` are not checked, so use only files you trust. The one-off model conversion also installs Python packages (PyTorch, torchvision, coremltools and NumPy) from PyPI into a temporary environment through uv. Problems in OBS Studio itself should be reported to the OBS project.
