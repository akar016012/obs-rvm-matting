# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Aditya Kar

"""Converts Robust Video Matting weights to a Core ML ML Program.

The Core ML files published with RVM use the older NeuralNetwork format and
run at about 60 ms per 1080p frame on Apple silicon. This script re-exports
the weights as an ML Program in float16, and drops the foreground-color
output, so only the alpha matte is computed at full resolution. Benchmarked
on their own, the MobileNetV3 model then takes about 11 ms per frame on the
GPU and the ResNet-50 model about 23 to 30 ms; inside OBS, which shares the
GPU, the README gives the measured figures.

Usage:
    python convert_model.py RVM_REPO WEIGHTS.pth OUTPUT.mlpackage \
        [--variant mobilenetv3|resnet50] [--width 1920] [--height 1080] \
        [--downsample 0.25]

RVM_REPO is a checkout of https://github.com/PeterL1n/RobustVideoMatting and
WEIGHTS.pth is rvm_mobilenetv3.pth or rvm_resnet50.pth from its v1.0.0
release, or weights fine-tuned from them. build.sh runs
this script for both variants.
"""

import argparse
import sys

import coremltools as ct
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class AlphaOnlyRVM(nn.Module):
    """Runs RVM on one frame and returns the alpha matte and recurrent states.

    The refiner is the RVM deep guided filter restricted to the alpha channel:
    it upsamples the per-pixel linear coefficients for alpha only and applies
    them to the full-resolution luminance guide.
    """

    def __init__(self, model, base_size):
        super().__init__()
        self.model = model
        self.base_size = base_size

    def forward(self, src, r1, r2, r3, r4):
        m = self.model
        src_sm = F.interpolate(src, size=self.base_size,
                               mode="bilinear", align_corners=False)

        f1, f2, f3, f4 = m.backbone(src_sm)
        f4 = m.aspp(f4)
        hid, r1o, r2o, r3o, r4o = m.decoder(
            src_sm, f1, f2, f3, f4, r1, r2, r3, r4)
        base_fgr, base_pha = m.project_mat(hid).split([3, 1], dim=-3)

        refiner = m.refiner
        base_x = torch.cat([src_sm, src_sm.mean(1, keepdim=True)], dim=1)
        base_y = torch.cat([base_fgr, base_pha], dim=1)
        mean_x = refiner.box_filter(base_x)
        mean_y = refiner.box_filter(base_y)
        cov_xy = refiner.box_filter(base_x * base_y) - mean_x * mean_y
        var_x = refiner.box_filter(base_x * base_x) - mean_x * mean_x
        A = refiner.conv(torch.cat([cov_xy, var_x, hid], dim=1))
        b = mean_y - A * mean_x

        # Channel 3 holds the alpha coefficients, guided by mean luminance.
        full_size = src.shape[2:]
        a_pha = F.interpolate(A[:, 3:4], size=full_size,
                              mode="bilinear", align_corners=False)
        b_pha = F.interpolate(b[:, 3:4], size=full_size,
                              mode="bilinear", align_corners=False)
        pha = a_pha * src.mean(1, keepdim=True) + b_pha
        pha = pha.clamp(0.0, 1.0) * 255.0
        return pha, r1o, r2o, r3o, r4o


def state_shapes(model, base_h, base_w):
    """Runs one frame at the base size to read the recurrent state shapes."""
    with torch.no_grad():
        src = torch.zeros(1, 3, base_h, base_w)
        f1, f2, f3, f4 = model.backbone(src)
        f4 = model.aspp(f4)
        _, *rec = model.decoder(src, f1, f2, f3, f4, None, None, None, None)
    return [tuple(r.shape) for r in rec]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("rvm_repo")
    parser.add_argument("weights")
    parser.add_argument("output")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--downsample", type=float, default=0.25)
    parser.add_argument(
        "--variant", choices=["mobilenetv3", "resnet50"], default="mobilenetv3")
    args = parser.parse_args()

    sys.path.insert(0, args.rvm_repo)
    from model import MattingNetwork

    model = MattingNetwork(args.variant).eval()
    model.load_state_dict(torch.load(
        args.weights, map_location="cpu", weights_only=True))

    base_h = round(args.height * args.downsample)
    base_w = round(args.width * args.downsample)
    shapes = state_shapes(model, base_h, base_w)
    print("Base size", (base_h, base_w), "state shapes", shapes)

    wrapper = AlphaOnlyRVM(model, (base_h, base_w)).eval()
    example = [torch.rand(1, 3, args.height, args.width)] + \
        [torch.zeros(s) for s in shapes]
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, example)

    inputs = [
        ct.ImageType(
            name="src",
            shape=(1, 3, args.height, args.width),
            scale=1.0 / 255.0,
            color_layout=ct.colorlayout.RGB,
        )
    ] + [ct.TensorType(name=f"r{i}i", shape=s, dtype=np.float16) for i, s in enumerate(shapes, 1)]
    outputs = [ct.ImageType(name="pha", color_layout=ct.colorlayout.GRAYSCALE)] + [
        ct.TensorType(name=f"r{i}o", dtype=np.float16) for i in range(1, 5)
    ]

    mlmodel = ct.convert(
        traced,
        inputs=inputs,
        outputs=outputs,
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS13,
    )
    mlmodel.short_description = (
        f"Robust Video Matting ({args.variant}), alpha only, {args.width}x{args.height}, "
        f"downsample {args.downsample}"
    )
    mlmodel.license = "GPL-3.0"
    mlmodel.save(args.output)
    print("Saved", args.output)


if __name__ == "__main__":
    main()
