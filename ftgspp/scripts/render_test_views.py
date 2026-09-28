#!/usr/bin/env python3
"""Render dataset test views with their calibrated intrinsics and extrinsics."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from ftgspp.config import load_run_config
from ftgspp.data.utils import (
    load_multiview_dataset,
    make_mvdataset_eval_loader,
)
from ftgspp.models.gaussians import Gaussians
from ftgspp.utils.sys import tensor_to_png


@torch.inference_mode()
def render_test_views(
    *,
    run_path: Path,
    output_path: Path,
    filename: str = "gaussians.pt",
    overwrite: bool = False,
    frame_interval: int | None = None,
    num_workers: int | None = None,
    prefetch_factor: int | None = None,
    pin_memory: bool | None = None,
) -> int:
    config = load_run_config(run_path)
    dataset = load_multiview_dataset(config)
    test_cameras = dataset.eval_cameras
    test_set = dataset.at(config.data.frames.into_slice(), test_cameras)
    frame_interval = (
        config.eval.frame_interval if frame_interval is None else frame_interval
    )
    num_workers = config.eval.num_workers if num_workers is None else num_workers
    prefetch_factor = (
        config.eval.prefetch_factor if prefetch_factor is None else prefetch_factor
    )
    pin_memory = config.eval.pin_memory if pin_memory is None else pin_memory

    views = make_mvdataset_eval_loader(
        test_set,
        frame_interval=frame_interval,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
    )
    total = len(views)

    gaussians = Gaussians.load(run_path / filename).cuda().eval()
    output_path.mkdir(parents=True, exist_ok=True)

    rendered = 0
    skipped = 0
    for view in tqdm(views, total=total, desc="render test views"):
        camera = int(view.camera[0])
        camera_name = dataset.camera_name(camera)
        source_frame = Path(view.path[0]).stem
        output = output_path / camera_name / f"{source_frame}.png"
        if output.exists() and not overwrite:
            skipped += 1
            continue

        output.parent.mkdir(parents=True, exist_ok=True)
        view = view.cuda(non_blocking=pin_memory)
        image, _, _ = gaussians(
            t=view.time[0],
            w2c=view.w2c,
            intrinsic=view.intrinsic,
            shape=(view.height, view.width),
        )
        output.write_bytes(tensor_to_png(image.squeeze(0)))
        rendered += 1

    print(
        f"rendered={rendered} skipped={skipped} total={total} "
        f"frame_interval={frame_interval} cameras={list(test_cameras)} "
        f"output={output_path.resolve()}"
    )
    return rendered


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render all resolved test cameras using dataset calibration and timestamps."
        )
    )
    parser.add_argument("run_path", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--filename", "-f", default="gaussians.pt")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--frame-interval", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--prefetch-factor", type=int)
    parser.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    return parser


def main() -> None:
    args = make_parser().parse_args()
    render_test_views(
        run_path=args.run_path,
        output_path=args.output_path,
        filename=args.filename,
        overwrite=args.overwrite,
        frame_interval=args.frame_interval,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        pin_memory=args.pin_memory,
    )


if __name__ == "__main__":
    main()
