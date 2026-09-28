# Modified by Team Doreen071 for the SIGGRAPH Asia 2026 VVC Sparse-View track (2026-09); see MODIFICATIONS.md.
# FreeTimeGS++
# 2025-2026 Lucas Yunkyu Lee <lucaslee@postech.ac.kr>, SNU VGI Lab

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ftgspp.data.utils import (
    MVDataset,
    load_multiview_dataset,
)
from ftgspp.init import keyframe_indices, to_point_filename
from ftgspp.init.matcher import RoMaMatcher, create_roma_matcher
from ftgspp.utils import Config
from ftgspp.utils.io import write_ply_points

# Helpers


@dataclass
class FrameCameraPair:
    reference: object
    target: object


def disjoint_nearest_camera_pairs(w2cs: Tensor) -> list[tuple[int, int]]:
    """Pair each camera at most once with its nearest unpaired camera."""

    c2ws = torch.inverse(w2cs)
    centers = c2ws[:, :3, 3]
    dists = torch.cdist(centers, centers)
    unpaired = list(range(len(w2cs)))
    pairs = []
    while len(unpaired) >= 2:
        reference = unpaired.pop(0)
        target_position = min(
            range(len(unpaired)),
            key=lambda position: float(dists[reference, unpaired[position]]),
        )
        target = unpaired.pop(target_position)
        pairs.append((reference, target))
    return pairs


class FrameCameraPairDataset(Dataset[FrameCameraPair]):
    """Load disjoint A->B camera pairs for one frame."""

    def __init__(self, frame_data: MVDataset):
        if len(frame_data) < 2:
            raise ValueError("RoMa point initialization requires at least two cameras")
        self.frame_data = frame_data
        import os as _os
        if _os.environ.get("FTGSPP_ALL_PAIRS", "0") == "1":
            n = len(frame_data.w2c); c2ws = torch.inverse(frame_data.w2c); centers = c2ws[:, :3, 3]
            maxd = float(_os.environ.get("FTGSPP_PAIR_MAXDIST", "1e9"))
            self.camera_pairs = [(i, j) for i in range(n) for j in range(i + 1, n) if float(torch.norm(centers[i] - centers[j])) <= maxd]
            print(f"[points] all-pairs init: {len(self.camera_pairs)} camera pairs")
        else:
            self.camera_pairs = disjoint_nearest_camera_pairs(frame_data.w2c)

    def __len__(self) -> int:
        return len(self.camera_pairs)

    def __getitem__(self, pair_index: int) -> FrameCameraPair:
        reference_index, target_index = self.camera_pairs[pair_index]
        reference = self.frame_data[reference_index].load()
        target = self.frame_data[target_index].load()
        return FrameCameraPair(reference=reference, target=target)


def _init_points_worker(_worker_id: int) -> None:
    cv2.setNumThreads(1)


def make_frame_pair_loader(
    frame_data: MVDataset,
    *,
    num_workers: int,
    prefetch_factor: int,
) -> DataLoader[FrameCameraPair]:
    worker_options = {}
    if num_workers > 0:
        worker_options = {
            "prefetch_factor": prefetch_factor,
            "worker_init_fn": _init_points_worker,
            "multiprocessing_context": "spawn",
        }

    return DataLoader(
        FrameCameraPairDataset(frame_data),
        batch_size=None,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        **worker_options,
    )


def tensor_to_pil(tensor: Tensor):
    assert tensor.dtype == torch.uint8
    assert tensor.ndim == 3 and tensor.shape[-1] == 3
    return Image.fromarray(tensor.detach().cpu().numpy())


# Pipeline


@dataclass
class PairCandidates:
    random_keys: Tensor
    points_a: Tensor
    points_b: Tensor
    reference_rgb: Tensor
    image_shape_a: tuple[int, int]
    image_shape_b: tuple[int, int]
    intrinsic_a: Tensor
    intrinsic_b: Tensor
    w2c_a: Tensor
    w2c_b: Tensor


def triangulate(
    uv,  # (cams, n, 2)
    projmat,  # (cams, 3, 4)
):
    # Multiple View Geometry (Hartley and Zisserman) Sec. 12.2
    p0, p1, p2 = torch.unbind(projmat, -2)
    x = torch.einsum("ij,ik->ijk", uv[..., 0], p2) - p0.unsqueeze(-2)
    y = torch.einsum("ij,ik->ijk", uv[..., 1], p2) - p1.unsqueeze(-2)
    eq = torch.cat([x, y]).transpose(0, 1)

    _, _, vh = torch.linalg.svd(eq, full_matrices=False)
    homogeneous = vh[..., -1, :]

    return homogeneous[..., :3] / homogeneous[..., 3:4]


def triangulate_matches(
    *,
    pts_0: Tensor,
    pts_1: Tensor,
    image_shape_0: tuple[int, int],
    image_shape_1: tuple[int, int],
    intrinsic_0: Tensor,
    intrinsic_1: Tensor,
    w2c_0: Tensor,
    w2c_1: Tensor,
) -> Tensor:
    projmat0 = intrinsic_0 @ w2c_0[:3]
    projmat1 = intrinsic_1 @ w2c_1[:3]

    normalizer_0 = torch.tensor(
        [
            [2 / image_shape_0[1], 0, -1],
            [0, 2 / image_shape_0[0], -1],
            [0, 0, 1],
        ],
        dtype=pts_0.dtype,
        device=pts_0.device,
    )
    normalizer_1 = torch.tensor(
        [
            [2 / image_shape_1[1], 0, -1],
            [0, 2 / image_shape_1[0], -1],
            [0, 0, 1],
        ],
        dtype=pts_1.dtype,
        device=pts_1.device,
    )

    return triangulate(
        torch.stack(
            [
                pts_0 @ normalizer_0[:2, :2] + normalizer_0[:2, 2],
                pts_1 @ normalizer_1[:2, :2] + normalizer_1[:2, 2],
            ]
        ),
        torch.stack([normalizer_0 @ projmat0, normalizer_1 @ projmat1]),
    )


def sample_reference_rgb(rgb: Tensor, points: Tensor) -> Tensor:
    height, width = rgb.shape[:2]
    x = points[:, 0].floor().long().clamp(0, width - 1)
    y = points[:, 1].floor().long().clamp(0, height - 1)
    return rgb[y, x]


def roma_points(
    matcher: RoMaMatcher,
    camera_pairs: Iterable[FrameCameraPair],
    num_points: int,
    threshold: float,
    seed: int,
) -> tuple[Tensor, Tensor]:
    candidates: list[PairCandidates] = []
    generator: torch.Generator | None = None

    for camera_pair in camera_pairs:
        view_a = camera_pair.reference
        view_b = camera_pair.target
        pts_a, pts_b = matcher.match_points(
            img_a=tensor_to_pil(view_a.rgb),  # type: ignore
            img_b=tensor_to_pil(view_b.rgb),  # type: ignore
        )
        if len(pts_a) == 0:
            continue

        if generator is None:
            generator = torch.Generator(device=pts_a.device).manual_seed(seed)
        random_keys = torch.rand(
            len(pts_a),
            generator=generator,
            device=pts_a.device,
        )
        local_count = min(num_points, len(pts_a))
        # A pair cannot contribute more than the final target count. Retaining
        # its lowest random keys is therefore lossless for the global top-k.
        local_indices = torch.topk(
            random_keys,
            k=local_count,
            largest=False,
            sorted=False,
        ).indices
        selected_a = pts_a[local_indices]
        selected_b = pts_b[local_indices]
        candidates.append(
            PairCandidates(
                random_keys=random_keys[local_indices],
                points_a=selected_a,
                points_b=selected_b,
                reference_rgb=view_a.rgb,  # type: ignore
                image_shape_a=tuple(view_a.rgb.shape[:2]),  # type: ignore
                image_shape_b=tuple(view_b.rgb.shape[:2]),  # type: ignore
                intrinsic_a=view_a.intrinsic.to(pts_a.device),  # type: ignore
                intrinsic_b=view_b.intrinsic.to(pts_a.device),  # type: ignore
                w2c_a=view_a.w2c.to(pts_a.device),  # type: ignore
                w2c_b=view_b.w2c.to(pts_a.device),  # type: ignore
            )
        )

    if not candidates:
        raise RuntimeError(
            f"no RoMa matches survived threshold={threshold}; lower init.roma.threshold"
        )

    all_keys = torch.cat([candidate.random_keys for candidate in candidates])
    selected = torch.topk(
        all_keys,
        k=min(num_points, len(all_keys)),
        largest=False,
        sorted=False,
    ).indices

    xyzs = []
    rgbs = []
    offset = 0
    for candidate in candidates:
        count = len(candidate.random_keys)
        local = selected[(selected >= offset) & (selected < offset + count)] - offset
        offset += count
        if len(local) == 0:
            continue

        xyz = triangulate_matches(
            pts_0=candidate.points_a[local],
            pts_1=candidate.points_b[local],
            image_shape_0=candidate.image_shape_a,
            image_shape_1=candidate.image_shape_b,
            intrinsic_0=candidate.intrinsic_a,
            intrinsic_1=candidate.intrinsic_b,
            w2c_0=candidate.w2c_a,
            w2c_1=candidate.w2c_b,
        )
        finite = torch.isfinite(xyz).all(dim=-1)
        selected_rgb = sample_reference_rgb(
            candidate.reference_rgb,
            candidate.points_a[local].cpu(),
        )
        xyzs.append(xyz[finite].cpu())
        rgbs.append(selected_rgb[finite.cpu()])

    if not xyzs:
        raise RuntimeError("all sampled RoMa triangulations were non-finite")
    return torch.cat(xyzs), torch.cat(rgbs)


def generate_points(config: Config) -> None:
    dataset = load_multiview_dataset(config)
    dataset = dataset.at(config.data.frames.into_slice(), dataset.train_cameras)

    out_path = config.init.points_path
    out_path.mkdir(exist_ok=True, parents=True)

    idxs = keyframe_indices(dataset.num_frames, config.init.keyframe_stride)
    matcher: RoMaMatcher | None = None
    for idx in (bar := tqdm(idxs)):
        frame_data = dataset[idx]
        frame = int(frame_data[0].frame)  # type: ignore
        point_path = out_path / to_point_filename(frame)
        if point_path.exists():
            bar.set_postfix({"status": "exists"})
            continue

        loader = make_frame_pair_loader(
            frame_data,
            num_workers=config.init.roma.num_workers,
            prefetch_factor=config.init.roma.prefetch_factor,
        )
        camera_pairs = iter(loader)
        if matcher is None:
            matcher = create_roma_matcher(config.init.roma, device="cuda")

        xyz, rgb = roma_points(
            matcher=matcher,
            camera_pairs=camera_pairs,
            num_points=config.init.num_points_per_frame,
            threshold=config.init.roma.threshold,
            seed=config.train.seed + frame,
        )

        write_ply_points(
            point_path,
            xyz=xyz.cpu().numpy(),
            rgb=rgb.cpu().numpy(),
        )
        bar.set_postfix({"n": str(len(xyz))})

    torch.cuda.empty_cache()


def main(args: argparse.Namespace):
    generate_points(Config.load(args.config))


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)

    return parser


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()

    main(args)
