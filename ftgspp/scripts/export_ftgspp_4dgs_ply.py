#!/usr/bin/env python3
"""Export a FreeTimeGS++ Gaussians checkpoint to the 4DGS PLY format.

The target PLY layout is the one consumed by the challenge renderer:
x/y/z, opacity logit, SH coefficients, log-scales, quaternion, temporal
center/log-scale, and constant motion.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ftgspp.models.gaussians import Gaussians, TEMPORAL_SCALE_MULTIPLIER  # noqa: E402


def _tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().contiguous().numpy().astype(np.float32, copy=False)


def _load_gaussians(path: Path, map_location: str) -> Gaussians:
    obj = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(obj, Gaussians):
        raise TypeError(f"expected ftgspp.models.gaussians.Gaussians, got {type(obj)!r}")
    return obj


@torch.no_grad()
def _constant_motion(gs: Gaussians, device: torch.device) -> torch.Tensor:
    velocity_model = gs.velocity_model
    if isinstance(velocity_model, torch.Tensor):
        return velocity_model

    # A learned velocity field has no exact constant-motion slot in the target
    # PLY format. Export the velocity sampled at each Gaussian's temporal center.
    gs = gs.to(device)
    return gs.velocities_t(gs.times)


def _temporal_log_scale(gs: Gaussians) -> torch.Tensor:
    if gs.max_duration == float("inf"):
        return gs.durations
    sigma_t = gs.max_duration * TEMPORAL_SCALE_MULTIPLIER * torch.sigmoid(gs.durations)
    return torch.log(sigma_t.clamp_min(torch.finfo(sigma_t.dtype).tiny))


def _structured_vertex_array(gs: Gaussians, device: torch.device) -> np.ndarray:
    n = len(gs)
    sh_0 = gs.sh_0
    sh_n = gs.sh_n
    if sh_0.shape != (n, 1, 3):
        raise ValueError(f"unexpected sh_0 shape: {tuple(sh_0.shape)}")
    if sh_n.dim() != 3 or sh_n.shape[0] != n or sh_n.shape[2] != 3:
        raise ValueError(f"unexpected sh_n shape: {tuple(sh_n.shape)}")

    rest_flat = sh_n.permute(0, 2, 1).reshape(n, -1)
    motion = _constant_motion(gs, device)
    t_scale = _temporal_log_scale(gs)

    field_names: list[str] = [
        "x",
        "y",
        "z",
        "opacity",
        "f_dc_0",
        "f_dc_1",
        "f_dc_2",
    ]
    field_names += [f"f_rest_{idx}" for idx in range(rest_flat.shape[1])]
    field_names += [f"scale_{idx}" for idx in range(gs.scales.shape[1])]
    field_names += [f"rot_{idx}" for idx in range(gs.quats.shape[1])]
    field_names += ["t", "t_scale"]
    field_names += [f"motion_{idx}" for idx in range(motion.shape[1])]

    vertices = np.empty(n, dtype=[(name, "<f4") for name in field_names])

    means = _tensor_to_numpy(gs.means)
    vertices["x"] = means[:, 0]
    vertices["y"] = means[:, 1]
    vertices["z"] = means[:, 2]
    vertices["opacity"] = _tensor_to_numpy(gs.opacities).reshape(n)

    sh0_np = _tensor_to_numpy(sh_0[:, 0, :])
    vertices["f_dc_0"] = sh0_np[:, 0]
    vertices["f_dc_1"] = sh0_np[:, 1]
    vertices["f_dc_2"] = sh0_np[:, 2]

    rest_np = _tensor_to_numpy(rest_flat)
    for idx in range(rest_np.shape[1]):
        vertices[f"f_rest_{idx}"] = rest_np[:, idx]

    scales_np = _tensor_to_numpy(gs.scales)
    for idx in range(scales_np.shape[1]):
        vertices[f"scale_{idx}"] = scales_np[:, idx]

    quats_np = _tensor_to_numpy(gs.quats)
    for idx in range(quats_np.shape[1]):
        vertices[f"rot_{idx}"] = quats_np[:, idx]

    vertices["t"] = _tensor_to_numpy(gs.times).reshape(n)
    vertices["t_scale"] = _tensor_to_numpy(t_scale).reshape(n)

    motion_np = _tensor_to_numpy(motion)
    for idx in range(motion_np.shape[1]):
        vertices[f"motion_{idx}"] = motion_np[:, idx]

    return vertices


def write_binary_little_endian_ply(path: Path, vertices: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header_lines = [
        "ply",
        "format binary_little_endian 1.0",
        "comment exported from FreeTimeGS++ Gaussians checkpoint",
        f"element vertex {len(vertices)}",
    ]
    header_lines += [f"property float {name}" for name in vertices.dtype.names or ()]
    header_lines.append("end_header")

    with path.open("wb") as f:
        f.write(("\n".join(header_lines) + "\n").encode("ascii"))
        vertices.tofile(f)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert FreeTimeGS++ gaussians.pt to challenge-compatible 4DGS PLY."
    )
    parser.add_argument("checkpoint", type=Path, help="Path to FreeTimeGS++ gaussians.pt")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output .ply path. Defaults to <checkpoint_dir>/gaussians_4dgs.ply",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for evaluating non-explicit velocity fields. Explicit models use CPU safely.",
    )
    return parser


def export_gaussians(
    checkpoint: Path,
    output: Path | None = None,
    device: str = "cpu",
) -> Path:
    checkpoint = checkpoint.expanduser().resolve()
    output = (
        output.expanduser().resolve()
        if output is not None
        else checkpoint.with_name("gaussians_4dgs.ply")
    )
    torch_device = torch.device(device)

    gs = _load_gaussians(checkpoint, map_location=str(torch_device))
    vertices = _structured_vertex_array(gs, device=torch_device)
    write_binary_little_endian_ply(output, vertices)

    sh_degree = int(np.sqrt(gs.sh_n.shape[1] + 1) - 1)
    print(f"wrote {output}")
    print(f"vertices={len(vertices)} sh_degree={sh_degree} properties={len(vertices.dtype.names or ())}")
    return output


def main() -> None:
    args = make_parser().parse_args()
    export_gaussians(args.checkpoint, args.output, args.device)


if __name__ == "__main__":
    main()
