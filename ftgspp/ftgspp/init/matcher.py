# FreeTimeGS++
# 2025-2026 Lucas Yunkyu Lee <lucaslee@postech.ac.kr>, SNU VGI Lab

from pathlib import Path
from typing import Any, Literal, Optional, Protocol

import torch
from PIL import Image
from torch import Tensor

from ftgspp.utils import PathLike


class RoMaConfigLike(Protocol):
    backend: Literal["v1", "v2"]
    model: Literal["indoor", "outdoor"]
    upsample_preds: bool
    symmetric: bool
    threshold: float


class RoMaMatcher(Protocol):
    def match_points(
        self,
        img_a: Image.Image,
        img_b: Image.Image,
        debug_path: Optional[PathLike] = None,
    ) -> tuple[Tensor, Tensor]: ...


class RoMaV1Matcher:
    _model: Any

    def __init__(
        self,
        *,
        device: str,
        model: Literal["indoor", "outdoor"],
        upsample_preds: bool,
        symmetric: bool,
        threshold: float,
    ):
        import romatch

        factories = {"indoor": romatch.roma_indoor, "outdoor": romatch.roma_outdoor}
        self._model = factories[model](device)
        self._model.symmetric = symmetric
        self._model.upsample_preds = upsample_preds
        self._model.sample_mode = "threshold"
        self._model.sample_thresh = threshold
        self._threshold = threshold

    def match_points(
        self,
        img_a: Image.Image,
        img_b: Image.Image,
        debug_path: Optional[PathLike] = None,
    ) -> tuple[Tensor, Tensor]:
        warp, certainty = self._model.match(img_a, img_b)
        mask = certainty.reshape(-1) > self._threshold
        matches = warp.reshape(-1, 4)[mask]

        if debug_path is not None:
            self._model.visualize_warp(
                warp,
                certainty,
                img_a,
                img_b,
                symmetric=self._model.symmetric,
                save_path=Path(debug_path),
            )

        pts_a, pts_b = self._model.to_pixel_coordinates(
            matches, img_a.height, img_a.width, img_b.height, img_b.width
        )
        return pts_a, pts_b


class RoMaV2Matcher:
    _model: Any

    def __init__(
        self,
        *,
        device: str,
        model: Literal["indoor", "outdoor"],
        upsample_preds: bool,
        symmetric: bool,
        threshold: float,
    ):
        try:
            from romav2 import RoMaV2
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "RoMa v2 backend requires `romav2`. Run `bash scripts/install_env.sh` "
                "or switch `init.roma.backend` to `v1`."
            ) from e

        del model, upsample_preds, symmetric
        self._model = RoMaV2().to(device).eval()
        # Keep raw overlap probabilities and only compute the requested A->B
        # direction. Thresholding is applied explicitly below.
        self._model.threshold = None
        self._model.bidirectional = False
        self._threshold = threshold

    def match_points(
        self,
        img_a: Image.Image,
        img_b: Image.Image,
        debug_path: Optional[PathLike] = None,
    ) -> tuple[Tensor, Tensor]:
        del debug_path

        preds = self._model.match(img_a, img_b)
        warp_ab = preds["warp_AB"][0]
        overlaps = preds["overlap_AB"][0].reshape(-1)
        height, width = warp_ab.shape[:2]
        ys, xs = torch.meshgrid(
            torch.linspace(
                -1 + 1 / height,
                1 - 1 / height,
                height,
                device=warp_ab.device,
            ),
            torch.linspace(
                -1 + 1 / width,
                1 - 1 / width,
                width,
                device=warp_ab.device,
            ),
            indexing="ij",
        )
        source_grid = torch.stack((xs, ys), dim=-1)
        matches = torch.cat((source_grid, warp_ab), dim=-1).reshape(-1, 4)
        matches = matches[overlaps > self._threshold]
        pts_a, pts_b = self._model.to_pixel_coordinates(
            matches, img_a.height, img_a.width, img_b.height, img_b.width
        )

        return pts_a, pts_b


def create_roma_matcher(
    config: RoMaConfigLike,
    *,
    device: str,
) -> RoMaMatcher:
    match config.backend:
        case "v1":
            return RoMaV1Matcher(
                device=device,
                model=config.model,
                upsample_preds=config.upsample_preds,
                symmetric=config.symmetric,
                threshold=config.threshold,
            )
        case "v2":
            return RoMaV2Matcher(
                device=device,
                model=config.model,
                upsample_preds=config.upsample_preds,
                symmetric=config.symmetric,
                threshold=config.threshold,
            )

    raise ValueError(f"Unsupported RoMa backend: {config.backend}")
