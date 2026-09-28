# Modified by Team Doreen071 for the SIGGRAPH Asia 2026 VVC Sparse-View track (2026-09); see MODIFICATIONS.md.
# FreeTimeGS++
# 2025-2026 Lucas Yunkyu Lee <lucaslee@postech.ac.kr>, SNU VGI Lab

import logging
import os
import random
from pathlib import Path
from typing import Optional, cast

import numpy as np
import torch

from ftgspp.data.utils import load_multiview_dataset
from ftgspp.models import (
    Bounds,
    ColorCorrectors,
    ExplicitVelocities,
    Gaussians,
    VelocityField,
)
from ftgspp.train.train import train
from ftgspp.utils import Config, PathLike, Pipeline


class Trainer(Pipeline):
    stages_to_run: list[str]
    config: Config
    run_path: Path
    logger: logging.Logger

    @staticmethod
    def _close_logger_handlers(logger: logging.Logger):
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()

    @staticmethod
    def _seed_train_rngs(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def _resolve_camera_splits(self, dataset):
        cfg = self.config
        train_cameras = dataset.train_cameras
        eval_cameras = dataset.eval_cameras
        cfg.data.train_cameras = train_cameras
        cfg.data.eval_cameras = eval_cameras
        cfg.data.camera_split_source = dataset.camera_split_source
        cfg.save(self.run_path / "config.resolved.yaml")
        self.logger.info(
            "Camera split from %s: train=%s eval=%s",
            dataset.camera_split_source,
            train_cameras,
            eval_cameras,
        )
        return train_cameras, eval_cameras

    def __init__(
        self,
        config: Config,
        run_path: PathLike,
        stages_to_run: Optional[list[str]] = None,
    ):
        self.config = config
        self.run_path = Path(run_path)
        self.run_path.mkdir(exist_ok=True, parents=True)

        if stages_to_run is None:
            stages_to_run = list(self.stages.keys())
        self.stages_to_run = stages_to_run

        self.logger = logging.getLogger("ftgspp.train")
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        self._close_logger_handlers(self.logger)
        fh = logging.FileHandler(self.run_path / "train.log", mode="w")
        fh.setLevel(logging.DEBUG)
        self.logger.addHandler(fh)

    @Pipeline.stage("train")
    def train(self):
        self.logger.info("Starting training")
        cfg = self.config
        self._seed_train_rngs(cfg.train.seed)
        self.logger.info("Using train seed %d", cfg.train.seed)

        dataset = load_multiview_dataset(cfg, logger=self.logger)
        train_cameras, eval_cameras = self._resolve_camera_splits(dataset)
        gs = Gaussians.load(self.run_path / "init.pt")
        velocity_distill = None
        velocity_distill_warmup_steps = 0
        velocity_distill_weight = 0.0
        velocity_distill_batch_size = 0
        if cfg.model.velocity_model == "field":
            if isinstance(gs.velocity_model, ExplicitVelocities):
                velocity_distill = (
                    gs.means.detach().clone(),
                    gs.times.detach().clone(),
                    gs.velocity_model.detach().clone(),
                )
                velocity_distill_warmup_steps = min(3000, cfg.train.iterations)
                velocity_distill_weight = 0.05
                velocity_distill_batch_size = 32768
                self.logger.info(
                    "Warm-start velocity distillation enabled "
                    "(steps=%d, weight=%.3f, batch=%d).",
                    velocity_distill_warmup_steps,
                    velocity_distill_weight,
                    velocity_distill_batch_size,
                )
            else:
                self.logger.warning(
                    "Warm-start velocity distillation skipped: "
                    "init velocity model is not explicit."
                )
            minmax = gs.means.aminmax(dim=0)
            bounds_xyz = Bounds(torch.stack([minmax.min, minmax.max], dim=-1))
            bounds_t = Bounds((0, dataset.duration))
            del gs.velocity_model
            gs.velocity_model = VelocityField(bounds_xyz=bounds_xyz, bounds_t=bounds_t)
        gs.cuda().train()

        if not cfg.model.marginal_gating:
            with torch.no_grad():
                gs.marginal_gates.fill_(-100)
            gs.marginal_gates.requires_grad = False

        if cfg.train.color_correction:
            color_correctors = ColorCorrectors(max(int(c) for c in list(dataset.train_cameras) + list(dataset.eval_cameras)) + 1)  # patched: index by camera id
            color_correctors.cuda().train()
        else:
            color_correctors = None

        color_correction_start = (
            cfg.train.color_correction_start
            if cfg.train.color_correction
            else cfg.train.iterations
        )
        gate_weight = 0.001 if cfg.model.marginal_gating else 0.0
        lpips_weight = float(os.environ.get("FTGSPP_LPIPS_W", "0.01")) if cfg.train.lpips_loss else 0.0
        color_corrector_weight = (
            cfg.train.color_corrector_weight if cfg.train.color_correction else 0.0
        )

        gs = train(
            gs=gs,
            color_correctors=color_correctors,
            dataset=dataset,
            batch_size=cfg.train.batch_size,
            iterations=cfg.train.iterations,
            num_workers=cfg.train.num_workers,
            prefetch_factor=cfg.train.prefetch_factor,
            persistent_workers=cfg.train.persistent_workers,
            pin_memory=cfg.train.pin_memory,
            seed=cfg.train.seed,
            eval_frame_interval=cfg.eval.frame_interval,
            eval_num_workers=cfg.eval.num_workers,
            eval_prefetch_factor=cfg.eval.prefetch_factor,
            eval_pin_memory=cfg.eval.pin_memory,
            frames=cfg.data.frames.into_slice(),
            train_cameras=train_cameras,
            eval_cameras=eval_cameras,
            relocation=True,
            relocation_start=cfg.train.relocation.start,
            relocation_every=cfg.train.relocation.every,
            relocation_stop=cfg.train.relocation.stop,
            relocation_opacity_threshold=cfg.train.relocation.opacity_threshold,
            relocation_mode=cfg.train.relocation.mode,
            relocation_score_mode=cfg.train.relocation.score_mode,
            relocation_score_mode_start=cfg.train.relocation.score_mode_start,
            color_correction_start=color_correction_start,
            sh_degree_schedule=True,
            loss_weights={
                "l1": float(os.environ.get("FTGSPP_L1_W", "0.8")),
                "ssim": float(os.environ.get("FTGSPP_SSIM_W", "0.2")),
                "lpips": lpips_weight,
                "reg_opacity": float(os.environ.get("FTGSPP_REG_OPACITY", "0.01")),
                "reg_scale": float(os.environ.get("FTGSPP_REG_SCALE", "0.01")),
                "reg_duration": float(os.environ.get("FTGSPP_REG_DURATION", "0.0")),
                "gate": gate_weight,
                "color_correctors": color_corrector_weight,
            },
            run_path=self.run_path,
            lrs=cfg.train.lrs,
            lr_schedules=cfg.train.lr_schedules,
            checkpoint_iterations=list(range(1000, cfg.train.iterations, int(os.environ.get("FTGSPP_CHECKPOINT_INTERVAL", "1000")))) if int(os.environ.get("FTGSPP_CHECKPOINT_INTERVAL", "1000")) > 0 else [],
            logger=self.logger,
            velocity_distill=velocity_distill,
            velocity_distill_warmup_steps=velocity_distill_warmup_steps,
            velocity_distill_weight=velocity_distill_weight,
            velocity_distill_batch_size=velocity_distill_batch_size,
        )

        gs.save(self.run_path / "gaussians.pt")
        if color_correctors is not None:
            torch.save(color_correctors, self.run_path / "color_correctors.pt")

        self.logger.info("Done training")

    @Pipeline.stage("separate")
    def separate(self):
        cfg = self.config

        if not cfg.train.hard_separation:
            self.logger.info("Skipping hard separation stage")
            return

        self.logger.info("Starting hard separation")

        dataset = load_multiview_dataset(cfg, logger=self.logger)
        train_cameras, eval_cameras = self._resolve_camera_splits(dataset)

        gs = Gaussians.load(self.run_path / "gaussians.pt")
        gs.cuda().train()
        color_correctors: ColorCorrectors | None = None
        if (
            cfg.train.color_correction
            and (self.run_path / "color_correctors.pt").exists()
        ):
            color_correctors = cast(
                ColorCorrectors,
                torch.load(self.run_path / "color_correctors.pt", weights_only=False),
            )
            color_correctors.cuda().eval()

        use_color_corrector = color_correctors is not None
        separate_iterations = 5000
        color_correction_start = 0 if use_color_corrector else separate_iterations
        lpips_weight = float(os.environ.get("FTGSPP_LPIPS_W", "0.01")) if cfg.train.lpips_loss else 0.0

        with torch.no_grad():
            threshold = 0.5
            marginalize = gs.gate().squeeze(-1) > threshold
            n_marginalize = int(torch.count_nonzero(marginalize))
            self.logger.info(f"Marginalizing {n_marginalize} Gaussians")
            gs.marginal_gates[marginalize] = 100
            gs.marginal_gates[~marginalize] = -100
            gs.marginal_gates.requires_grad = False

            if isinstance(gs.velocity_model, ExplicitVelocities):
                gs.velocity_model[marginalize] = 0
                gs.velocity_model.requires_grad = False

        gs = train(
            gs=gs,
            color_correctors=color_correctors,
            dataset=dataset,
            batch_size=1,
            iterations=separate_iterations,
            num_workers=cfg.train.num_workers,
            prefetch_factor=cfg.train.prefetch_factor,
            persistent_workers=cfg.train.persistent_workers,
            pin_memory=cfg.train.pin_memory,
            seed=cfg.train.seed,
            eval_frame_interval=cfg.eval.frame_interval,
            eval_num_workers=cfg.eval.num_workers,
            eval_prefetch_factor=cfg.eval.prefetch_factor,
            eval_pin_memory=cfg.eval.pin_memory,
            frames=cfg.data.frames.into_slice(),
            train_cameras=train_cameras,
            eval_cameras=eval_cameras,
            relocation=True,
            relocation_start=500,
            relocation_every=100,
            relocation_stop=3000,
            relocation_opacity_threshold=cfg.train.relocation.opacity_threshold,
            relocation_mode=cfg.train.relocation.mode,
            relocation_score_mode_start=-1,
            relocation_score_mode=cfg.train.relocation.score_mode,
            color_correction_start=color_correction_start,
            sh_degree_schedule=False,
            loss_weights={
                "l1": float(os.environ.get("FTGSPP_L1_W", "0.8")),
                "ssim": float(os.environ.get("FTGSPP_SSIM_W", "0.2")),
                "lpips": lpips_weight,
                "reg_opacity": 0.001,
                "reg_scale": 0.001,
                "reg_duration": 0.0,
                "gate": 0,
                "color_correctors": 0,
            },
            run_path=self.run_path,
            lrs={
                "means": 0.001 * 1.6e-4,
                "quats": 1e-3,
                "scales": 5e-3,
                "opacities": 5e-2,
                "sh_0": 2.5e-3,
                "sh_n": 1.25e-4,
                "durations": 5e-3,
                "times": 1.6e-4,
            },
            lr_schedules={},
            checkpoint_iterations=list(range(cfg.train.iterations + 1000, cfg.train.iterations + separate_iterations, int(os.environ.get("FTGSPP_CHECKPOINT_INTERVAL", "1000")))) if int(os.environ.get("FTGSPP_CHECKPOINT_INTERVAL", "1000")) > 0 else [],
            logger=self.logger,
            start_iteration=cfg.train.iterations,
        )
        gs.save(self.run_path / "separated.pt")

        self.logger.info("Done separating")

    def __call__(self):
        """Main training routine."""
        try:
            self.config.save(self.run_path / "config.resolved.yaml")

            for stage_name in self.stages_to_run:
                self.logger.info(f"Running stage: {stage_name}")
                stage = self.stages[stage_name]
                stage(self)
        finally:
            self._close_logger_handlers(self.logger)


def train_gaussians(
    config: Config,
    run_path: PathLike,
    stages: Optional[list[str]] = None,
) -> None:
    Trainer(config=config, run_path=run_path, stages_to_run=stages)()
