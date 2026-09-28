from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from omegaconf import DictConfig

from ftgspp.config import from_hydra, save_composed_config, save_resolved_config
from ftgspp.config.schema import AppConfig
from ftgspp.config.validate import validate_resolved_config


logger = logging.getLogger("ftgspp.runner")


@dataclass
class Runner:
    config: AppConfig
    composed: DictConfig

    @classmethod
    def from_hydra(cls, config: DictConfig) -> "Runner":
        return cls(config=from_hydra(config), composed=config)

    @property
    def output_path(self) -> Path:
        return self.config.run.output_path

    def _save_composed(self) -> None:
        save_composed_config(
            self.composed,
            self.output_path / "config.composed.yaml",
        )

    def _save_resolved(self) -> None:
        save_resolved_config(
            self.config,
            self.output_path / "config.resolved.yaml",
        )

    def _preflight(self):
        from ftgspp.data.utils import load_multiview_dataset

        dataset = load_multiview_dataset(self.config)
        train_cameras = dataset.train_cameras
        eval_cameras = dataset.eval_cameras
        self.config.data.train_cameras = list(train_cameras)
        self.config.data.eval_cameras = list(eval_cameras)
        self.config.data.camera_split_source = dataset.camera_split_source

        available_cameras = [int(camera) for camera in dataset[0].camera]
        validate_resolved_config(self.config, available_cameras)

        first_train = self.config.data.train_cameras[0]
        sample = dataset.at(
            self.config.data.frames.into_slice(), [first_train]
        )[0, 0]
        shape = tuple(sample.rgb.shape)

        logger.info(
            "Dataset check: frames=%d cameras=%d fps=%.6f first_train_image=%s shape=%s",
            dataset.num_frames,
            dataset.num_cameras,
            dataset.fps,
            sample.path,
            shape,
        )
        logger.info(
            "Camera split from %s: train=%s eval=%s",
            self.config.data.camera_split_source,
            self.config.data.train_cameras,
            self.config.data.eval_cameras,
        )
        return dataset

    def _keyframe_points_complete(self, num_frames: int) -> bool:
        start = self.config.data.frames.start
        stride = self.config.init.keyframe_stride
        offsets = list(range(0, num_frames, stride))
        if offsets[-1] != num_frames - 1:
            offsets.append(num_frames - 1)
        return all(
            (self.config.init.points_path / f"f{start + offset:06d}.ply").exists()
            for offset in offsets
        )

    def _run_stage(self, stage: str, dataset) -> None:
        if stage == "points":
            if self._keyframe_points_complete(dataset.num_frames):
                logger.info("Keyframe points already exist: %s", self.config.init.points_path)
                return
            from ftgspp.init.points import generate_points

            generate_points(self.config)
            return

        if stage == "init":
            init_path = self.output_path / "init.pt"
            if init_path.exists():
                logger.info("Initialization already exists: %s", init_path)
                return
            from ftgspp.init.__main__ import initialize_gaussians

            initialize_gaussians(self.config, self.output_path)
            return

        if stage == "train":
            from ftgspp.train import train_gaussians

            train_gaussians(self.config, self.output_path, stages=["train"])
            return

        if stage == "export":
            checkpoint = self.output_path / "gaussians.pt"
            if not checkpoint.exists():
                raise FileNotFoundError(f"missing trained checkpoint: {checkpoint}")
            from ftgspp.export import export_gaussians

            output = self.output_path / self.config.run.export_filename
            export_gaussians(checkpoint, output)
            if not output.is_file() or output.stat().st_size == 0:
                raise RuntimeError(f"exported PLY is missing or empty: {output}")
            return

        raise ValueError(f"unsupported pipeline stage: {stage}")

    def run(self) -> None:
        self.output_path.mkdir(parents=True, exist_ok=True)
        self._save_composed()

        dataset = self._preflight()
        self._save_resolved()

        if self.config.pipeline.check_only:
            logger.info("Config and dataset check passed; no training stages were run.")
            return

        self.config.init.points_path.mkdir(parents=True, exist_ok=True)
        self.config.run.log_dir.mkdir(parents=True, exist_ok=True)

        completed = self.output_path / "gaussians.pt"
        if completed.exists() and not self.config.run.overwrite_completed:
            raise FileExistsError(
                f"refusing to overwrite completed result: {completed}; "
                "set run.overwrite_completed=true to override"
            )

        for stage in self.config.pipeline.stages:
            logger.info("Running pipeline stage: %s", stage)
            self._run_stage(stage, dataset)

        self._save_resolved()
        logger.info("FTGSPP pipeline completed: %s", self.output_path)
