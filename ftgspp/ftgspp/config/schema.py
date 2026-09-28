from __future__ import annotations

from dataclasses import dataclass, field
from os import PathLike as OSPathLike
from pathlib import Path
from typing import Self

from omegaconf import MISSING

type PathLike = str | OSPathLike[str]


@dataclass
class IntervalConfig:
    start: int = MISSING
    stop: int = MISSING

    def into_range(self) -> range:
        return range(self.start, self.stop)

    def into_slice(self) -> slice:
        return slice(self.start, self.stop)


DEFAULT_FPS = 60.0


@dataclass
class ModelConfig:
    velocity_model: str = MISSING
    marginal_gating: bool = False
    max_duration: float = 0.0


@dataclass
class DataConfig:
    name: str = MISSING
    fps: float = DEFAULT_FPS
    root: Path = MISSING
    video_path: Path = MISSING
    calibration_path: Path = MISSING
    extracted_path: Path = MISSING
    scale: float = MISSING
    frames: IntervalConfig = field(default_factory=IntervalConfig)
    train_cameras: list[int] | None = None
    eval_cameras: list[int] | None = None
    camera_split_source: str | None = None


@dataclass
class RoMaConfig:
    model: str = MISSING
    upsample_preds: bool = MISSING
    symmetric: bool = MISSING
    backend: str = "v2"
    threshold: float = 0.25
    num_workers: int = 4
    prefetch_factor: int = 1


@dataclass
class InitConfig:
    points_path: Path = MISSING
    keyframe_stride: int = MISSING
    num_points_per_frame: int = MISSING
    num_gaussians: int = MISSING
    sh_degree: int = MISSING
    scale: float = MISSING
    roma: RoMaConfig = field(default_factory=RoMaConfig)
    duration: float = 0.0
    opacity: float = 0.1
    num_velocity_nns: int = 0
    temporal_motion_adapted: bool = False
    temporal_flow_path: Path = Path("_flow")
    temporal_covis_thresh: float = 0.3


@dataclass
class RelocationConfig:
    start: int = MISSING
    stop: int = MISSING
    every: int = MISSING
    opacity_threshold: float = MISSING
    mode: str = "3d_mcmc"
    score_mode: str = "default"
    score_mode_start: int = 3000


@dataclass
class TrainConfig:
    iterations: int = MISSING
    batch_size: int = MISSING
    num_workers: int = 4
    prefetch_factor: int = 2
    persistent_workers: bool = True
    pin_memory: bool = True
    lrs: dict[str, float] = field(default_factory=dict)
    lr_schedules: dict[str, float] = field(default_factory=dict)
    relocation: RelocationConfig = field(default_factory=RelocationConfig)
    lpips_loss: bool = False
    color_correction: bool = False
    color_correction_start: int = 3000
    color_corrector_weight: float = 1e-3
    hard_separation: bool = False
    seed: int = 0


@dataclass
class EvalConfig:
    frame_interval: int = 10
    num_workers: int = 4
    prefetch_factor: int = 2
    pin_memory: bool = True


@dataclass
class RunConfig:
    gpu: int = 0
    experiment_root: Path = MISSING
    output_path: Path = MISSING
    log_dir: Path = MISSING
    export_filename: str = "gaussians_4dgs.ply"
    overwrite_completed: bool = False


@dataclass
class PipelineConfig:
    stages: list[str] = field(
        default_factory=lambda: ["points", "init", "train", "export"]
    )
    check_only: bool = False


@dataclass
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    init: InitConfig = field(default_factory=InitConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    run: RunConfig = field(default_factory=RunConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)

    @classmethod
    def load(cls, path: PathLike) -> Self:
        from ftgspp.config.compose import load_config

        return load_config(path)

    def save(self, path: PathLike) -> None:
        from ftgspp.config.compose import save_resolved_config

        save_resolved_config(self, path)


# Compatibility aliases used by the existing training core.
Config = AppConfig
Interval = IntervalConfig
