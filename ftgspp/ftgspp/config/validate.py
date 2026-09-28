# Modified by Team Doreen071 for the SIGGRAPH Asia 2026 VVC Sparse-View track (2026-09); see MODIFICATIONS.md.
from pathlib import Path

from ftgspp.config.schema import AppConfig, DEFAULT_FPS


MODEL_TYPES = {"explicit", "field"}
ROMA_MODELS = {"indoor", "outdoor"}
ROMA_BACKENDS = {"v1", "v2"}
RELOCATION_MODES = {"partial_copy", "exact_copy", "3d_mcmc"}
SCORE_MODES = {"default", "gate_included"}
PIPELINE_STAGES = {"points", "init", "train", "export"}


def compute_derived_fields(config: AppConfig) -> AppConfig:
    fps = config.data.fps if config.data.fps > 0 else DEFAULT_FPS
    config.data.fps = fps
    num_frames = max(config.data.frames.stop - config.data.frames.start, 1)
    config.model.max_duration = num_frames / fps
    config.init.duration = max(config.init.keyframe_stride, 1) / fps
    return config


def validate_static_config(config: AppConfig) -> None:
    if config.data.frames.stop <= config.data.frames.start:
        raise ValueError("data.frames.stop must be greater than data.frames.start")
    if config.init.keyframe_stride <= 0:
        raise ValueError("init.keyframe_stride must be greater than zero")
    if config.init.num_gaussians <= 0:
        raise ValueError("init.num_gaussians must be greater than zero")
    if config.init.num_points_per_frame <= 0:
        raise ValueError("init.num_points_per_frame must be greater than zero")
    if config.train.iterations <= 0:
        raise ValueError("train.iterations must be greater than zero")
    if config.train.batch_size != 1:
        raise ValueError("FTGSPP MVDataset requires train.batch_size = 1")
    if config.train.num_workers < 0:
        raise ValueError("train.num_workers must not be negative")
    if config.train.prefetch_factor <= 0:
        raise ValueError("train.prefetch_factor must be greater than zero")
    if config.eval.frame_interval <= 0:
        raise ValueError("eval.frame_interval must be greater than zero")
    if config.eval.num_workers < 0:
        raise ValueError("eval.num_workers must not be negative")
    if config.eval.prefetch_factor <= 0:
        raise ValueError("eval.prefetch_factor must be greater than zero")
    if config.train.relocation.stop > config.train.iterations:
        raise ValueError("train.relocation.stop must not exceed train.iterations")
    if config.model.velocity_model not in MODEL_TYPES:
        raise ValueError(f"unsupported model.velocity_model: {config.model.velocity_model}")
    if config.init.roma.model not in ROMA_MODELS:
        raise ValueError(f"unsupported init.roma.model: {config.init.roma.model}")
    if config.init.roma.backend not in ROMA_BACKENDS:
        raise ValueError(f"unsupported init.roma.backend: {config.init.roma.backend}")
    if not 0 <= config.init.roma.threshold <= 1:
        raise ValueError("init.roma.threshold must be between zero and one")
    if config.init.roma.num_workers < 0:
        raise ValueError("init.roma.num_workers must not be negative")
    if config.init.roma.prefetch_factor <= 0:
        raise ValueError("init.roma.prefetch_factor must be greater than zero")
    if config.train.relocation.mode not in RELOCATION_MODES:
        raise ValueError(
            f"unsupported train.relocation.mode: {config.train.relocation.mode}"
        )
    if config.train.relocation.score_mode not in SCORE_MODES:
        raise ValueError(
            "unsupported train.relocation.score_mode: "
            f"{config.train.relocation.score_mode}"
        )
    unknown_stages = sorted(set(config.pipeline.stages) - PIPELINE_STAGES)
    if unknown_stages:
        raise ValueError(f"unsupported pipeline stages: {unknown_stages}")
    if not config.run.export_filename or Path(config.run.export_filename).name != config.run.export_filename:
        raise ValueError("run.export_filename must be a filename, not a path")


def validate_resolved_config(config: AppConfig, available_cameras: list[int]) -> None:
    train = config.data.train_cameras
    eval_ = config.data.eval_cameras
    if not train:
        raise ValueError("resolved config contains no training cameras")


    overlap = sorted(set(train) & set(eval_))
    if overlap:
        raise ValueError(f"training and evaluation cameras overlap: {overlap}")
    available = set(available_cameras)
    unknown = sorted((set(train) | set(eval_)) - available)
    if unknown:
        raise ValueError(f"resolved cameras are missing from images: {unknown}")
