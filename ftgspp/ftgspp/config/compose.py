from __future__ import annotations

from os import PathLike as OSPathLike
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from ftgspp.config.paths import resolve_paths
from ftgspp.config.register import register_resolvers
from ftgspp.config.schema import AppConfig
from ftgspp.config.validate import compute_derived_fields, validate_static_config
type PathLike = str | OSPathLike[str]


def _yaml_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _yaml_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_yaml_safe(item) for item in value]
    return value


def from_hydra(config: DictConfig) -> AppConfig:
    # Keep resolved configs from older runs loadable after removing obsolete
    # preprocessing and RoMa initialization options.
    config = OmegaConf.create(OmegaConf.to_container(config, resolve=False))
    data = config.get("data")
    if isinstance(data, DictConfig):
        data.pop("memmap_path", None)
        data.pop("colmap_path", None)
    init = config.get("init")
    if isinstance(init, DictConfig):
        roma = init.get("roma")
        if isinstance(roma, DictConfig):
            roma.pop("num_nearest_cameras", None)
            roma.pop("geometric_verification", None)

    structured = OmegaConf.structured(AppConfig)
    merged = OmegaConf.merge(structured, config)
    OmegaConf.resolve(merged)
    result = OmegaConf.to_object(merged)
    if not isinstance(result, AppConfig):
        raise TypeError(f"expected AppConfig, got {type(result)!r}")
    resolve_paths(result)
    compute_derived_fields(result)
    validate_static_config(result)
    return result


def load_config(path: PathLike) -> AppConfig:
    register_resolvers()
    path = Path(path).expanduser().resolve()
    return from_hydra(OmegaConf.load(path))


def save_composed_config(config: DictConfig, path: PathLike) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plain = _yaml_safe(OmegaConf.to_container(config, resolve=False, enum_to_str=True))
    OmegaConf.save(config=OmegaConf.create(plain), f=path, resolve=False)


def save_resolved_config(config: AppConfig, path: PathLike) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    node = OmegaConf.structured(config)
    plain = _yaml_safe(OmegaConf.to_container(node, resolve=True, enum_to_str=True))
    OmegaConf.save(config=OmegaConf.create(plain), f=path, resolve=True)


def load_run_config(run_path: PathLike) -> AppConfig:
    run_path = Path(run_path)
    resolved = run_path / "config.resolved.yaml"
    if not resolved.exists():
        raise FileNotFoundError(f"missing config.resolved.yaml in {run_path}")
    return load_config(resolved)
