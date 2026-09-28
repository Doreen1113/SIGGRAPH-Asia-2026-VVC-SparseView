from pathlib import Path


def export_gaussians(
    checkpoint: Path,
    output: Path | None = None,
    device: str = "cpu",
) -> Path:
    # Keep the challenge exporter script as the compatibility CLI while making
    # its implementation reusable by the Hydra runner.
    from scripts.export_ftgspp_4dgs_ply import export_gaussians as _export

    return _export(checkpoint=checkpoint, output=output, device=device)


__all__ = ["export_gaussians"]
