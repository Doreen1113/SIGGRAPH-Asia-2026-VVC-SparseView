import os

import hydra
from omegaconf import DictConfig

from ftgspp.config.register import register_configs


register_configs()


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(config: DictConfig) -> None:
    # CUDA-sensitive modules are imported only after the selected GPU is set.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.run.gpu)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")

    from ftgspp.runner import Runner

    Runner.from_hydra(config).run()


if __name__ == "__main__":
    main()
