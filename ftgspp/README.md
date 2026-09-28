# FTGSPP

FTGSPP is a Hydra-based training pipeline for
[FreeTimeGS++](https://github.com/SNU-VGILab/FreeTimeGSPlusPlus). It is designed
for the EasyMocap dataset format.

## Acknowledgements

We thank the [FreeTimeGS](https://zju3dv.github.io/freetimegs/) and FreeTimeGS++
authors for their foundational contributions. Please cite both works when
using this repository.

## Installation

The tested environment requires Linux, an NVIDIA GPU, Python 3.12, the CUDA
Toolkit, and the following build tools: `git`, `gcc`, `g++`, `make`, `cmake`,
and `ninja`. It uses PyTorch `2.5.1+cu121` and `gsplat==1.5.3`.

From the repository root, run:

```bash
bash scripts/install_env.sh
```

The script creates `.venv` and installs all training dependencies. If CUDA is
not detected automatically, set it explicitly:

```bash
export CUDA_HOME=/usr/local/cuda-12.1
bash scripts/install_env.sh
```

Verify the environment with:

```bash
bash scripts/install_env.sh --check
```

## Dataset Layout

The recommended layout defines training and test cameras directly through
calibration files:

```text
scene/
  train_intri.yml
  train_extri.yml
  test_intri.yml
  test_extri.yml
  t_offsets.json
  images/
    0000/000000.jpg ...
    0001/000000.jpg ...
```

When all four split calibration files are present:

- cameras in `train_*.yml` are used for point generation and training;
- cameras in `test_*.yml` are used for evaluation and test-view rendering;
- camera lists do not need to be specified manually in the config.

If the dataset only provides `intri.yml` and `extri.yml`, set `eval_cameras`
in the data config. The complement is used as the training-camera set.

## Adding a Dataset

Copy an existing data config:

```bash
cp configs/data/corgi_first300_scale05.yaml configs/data/my_scene.yaml
```

Edit `configs/data/my_scene.yaml`:

```yaml
name: my_scene
fps: 60.0
root: /absolute/path/to/scene
video_path: ${data.root}
calibration_path: ${data.root}
extracted_path: ${data.root}/images
scale: 1.0
frames:
  start: 0
  stop: 300
train_cameras: null
eval_cameras: null
camera_split_source: null
```

## Training

Check the config, images, calibration, and camera split without running any
training stages:

```bash
.venv/bin/python -m ftgspp.run data=my_scene pipeline=check
```

Start the complete pipeline after the check succeeds:

```bash
.venv/bin/python -m ftgspp.run data=my_scene run.gpu=0
```

The default pipeline runs `points -> init -> train -> export` with RoMaV2 and
a two-million Gaussian capacity.

Hydra parameters can be overridden from the command line:

```bash
.venv/bin/python -m ftgspp.run \
  data=my_scene \
  data.frames.stop=30 \
  data.scale=1.0 \
  init.num_gaussians=2000000 \
  init.roma.threshold=0.25 \
  train.iterations=10000 \
  train.num_workers=4 \
  run.gpu=0
```

Start the bundled Corgi config with:

```bash
.venv/bin/python -m ftgspp.run data=corgi_first300_scale05 run.gpu=0
```

Outputs are written to:

```text
runs/<data.name>/
  points_stride10/
  run_<iterations>/
    config.resolved.yaml
    init.pt
    init.ply
    ckpt/
    gaussians.pt
    gaussians_4dgs.ply
```

## Evaluation and Test-View Rendering

Only the training run directory is required. The dataset path, frame range,
and test-camera split are loaded from `config.resolved.yaml`.

Compute evaluation metrics:

```bash
RUN=runs/my_scene/run_30000
.venv/bin/python -m ftgspp.eval "$RUN" "$RUN/metrics.json"
```

Render the calibrated test views from `test_intri.yml` and `test_extri.yml`:

```bash
RUN=runs/my_scene/run_30000
.venv/bin/python scripts/render_test_views.py "$RUN" "$RUN/test_render"
```

Evaluation and rendering use every tenth frame by default. Process every test
frame with:

```bash
RUN=runs/my_scene/run_30000
.venv/bin/python -m ftgspp.eval \
  "$RUN" "$RUN/metrics_all.json" --frame-interval 1

.venv/bin/python scripts/render_test_views.py \
  "$RUN" "$RUN/test_render_all" --frame-interval 1
```

Rendered images preserve the camera names recorded in the calibration YAML:

```text
test_render/
  <camera_name_from_yml>/<source_frame>.png
```

## Citation

If this repository contributes to your research, please cite
[FreeTimeGS++](https://arxiv.org/abs/2605.03337) and the original
[FreeTimeGS](https://openaccess.thecvf.com/content/CVPR2025/html/Wang_FreeTimeGS_Free_Gaussian_Primitives_at_Anytime_Anywhere_for_Dynamic_Scene_CVPR_2025_paper.html)
paper.

```bibtex
@article{lee2026freetimegspp,
  title   = {FreeTimeGS++: Secrets of Dynamic Gaussian Splatting and Their Principles},
  author  = {Lee, Lucas Yunkyu and Kim, Soonho and Kim, Youngwook and Kim, Sangmin and Park, Jaesik},
  journal = {arXiv preprint arXiv:2605.03337},
  year    = {2026}
}
```

```bibtex
@inproceedings{Wang_2025_CVPR,
  author    = {Wang, Yifan and Yang, Peishan and Xu, Zhen and Sun, Jiaming and Zhang, Zhanhua and Chen, Yong and Bao, Hujun and Peng, Sida and Zhou, Xiaowei},
  title     = {FreeTimeGS: Free Gaussian Primitives at Anytime Anywhere for Dynamic Scene Reconstruction},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month     = {June},
  year      = {2025},
  pages     = {21750--21760}
}
```

## License

See [LICENSE](LICENSE) for the source-code license and
[THIRD_PARTY.md](THIRD_PARTY.md) for third-party software notices.
