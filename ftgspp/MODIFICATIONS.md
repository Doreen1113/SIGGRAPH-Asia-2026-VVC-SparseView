# Modifications to FreeTimeGS++ (FTGSPP)

This directory is the FTGSPP baseline distributed with the SIGGRAPH Asia 2026 Volumetric Video Challenge
(`SOURCE_REVISION`: commit `92f962eb`, bundle date 2026-07-14), modified by Team Doreen071.
It remains under the FreeTimeGS++ Research and Educational Use License in [LICENSE](LICENSE)
(non-commercial research and education only). As that license requires, every modified file carries a
notice on its first line. `checksums.sha256` lists the **original** hashes, so it reports those eight files
as changed.

Every addition is switched off by default and enabled through an environment variable, so the unmodified
code path is still the baseline.

## Exact patches

For three files a checksum-verified copy of the original was kept, so their full diffs are in `patches/`:

| file | patch | what changed |
|---|---|---|
| `ftgspp/train/train.py` | [patches/ftgspp_train_train.patch](patches/ftgspp_train_train.patch) | all new training losses (table below) |
| `ftgspp/models/gaussians.py` | [patches/ftgspp_models_gaussians.patch](patches/ftgspp_models_gaussians.patch) | optional expected-depth output (`RGB+ED`); `eps2d` and rasterize mode made configurable |
| `ftgspp/data/utils/mvdataset.py` | [patches/ftgspp_data_utils_mvdataset.patch](patches/ftgspp_data_utils_mvdataset.patch) | hidden test cameras (calibration without images) no longer abort loading |

No original copy was kept for the other five files, so they have no patch. Their changes are small:

| file | change |
|---|---|
| `ftgspp/train/__init__.py` | loss weights read from the environment: `FTGSPP_LPIPS_W` (default 0.01), `FTGSPP_L1_W`, `FTGSPP_SSIM_W`, `FTGSPP_REG_OPACITY`, `FTGSPP_REG_SCALE`, `FTGSPP_REG_DURATION`, `FTGSPP_CHECKPOINT_INTERVAL` |
| `ftgspp/init/points.py` | `FTGSPP_ALL_PAIRS=1` matches every camera pair for RoMa initialisation (`FTGSPP_PAIR_MAXDIST` caps the baseline) |
| `ftgspp/config/validate.py`, `ftgspp/data/utils/loader.py` | checks relaxed so that a scene with no evaluation images (hidden test views) can be trained |
| `pyproject.toml` | edited while setting up the environment on the cluster; the original was not kept, so the exact change is unrecorded |

## Training-time additions (`ftgspp/train/train.py`)

| env var(s) | mechanism | outcome in the challenge |
|---|---|---|
| `FTGSPP_DEPTH_W`, `FTGSPP_POINTS_DIR` | sparse depth loss against the RoMa keyframe point cloud | used from recipe F onwards |
| `FTGSPP_DENSE_W`, `FTGSPP_DENSE_DEPTH_DIR`, `FTGSPP_DENSE_CONF` | dense depth loss against per-frame VGGT depth (confidence-thresholded) | used (recipe J, weight 0.1) |
| `FTGSPP_PSEUDO_DIR`, `FTGSPP_PSEUDO_W`, `_LPIPS`, `_START`, `_EVERY`, `_COLORFIT`, `_COLORFIT_DAMP`, `_POOL`, `_MASK`, `_COVERAGE`, `_AGREE`, `_SAMPLE_POW` | supervision from generated pseudo-views (SEVA or Difix output) at the hidden camera poses, restricted to pixels **no training camera covers**, with a damped per-image colour fit | used (P8/P9, and in every final person model) |
| `FTGSPP_FG_WEIGHT`, `FTGSPP_FG_DILATE` | extra L1+SSIM on the person region of the real training views (DeepLabV3 mask) | used (weight 4.0 with 60k iterations) |
| `FTGSPP_SIL_WEIGHT` | BCE+Dice between the rasteriser's accumulated alpha and the person mask | trained, never submitted alone |
| `FTGSPP_ODECAY`, `_EVERY`, `_START` | visibility-aware opacity decay: every N iterations, scale down the opacity of the Gaussians rasterised in the current view | combined with the silhouette loss it failed badly on test (FG-PSNR -2.1 dB) |
| `FTGSPP_UNSEEN_*` | depth total-variation at random unseen poses | failed on validation, not used |
| `FTGSPP_CARVE_*` | free-space carving against dense depth | no gain, not used |
