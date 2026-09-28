# Modified by Team Doreen071 for the SIGGRAPH Asia 2026 VVC Sparse-View track (2026-09); see MODIFICATIONS.md.
# FreeTimeGS++
# 2025-2026 Lucas Yunkyu Lee <lucaslee@postech.ac.kr>, SNU VGI Lab

import logging
import os
from dataclasses import asdict
from pathlib import Path
from typing import Mapping, Sequence

import rerun as rr
import torch
from torch.utils.tensorboard import SummaryWriter
from fused_ssim import fused_ssim
from torch import Tensor, nn
from torchmetrics.image import (
    LearnedPerceptualImagePatchSimilarity,
    PeakSignalNoiseRatio,
)
from tqdm import tqdm

from ftgspp.data.utils import MVDataset, make_mvdataset_view_loader
from ftgspp.eval import eval_rgb_camera_chunks
from ftgspp.models import ColorCorrectors, Gaussians, VelocityField
from ftgspp.train.optim import (
    LRSchedulerCollection,
    OptimizerCollection,
)
from ftgspp.train.relocation import relocate, relocation_binoms
from ftgspp.train.state import TrainState
from ftgspp.utils import tensor_to_png
from ftgspp.utils.math import chw, scene_extent


def train(
    gs: Gaussians,
    color_correctors: ColorCorrectors | None,
    dataset: MVDataset,
    batch_size: int,
    iterations: int,
    num_workers: int,
    prefetch_factor: int,
    persistent_workers: bool,
    pin_memory: bool,
    seed: int,
    eval_frame_interval: int,
    eval_num_workers: int,
    eval_prefetch_factor: int,
    eval_pin_memory: bool,
    frames: slice,
    train_cameras: Sequence[int],
    eval_cameras: Sequence[int],
    relocation: bool,
    relocation_start: int,
    relocation_every: int,
    relocation_stop: int,
    relocation_opacity_threshold: float,
    relocation_mode: str,
    relocation_score_mode: str,
    relocation_score_mode_start: int,
    color_correction_start: int,
    run_path: Path,
    loss_weights: Mapping[str, float],
    sh_degree_schedule: bool,
    lrs: Mapping[str, float],
    lr_schedules: Mapping[str, float],
    checkpoint_iterations: list[int],
    logger: logging.Logger,
    start_iteration: int = 0,
    velocity_distill: tuple[Tensor, Tensor, Tensor] | None = None,
    velocity_distill_warmup_steps: int = 0,
    velocity_distill_weight: float = 0.0,
    velocity_distill_batch_size: int = 32768,
):
    print(f"Using {len(gs)} Gaussians")

    # Data
    if batch_size != 1:
        raise ValueError("MVDataset training requires batch_size = 1")
    train_set = dataset.at(frames, train_cameras)
    view_loader = make_mvdataset_view_loader(
        train_set,
        num_samples=iterations,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
        seed=seed + start_iteration,
    )
    view_iterator = iter(view_loader)
    logger.info(
        "MVDataset DataLoader: workers=%d prefetch_factor=%d "
        "persistent_workers=%s pin_memory=%s",
        num_workers,
        prefetch_factor,
        persistent_workers and num_workers > 0,
        pin_memory,
    )

    # Optimizers
    optimizers, schedulers = make_optimizers(
        gs=gs,
        color_correctors=color_correctors,
        lrs=lrs,
        lr_schedules=lr_schedules,
        batch_size=batch_size,
        iterations=iterations,
        scene_extent=scene_extent(train_set[0].w2c),  # type: ignore
    )

    # Relocation
    state = None
    binoms = None
    if relocation:
        state = TrainState.new(len(gs))
        binoms = relocation_binoms().cuda()

    psnr_fn = PeakSignalNoiseRatio(data_range=1).cuda()
    lpips_fn = None
    if loss_weights["lpips"] > 0:
        lpips_fn = LearnedPerceptualImagePatchSimilarity(
            net_type="alex", normalize=False
        ).cuda()
    velocity_distill_xyz = None
    velocity_distill_t = None
    velocity_distill_target = None
    velocity_distill_scale = None
    if velocity_distill is not None:
        velocity_distill_xyz, velocity_distill_t, velocity_distill_target = velocity_distill
        velocity_distill_xyz = velocity_distill_xyz.to(gs.means.device)
        velocity_distill_t = velocity_distill_t.to(gs.means.device)
        velocity_distill_target = velocity_distill_target.to(gs.means.device)
        velocity_distill_scale = torch.linalg.norm(
            velocity_distill_target, dim=-1
        ).mean().clamp(min=1e-6)

    postfix = {}
    depth_w = float(os.environ.get("FTGSPP_DEPTH_W", "0"))
    depth_pts = {}
    if depth_w > 0:
        from ftgspp.utils.io import read_ply_points
        pdir = Path(os.environ["FTGSPP_POINTS_DIR"])
        for ply in sorted(pdir.glob("f*.ply")):
            xyz, _ = read_ply_points(ply)
            depth_pts[int(ply.stem[1:])] = torch.tensor(xyz, dtype=torch.float32).cuda()
        logger.info("depth supervision: weight=%.3f keyframes=%d", depth_w, len(depth_pts))
    dense_w = float(os.environ.get("FTGSPP_DENSE_W", "0"))
    dense_dir = Path(os.environ.get("FTGSPP_DENSE_DEPTH_DIR", "")) if dense_w > 0 else None
    dense_conf_thr = float(os.environ.get("FTGSPP_DENSE_CONF", "2.0"))
    dense_cache = {}
    if dense_w > 0:
        import numpy as _np
        cam_names_by_id = {int(c): dataset.camera_name(int(c)) for c in train_cameras}
        def _load_dense(frame_id):
            if frame_id not in dense_cache:
                f = dense_dir / f"f{frame_id:06d}.npz"
                if not f.exists():
                    dense_cache[frame_id] = None
                else:
                    z = _np.load(f); views = [str(v) for v in z["views"]]
                    dense_cache[frame_id] = {v: (torch.from_numpy(z["depth"][i].astype(_np.float32)), torch.from_numpy(z["conf"][i].astype(_np.float32))) for i, v in enumerate(views)}
                if len(dense_cache) > 64:
                    dense_cache.pop(next(iter(dense_cache)))
            return dense_cache[frame_id]
        logger.info("dense depth supervision: weight=%.3f dir=%s conf>%.1f", dense_w, dense_dir, dense_conf_thr)
    unseen_every = int(os.environ.get("FTGSPP_UNSEEN_EVERY", "0"))
    unseen_w = float(os.environ.get("FTGSPP_UNSEEN_W", "0.05"))
    unseen_patch = int(os.environ.get("FTGSPP_UNSEEN_PATCH", "256"))
    unseen_start = int(os.environ.get("FTGSPP_UNSEEN_START", "1000"))
    unseen_pool = None
    if unseen_every > 0:
        _tw = train_set[0, :].w2c.reshape(-1, 4, 4).float()
        _tk = train_set[0, :].intrinsic.reshape(-1, 3, 3).float()
        _c2w = torch.inverse(_tw)
        logger.info("unseen-view regularization: every=%d w=%.4f patch=%d cams=%d", unseen_every, unseen_w, unseen_patch, len(_c2w))
        unseen_pool = (_c2w.cuda(), _tk.cuda())
    tb_writer = SummaryWriter(log_dir=str(run_path / "tensorboard"))
    train_eval_interval = int(os.environ.get("FTGSPP_TRAIN_EVAL_INTERVAL", "500"))
    # ---- pseudo-view supervision (generated novel views at held-out cameras) ----
    pseudo_w = float(os.environ.get("FTGSPP_PSEUDO_W", "0"))
    pseudo_items = []
    pseudo_lpips_w = float(os.environ.get("FTGSPP_PSEUDO_LPIPS", "0"))
    pseudo_every = int(os.environ.get("FTGSPP_PSEUDO_EVERY", "1"))
    pseudo_start = int(os.environ.get("FTGSPP_PSEUDO_START", "0"))
    pseudo_colorfit = int(os.environ.get("FTGSPP_PSEUDO_COLORFIT", "1"))
    pseudo_pool = int(os.environ.get("FTGSPP_PSEUDO_POOL", "1"))  # avg-pool factor: supervise only low-frequency content
    pseudo_damp = float(os.environ.get("FTGSPP_PSEUDO_COLORFIT_DAMP", "0.5"))  # 1.0 = full affine color fit
    pseudo_lpips_fn = None
    if pseudo_w > 0:
        import json as _json
        import numpy as _np
        import cv2 as _cv2
        pdir = Path(os.environ["FTGSPP_PSEUDO_DIR"])
        for e in _json.load(open(pdir / "index.json")):
            im = _cv2.imread(str(pdir / e["path"]))[..., ::-1]
            mpath = pdir / e["path"].replace(".png", "_mask.png")
            pmask = None
            if int(os.environ.get("FTGSPP_PSEUDO_MASK", "1")) and mpath.exists():
                pmask = torch.from_numpy(_cv2.imread(str(mpath), 0)).float().div(255).cuda()[None, ..., None]  # 1HW1, 1 = supervise (background)
            cpath = pdir / e["path"].replace(".png", "_covered.png")
            if int(os.environ.get("FTGSPP_PSEUDO_COVERAGE", "1")) and cpath.exists():
                # covered.png: 255 = already seen by a train camera (skip), 0 = uncovered (safe to supervise)
                cov = torch.from_numpy(_cv2.imread(str(cpath), 0)).float().div(255).cuda()[None, ..., None]
                pmask = (1 - cov) if pmask is None else pmask * (1 - cov)
            apath = pdir / e["path"].replace(".png", "_agree.png")
            if int(os.environ.get("FTGSPP_PSEUDO_AGREE", "1")) and apath.exists():
                # agree.png: 255 = multi-seed samples agree (trustworthy), 0 = seva is guessing (skip)
                agr = torch.from_numpy(_cv2.imread(str(apath), 0)).float().div(255).cuda()[None, ..., None]
                pmask = agr if pmask is None else pmask * agr
            pseudo_items.append(dict(
                mask=pmask,
                rgb=torch.from_numpy(_np.ascontiguousarray(im)).float().div(255).cuda()[None],  # 1HW3
                t=torch.tensor(float(e["t"])).cuda(),
                w2c=torch.tensor(_np.array(e["w2c"], dtype=_np.float32)).cuda()[None],
                K=torch.tensor(_np.array(e["K"], dtype=_np.float32)).cuda()[None],
                h=int(e["h"]), w=int(e["w"]), view=e["view"], frame=int(e["frame"]),
                sample_w=float(pmask.mean()) if pmask is not None else 1.0,  # supervised-pixel fraction -> sampling weight
            ))
        if pseudo_lpips_w > 0:
            pseudo_lpips_fn = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=False).cuda()
        logger.info("pseudo-view supervision: weight=%.3f lpips=%.3f n=%d (masked %d, mean keep frac %.3f) every=%d start=%d colorfit=%d pool=%d damp=%.2f",
                    pseudo_w, pseudo_lpips_w, len(pseudo_items), sum(p["mask"] is not None for p in pseudo_items),
                    float(torch.stack([p["mask"].float().mean() for p in pseudo_items if p["mask"] is not None]).mean()) if any(p["mask"] is not None for p in pseudo_items) else 1.0,
                    pseudo_every, pseudo_start, pseudo_colorfit, pseudo_pool, pseudo_damp)
    pseudo_rng = torch.Generator().manual_seed(seed + 12345)
    # ---- foreground (person) emphasis on REAL training-view supervision ----
    fg_weight = float(os.environ.get("FTGSPP_FG_WEIGHT", "0"))
    sil_weight = float(os.environ.get("FTGSPP_SIL_WEIGHT", "0"))  # silhouette (rendered alpha vs person mask) BCE+Dice
    odecay = float(os.environ.get("FTGSPP_ODECAY", "0"))          # visibility-aware opacity decay factor, e.g. 0.9995
    odecay_every = int(os.environ.get("FTGSPP_ODECAY_EVERY", "200"))
    odecay_start = int(os.environ.get("FTGSPP_ODECAY_START", "2000"))
    fg_seg_model = None
    if fg_weight > 0 or sil_weight > 0:
        from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights
        fg_seg_model = deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1).cuda().eval().half()
        for p_ in fg_seg_model.parameters():
            p_.requires_grad_(False)
        fg_mean = torch.tensor([0.485, 0.456, 0.406]).cuda().view(1, 3, 1, 1)
        fg_std = torch.tensor([0.229, 0.224, 0.225]).cuda().view(1, 3, 1, 1)
        fg_dilate = int(os.environ.get("FTGSPP_FG_DILATE", "8"))
        logger.info("foreground emphasis: weight=%.3f dilate=%d", fg_weight, fg_dilate)
    pseudo_sample_pow = float(os.environ.get("FTGSPP_PSEUDO_SAMPLE_POW", "0"))  # >0: oversample low-coverage views ~ sample_w^pow
    pseudo_sample_probs = None
    if pseudo_items and pseudo_sample_pow > 0:
        w = torch.tensor([max(it["sample_w"], 1e-3) ** pseudo_sample_pow for it in pseudo_items])
        pseudo_sample_probs = w / w.sum()
        logger.info("pseudo sampling reweighted by (supervised fraction)^%.2f: min=%.4f max=%.4f (vs uniform %.4f)",
                    pseudo_sample_pow, pseudo_sample_probs.min().item(), pseudo_sample_probs.max().item(), 1.0 / len(pseudo_items))

    for it in (bar := tqdm(range(start_iteration, start_iteration + iterations))):
        rr.set_time("iteration", sequence=it)

        batch = next(view_iterator).cuda(non_blocking=pin_memory)

        t = batch.time[0]
        cams = [int(camera) for camera in batch.camera]  # type: ignore

        frame_id = int(batch.frame[0]) if (depth_w > 0 or dense_w > 0) else -1
        dense_tgt = None
        if dense_w > 0:
            dd = _load_dense(frame_id)
            cname = cam_names_by_id.get(cams[0])
            if dd is not None and cname in dd:
                dense_tgt = dd[cname]
        use_depth = (frame_id in depth_pts) or (dense_tgt is not None)
        pred_, pred_alpha, aux = gs(
            t=t,
            w2c=batch.w2c,
            intrinsic=batch.intrinsic,
            shape=(batch.height, batch.width),
            sh_degree=min(it // 1000, gs.sh_degree)
            if sh_degree_schedule
            else gs.sh_degree,
            render_depth=use_depth,
        )
        loss_depth = torch.zeros((), device=pred_.device)
        if dense_tgt is not None:
            dz, dc = dense_tgt
            dz = torch.nn.functional.interpolate(dz[None, None].cuda(), size=(batch.height, batch.width), mode="bilinear", align_corners=False)[0, 0]
            dc = torch.nn.functional.interpolate(dc[None, None].cuda(), size=(batch.height, batch.width), mode="bilinear", align_corners=False)[0, 0]
            dmask2 = (dc > dense_conf_thr) & (dz > 0.05)
            if dmask2.any():
                dr2 = aux["depth"][0, ..., 0]
                loss_depth = loss_depth + dense_w * (torch.abs(dr2[dmask2] - dz[dmask2]) / dz[dmask2]).mean()
        if frame_id in depth_pts:
            with torch.no_grad():
                P = depth_pts[frame_id]
                w2c0 = batch.w2c[0]; K0 = batch.intrinsic[0]
                c = P @ w2c0[:3, :3].T + w2c0[:3, 3]
                z = c[:, 2]
                u = (K0[0, 0] * c[:, 0] / z + K0[0, 2]).round().long()
                v = (K0[1, 1] * c[:, 1] / z + K0[1, 2]).round().long()
                ok = (z > 0.05) & (u >= 0) & (u < batch.width) & (v >= 0) & (v < batch.height)
                u, v, z = u[ok], v[ok], z[ok]
                lin = v * batch.width + u
                tgt = torch.full((batch.height * batch.width,), float("inf"), device=z.device)
                tgt.scatter_reduce_(0, lin, z, reduce="amin")
                dmask = torch.isfinite(tgt)
            if dmask.any():
                dr = aux["depth"][0].reshape(-1)
                loss_depth = loss_depth + depth_w * (torch.abs(dr[dmask] - tgt[dmask]) / tgt[dmask]).mean()
        if color_correctors is not None and it > color_correction_start:
            cc = color_correctors
            pred = cc(cams, pred_)
        else:
            pred = pred_
        aux["means2d"].retain_grad()

        # visibility-aware opacity decay (4C4D's core idea, ported rather than adopting their pipeline):
        # with only six training views a 4DGS happily raises the opacity of primitives that memorise the
        # training images, which reads as low error there and as broken geometry from the hidden cameras.
        # Periodically nudging down the opacity of the Gaussians that were actually rasterised keeps that
        # shortcut expensive; genuinely needed primitives are pushed back up by the next gradient steps, and
        # anything that falls through the floor is recycled by the existing relocation pass.
        if odecay > 0 and it > odecay_start and it % odecay_every == 0 and "gaussian_ids" in aux:
            with torch.no_grad():
                vis = aux["gaussian_ids"].reshape(-1).unique()
                if vis.numel() > 0:
                    o = gs.opacities.data[vis].sigmoid() * odecay
                    gs.opacities.data[vis] = torch.logit(o.clamp(1e-6, 1 - 1e-6))
                    postfix.update({"odecay": f"{vis.numel()}"})

        gt = batch.rgb.float() / 255
        zero = torch.zeros((), device=pred.device)
        loss_l1 = loss_weights["l1"] * nn.functional.l1_loss(pred, gt)
        loss_ssim = loss_weights["ssim"] * (
            1 - fused_ssim(chw(pred), chw(gt), padding="valid")
        )
        if lpips_fn is not None:
            # LPIPS with normalize=False expects inputs in [0, 1].
            # Color correction can push RGB slightly outside that range.
            pred_lpips = pred.clamp(0, 1)
            gt_lpips = gt.clamp(0, 1)
            loss_lpips = loss_weights["lpips"] * lpips_fn(chw(pred_lpips), chw(gt_lpips))
        else:
            loss_lpips = zero
        if lpips_fn is not None:
            lpips_fn.reset()
        loss_reg_opacity = loss_weights["reg_opacity"] * torch.mean(
            gs.opacities.sigmoid() * gs.temporal_opacity(t).detach()
        )
        loss_reg_scale = loss_weights["reg_scale"] * gs.scales.exp().mean()
        loss_reg_duration = loss_weights["reg_duration"] * torch.mean(
            (1 - gs.gate().detach()) * gs.temporal_scale()
        )
        loss_reg = loss_reg_opacity + loss_reg_scale + loss_reg_duration
        loss_gate = loss_weights["gate"] * torch.mean(1 - gs.gate())
        loss_color_correctors = (
            loss_weights["color_correctors"] * color_correctors.regularize()
            if color_correctors is not None
            else zero
        )
        loss_velocity_distill = zero
        if (
            velocity_distill_xyz is not None
            and velocity_distill_t is not None
            and velocity_distill_target is not None
            and velocity_distill_scale is not None
            and velocity_distill_weight > 0
            and velocity_distill_warmup_steps > 0
            and (it - start_iteration) < velocity_distill_warmup_steps
            and isinstance(gs.velocity_model, VelocityField)
        ):
            n_total = len(velocity_distill_target)
            n_sample = min(velocity_distill_batch_size, n_total)
            sample_idxs = torch.randint(
                0,
                n_total,
                (n_sample,),
                device=velocity_distill_target.device,
            )
            pred_velocity = gs.velocity_model(
                velocity_distill_xyz[sample_idxs],
                velocity_distill_t[sample_idxs],
            )
            target_velocity = velocity_distill_target[sample_idxs]
            warmup_ratio = 1 - (it - start_iteration) / velocity_distill_warmup_steps
            loss_velocity_distill = (
                velocity_distill_weight
                * warmup_ratio
                * nn.functional.smooth_l1_loss(
                    pred_velocity / velocity_distill_scale,
                    target_velocity / velocity_distill_scale,
                )
            )

        loss_pseudo = torch.zeros((), device=pred.device)
        if pseudo_items and it >= pseudo_start and it % pseudo_every == 0:
            if pseudo_sample_probs is not None:
                idx = int(torch.multinomial(pseudo_sample_probs, 1, generator=pseudo_rng))
            else:
                idx = int(torch.randint(len(pseudo_items), (1,), generator=pseudo_rng))
            pi = pseudo_items[idx]
            ppred, _, _ = gs(t=pi["t"], w2c=pi["w2c"], intrinsic=pi["K"], shape=(pi["h"], pi["w"]),
                             sh_degree=min(it // 1000, gs.sh_degree) if sh_degree_schedule else gs.sh_degree)
            ptgt = pi["rgb"]
            if pseudo_colorfit:
                with torch.no_grad():  # per-image affine color fit target->render (removes global tone mismatch)
                    sel = (pi["mask"].reshape(-1) > 0.5) if pi["mask"] is not None else slice(None)
                    X = ptgt.reshape(-1, 3)[sel][::7]; Y = ppred.detach().reshape(-1, 3)[sel][::7]
                    X1 = torch.cat([X, torch.ones_like(X[:, :1])], 1)
                    if pseudo_colorfit == 2:  # per-channel gain+bias only (robust)
                        A = torch.zeros(4, 3, device=X.device)
                        for ch in range(3):
                            xc = torch.stack([X[:, ch], torch.ones_like(X[:, ch])], 1)
                            sol = torch.linalg.lstsq(xc, Y[:, ch:ch + 1]).solution[:, 0]
                            A[ch, ch] = sol[0]; A[3, ch] = sol[1]
                    else:
                        A = torch.linalg.lstsq(X1, Y).solution  # 4x3
                    A[:3] = pseudo_damp * A[:3] + (1 - pseudo_damp) * torch.eye(3, device=A.device); A[3] *= pseudo_damp  # damp toward identity
                    ptgt = (torch.cat([ptgt.reshape(-1, 3), torch.ones_like(ptgt.reshape(-1, 3)[:, :1])], 1) @ A).reshape(ptgt.shape).clamp(0, 1)
            if pi["mask"] is not None:  # supervise only unmasked (static background) pixels
                pm = pi["mask"]
                ppred_m = ppred * pm + ptgt.detach() * (1 - pm)
            else:
                ppred_m = ppred
            if pseudo_pool > 1:
                pp_ = nn.functional.avg_pool2d(chw(ppred_m), pseudo_pool); pt_ = nn.functional.avg_pool2d(chw(ptgt), pseudo_pool)
            else:
                pp_ = chw(ppred_m); pt_ = chw(ptgt)
            loss_pseudo = pseudo_w * (0.8 * nn.functional.l1_loss(pp_, pt_) + 0.2 * (1 - fused_ssim(pp_, pt_, padding="valid")))
            if pseudo_lpips_fn is not None:
                loss_pseudo = loss_pseudo + pseudo_lpips_w * pseudo_lpips_fn(chw(ppred_m.clamp(0, 1)), chw(ptgt))
                pseudo_lpips_fn.reset()
        loss = (
            loss_l1
            + loss_ssim
            + loss_lpips
            + loss_reg
            + loss_gate
            + loss_color_correctors
            + loss_velocity_distill
            + loss_depth
            + loss_pseudo
        )

        loss_unseen = torch.zeros((), device=pred_.device)
        if unseen_pool is not None and it > unseen_start and it % unseen_every == 0:
            c2ws, Ks = unseen_pool
            n = len(c2ws)
            i0, i1 = torch.randint(0, n, (2,), device=c2ws.device).tolist()
            if i0 == i1: i1 = (i1 + 1) % n
            alpha = float(torch.rand(1)) * 1.6 - 0.3   # interpolate and mildly extrapolate
            A, B = c2ws[i0], c2ws[i1]
            Rn = A[:3, :3] + alpha * (B[:3, :3] - A[:3, :3])
            U, _, Vh = torch.linalg.svd(Rn.double()); Rn = (U @ Vh).float()
            if torch.det(Rn) < 0: Rn = -Rn
            tn_ = A[:3, 3] + alpha * (B[:3, 3] - A[:3, 3])
            c2w_n = torch.eye(4, device=c2ws.device); c2w_n[:3, :3] = Rn; c2w_n[:3, 3] = tn_
            w2c_n = torch.inverse(c2w_n)[None]
            Kn = Ks[i0].clone()
            ph = min(unseen_patch, int(batch.height)); pw = min(unseen_patch, int(batch.width))
            oy = int(torch.randint(0, max(1, int(batch.height) - ph), (1,))); ox = int(torch.randint(0, max(1, int(batch.width) - pw), (1,)))
            Kp = Kn.clone(); Kp[0, 2] -= ox; Kp[1, 2] -= oy
            u_img, _, u_meta = gs(t=t, w2c=w2c_n, intrinsic=Kp[None], shape=(ph, pw), clamp=False, render_depth=True)
            ud = u_meta["depth"][0, ..., 0]
            dx = torch.abs(ud[:, 1:] - ud[:, :-1]); dy = torch.abs(ud[1:] - ud[:-1])
            scale_d = ud.detach().clamp(min=0.1)
            loss_unseen = unseen_w * ((dx / scale_d[:, 1:]).mean() + (dy / scale_d[1:]).mean())
            loss = loss + loss_unseen
            postfix.update({"unseen": f"{float(loss_unseen.detach()):.4f}"})

        loss_fg = torch.zeros((), device=pred.device)
        loss_sil = torch.zeros((), device=pred.device)
        if fg_seg_model is not None:
            with torch.no_grad():
                gt_chw = chw(gt)
                seg_in = ((gt_chw - fg_mean) / fg_std).half()
                person = fg_seg_model(seg_in)["out"].float().argmax(1) == 15  # NCHW logits -> person class
                if fg_dilate > 0:
                    person = torch.nn.functional.max_pool2d(person.float()[:, None], fg_dilate * 2 + 1, stride=1, padding=fg_dilate) > 0
                    person = person[:, 0]
                fg_mask = person[..., None].float()  # NHW1
            if fg_weight > 0 and fg_mask.mean() > 1e-4:
                loss_fg = fg_weight * (
                    nn.functional.l1_loss(pred * fg_mask, gt * fg_mask)
                    + 0.2 * (1 - fused_ssim(chw(pred * fg_mask), chw(gt * fg_mask), padding="valid"))
                )
            loss = loss + loss_fg
            # silhouette: push the renderer's own accumulated alpha toward the segmentation mask. The rasterizer
            # already computes this alpha for free; it was previously discarded (`pred_, _, aux = gs(...)`).
            # This penalises Gaussians that make the person edge translucent to cheat photometric loss, which is
            # a distinct failure mode from just up-weighting FG appearance error (FTGSPP_FG_WEIGHT).
            if sil_weight > 0:
                a = pred_alpha.clamp(1e-4, 1 - 1e-4)
                bce = -(fg_mask * a.log() + (1 - fg_mask) * (1 - a).log()).mean()
                inter = (a * fg_mask).sum(); dice = 1 - (2 * inter + 1) / (a.sum() + fg_mask.sum() + 1)
                loss_sil = sil_weight * (bce + dice)
                loss = loss + loss_sil
            postfix.update({"fg": f"{float(loss_fg.detach()):.4f}", "sil": f"{float(loss_sil.detach()):.4f}"})

        loss.backward()

        optimizers.step()
        carve_every = int(os.environ.get("FTGSPP_CARVE_EVERY", "0"))
        if carve_every > 0 and dense_tgt is not None and it % carve_every == 0 and it > int(os.environ.get("FTGSPP_CARVE_START", "1000")):
            with torch.no_grad():
                cm = float(os.environ.get("FTGSPP_CARVE_MARGIN", "0.15")); dz0, dc0 = dense_tgt
                dz0 = dz0.cuda(); dc0 = dc0.cuda(); hh, ww = dz0.shape
                w2c0 = batch.w2c[0]; K0 = batch.intrinsic[0]
                mt = gs.means_t(t); c = mt @ w2c0[:3, :3].T + w2c0[:3, 3]; z = c[:, 2]
                u = (K0[0, 0] * c[:, 0] / z + K0[0, 2]) * (ww / batch.width); v = (K0[1, 1] * c[:, 1] / z + K0[1, 2]) * (hh / batch.height)
                ui = u.round().long(); vi = v.round().long()
                ok = (z > 0.05) & (ui >= 0) & (ui < ww) & (vi >= 0) & (vi < hh)
                act = (gs.temporal_opacity(t).squeeze(-1) > 0.5) & ok
                idx = act.nonzero(as_tuple=True)[0]
                d = dz0[vi[idx], ui[idx]]; cf = dc0[vi[idx], ui[idx]]
                bad = idx[(cf > dense_conf_thr) & (z[idx] < (1 - cm) * d)]
                if len(bad) > 0:
                    gs.opacities.data[bad] = -8.0  # sigmoid ~ 3e-4 -> will be relocated as dead
                postfix.update({"carved": f"{len(bad)}"})
        schedulers.step()
        optimizers.zero_grad()

        if relocation:
            assert state is not None
            assert binoms is not None
            with torch.no_grad():
                state.update_(aux)

                if (
                    it % relocation_every == 0
                    and relocation_start < it < relocation_stop
                ):
                    relocated, is_all_dead = relocate_gaussians(
                        state=state,
                        gs=gs,
                        opacity_threshold=relocation_opacity_threshold,
                        mode=relocation_mode,
                        score_mode=relocation_score_mode,
                        score_mode_start=relocation_score_mode_start,
                        optimizers=optimizers,
                        binoms=binoms,
                        it=it,
                        logger=logger,
                    )
                    state.zero_()
                    postfix.update({"relocated": f"{relocated}"})
                    if is_all_dead:
                        logger.warning(
                            "Stopping early at iteration %d: all Gaussians are dead "
                            "(threshold=%.6f).",
                            it,
                            relocation_opacity_threshold,
                        )
                        bar.set_postfix(postfix)
                        return gs

        if it in checkpoint_iterations:
            ckpt_dir = run_path / "ckpt"
            ckpt_dir.mkdir(exist_ok=True)
            gs.save(ckpt_dir / f"{it:06d}.pt")
            if train_eval_interval > 0:
                with torch.inference_mode():
                    psnr = eval_rgb_camera_chunks(
                        gs=gs,
                        dataset=dataset,
                        frames=frames,
                        cameras=eval_cameras,
                        metric_fns={"psnr": psnr_fn},
                        frame_interval=eval_frame_interval,
                        num_workers=eval_num_workers,
                        prefetch_factor=eval_prefetch_factor,
                        pin_memory=eval_pin_memory,
                    ).mean()["psnr"]
                    logger.info(f"Checkpoint {it:05d} PSNR {psnr:.2f}")
                    tb_writer.add_scalar("eval/psnr_checkpoint", psnr, it)

        if train_eval_interval > 0 and it % train_eval_interval == 0:
            with torch.inference_mode():
                psnr = eval_rgb_camera_chunks(
                    gs=gs,
                    dataset=dataset,
                    frames=frames,
                    cameras=eval_cameras,
                    metric_fns={"psnr": psnr_fn},
                    frame_interval=eval_frame_interval,
                    num_workers=eval_num_workers,
                    prefetch_factor=eval_prefetch_factor,
                    pin_memory=eval_pin_memory,
                ).mean()["psnr"]
                postfix.update({"psnr": f"{psnr:.2f}"})
                rr.log("psnr", rr.Scalars(psnr))
                logger.info(f"Iteration {it:05d} PSNR {psnr:.2f}")
                tb_writer.add_scalar("eval/psnr", psnr, it)

            rr.log(
                "gt",
                rr.EncodedImage(contents=tensor_to_png(gt[0]), media_type="image/png"),
            )
            rr.log(
                "pred",
                rr.EncodedImage(
                    contents=tensor_to_png(pred[0]), media_type="image/png"
                ),
            )
            rr.log(
                "pred-raw",
                rr.EncodedImage(
                    contents=tensor_to_png(pred_[0]), media_type="image/png"
                ),
            )

        loss_scalars = {
            "loss/total": float(loss.detach()),
            "loss/l1": float(loss_l1.detach()),
            "loss/ssim": float(loss_ssim.detach()),
            "loss/lpips": float(loss_lpips.detach()),
            "loss/reg": float(loss_reg.detach()),
            "loss/reg_opacity": float(loss_reg_opacity.detach()),
            "loss/reg_scale": float(loss_reg_scale.detach()),
            "loss/reg_duration": float(loss_reg_duration.detach()),
            "loss/gate": float(loss_gate.detach()),
            "loss/color_correctors": float(loss_color_correctors.detach()),
            "loss/velocity_distill": float(loss_velocity_distill.detach()),
            "loss/depth": float(loss_depth.detach()),
            "loss/unseen": float(loss_unseen.detach()),
        }
        for tag, value in loss_scalars.items():
            tb_writer.add_scalar(tag, value, it)
        rr.log("loss/total", rr.Scalars(loss_scalars["loss/total"]))
        rr.log("loss/l1", rr.Scalars(loss_scalars["loss/l1"]))
        rr.log("loss/ssim", rr.Scalars(loss_scalars["loss/ssim"]))
        rr.log("loss/lpips", rr.Scalars(loss_scalars["loss/lpips"]))
        rr.log("loss/reg", rr.Scalars(loss_scalars["loss/reg"]))
        rr.log("loss/gate", rr.Scalars(loss_scalars["loss/gate"]))
        rr.log("loss/color_correctors", rr.Scalars(loss_scalars["loss/color_correctors"]))
        rr.log("loss/velocity_distill", rr.Scalars(loss_scalars["loss/velocity_distill"]))

        bar.set_postfix(postfix)

    tb_writer.flush()
    tb_writer.close()

    return gs


def relocate_gaussians(
    state: TrainState,
    gs: Gaussians,
    opacity_threshold: float,
    mode: str,
    score_mode: str,
    score_mode_start: int,
    optimizers: OptimizerCollection,
    binoms: Tensor,
    it: int,
    logger: logging.Logger,
) -> tuple[int, bool]:
    opacities = torch.sigmoid(gs.opacities).flatten()
    opacities_probs = opacities / opacities.sum()
    grad2d = state.grad2d_acc()
    grad2d_probs = grad2d / grad2d.sum()
    probs = opacities_probs + grad2d_probs

    if score_mode == "gate_included":
        gate_scores = (1 - gs.gate().flatten())
        if score_mode_start <= it:
            gate_probs = gate_scores / gate_scores.sum()
            probs += gate_probs
    if (not torch.isfinite(probs).all()):
        probs = torch.ones_like(probs)

    dead_mask = opacities <= opacity_threshold
    n_gs = int(dead_mask.sum().item())
    n_alive = int((~dead_mask).sum().item())
    if n_gs > 0 and n_alive == 0:
        logger.warning(
            "Relocation skipped: alive Gaussian count is zero (dead=%d, total=%d).",
            n_gs,
            len(gs),
        )
        return n_gs, True
    if n_gs > 0:
        logger.debug(
            "Relocation trigger mode=%s score_mode=%s dead=%d",
            mode,
            score_mode,
            n_gs,
        )
        relocate(
            gs,
            optimizers=optimizers,
            probs=probs,
            state=asdict(state),
            mask=dead_mask,
            binoms=binoms,
            min_opacity=opacity_threshold,
            mode=mode,
        )
        logger.debug("Relocation applied relocated=%d", n_gs)

    return n_gs, False


def make_optimizers(
    gs: Gaussians,
    color_correctors: ColorCorrectors | None,
    *,
    lrs: Mapping[str, float],
    lr_schedules: Mapping[str, float],
    batch_size: int,
    iterations: int,
    scene_extent: float,
) -> tuple[OptimizerCollection, LRSchedulerCollection]:
    optimizers = {
        name: torch.optim.Adam(
            [{"params": param, "lr": lrs[name] * batch_size**0.5, "name": name}],
            eps=1e-15 / batch_size**0.5,
            betas=(0.9**batch_size, 0.999**batch_size),
        )
        for name, param in gs.named_parameters(recurse=False)
        if name in lrs and param.requires_grad
    }
    for name, module in gs.named_modules():
        if name in lrs and any(p.requires_grad for p in module.parameters()):
            optimizers[name] = torch.optim.Adam(
                [
                    {
                        "params": module.parameters(),
                        "lr": lrs[name] * batch_size**0.5,
                        "name": name,
                    }
                ],
                eps=1e-15 / batch_size**0.5,
                betas=(0.9**batch_size, 0.999**batch_size),
            )

    if "color_correctors" in lrs and color_correctors is not None:
        optimizers["color_correctors"] = torch.optim.Adam(
            [
                {
                    "params": color_correctors.parameters(),
                    "lr": lrs["color_correctors"] * batch_size**0.5,
                    "name": "color_correctors",
                }
            ],
            eps=1e-15 / batch_size**0.5,
            betas=(0.9**batch_size, 0.999**batch_size),
        )

    if "means" in optimizers:
        optimizers["means"].param_groups[0]["lr"] *= scene_extent
    if "velocity_model" in optimizers:
        optimizers["velocity_model"].param_groups[0]["lr"] *= scene_extent

    lr_schedulers: dict[str, torch.optim.lr_scheduler.LRScheduler] = {
        name: torch.optim.lr_scheduler.ExponentialLR(
            optimizers[name],
            schedule ** (1 / iterations),
        )
        for name, schedule in lr_schedules.items()
        if name in optimizers
    }

    optimizers = OptimizerCollection(optimizers)
    lr_schedulers = LRSchedulerCollection(lr_schedulers)

    return optimizers, lr_schedulers
