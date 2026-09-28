# Modified by Team Doreen071 for the SIGGRAPH Asia 2026 VVC Sparse-View track (2026-09); see MODIFICATIONS.md.
# FreeTimeGS++
# 2025-2026 Lucas Yunkyu Lee <lucaslee@postech.ac.kr>, SNU VGI Lab

import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler

from ftgspp.data.utils.mvdataset import MVBatch, MVDataset


class MVDatasetViewDataset(Dataset[MVBatch]):
    """Map flat indices to decoded single frame-camera views."""

    def __init__(self, dataset: MVDataset, frame_interval: int = 1):
        if dataset.num_frames <= 0 or dataset.num_cameras <= 0:
            raise ValueError("MVDataset view selection must not be empty")
        if frame_interval <= 0:
            raise ValueError("frame_interval must be greater than zero")
        self.dataset = dataset
        self.frame_positions = range(0, dataset.num_frames, frame_interval)

    def __len__(self) -> int:
        return len(self.frame_positions) * self.dataset.num_cameras

    def __getitem__(self, index: int) -> MVBatch:
        camera_count = self.dataset.num_cameras
        frame_position, camera_index = divmod(int(index), camera_count)
        frame_index = self.frame_positions[frame_position]
        return self.dataset[frame_index, [camera_index]].load()


def _init_image_worker(_worker_id: int) -> None:
    # Avoid multiplying OpenCV's own thread pool by the DataLoader worker count.
    import cv2

    cv2.setNumThreads(1)


def make_mvdataset_view_loader(
    dataset: MVDataset,
    *,
    num_samples: int,
    num_workers: int,
    prefetch_factor: int,
    persistent_workers: bool,
    pin_memory: bool,
    seed: int,
) -> DataLoader[MVBatch]:
    views = MVDatasetViewDataset(dataset)

    sampler_generator = torch.Generator().manual_seed(seed)
    worker_generator = torch.Generator().manual_seed(seed + 1)
    sampler = RandomSampler(
        views,
        replacement=True,
        num_samples=num_samples,
        generator=sampler_generator,
    )

    worker_options = {}
    if num_workers > 0:
        worker_options = {
            "prefetch_factor": prefetch_factor,
            "persistent_workers": persistent_workers,
            "worker_init_fn": _init_image_worker,
            "multiprocessing_context": "spawn",
        }

    return DataLoader(
        views,
        batch_size=None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=worker_generator,
        **worker_options,
    )


def make_mvdataset_eval_loader(
    dataset: MVDataset,
    *,
    frame_interval: int,
    num_workers: int,
    prefetch_factor: int,
    pin_memory: bool,
) -> DataLoader[MVBatch]:
    views = MVDatasetViewDataset(dataset, frame_interval=frame_interval)

    worker_options = {}
    if num_workers > 0:
        worker_options = {
            "prefetch_factor": prefetch_factor,
            "worker_init_fn": _init_image_worker,
            "multiprocessing_context": "spawn",
        }

    return DataLoader(
        views,
        batch_size=None,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        **worker_options,
    )
