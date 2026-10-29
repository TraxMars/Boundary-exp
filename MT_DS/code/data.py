"""Local 3D datasets and the original labeled-first two-stream sampling recipe."""
import hashlib
import itertools
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


SPECS = {
    "LA": dict(directory="LA", patch=(112, 112, 80), labels=8,
               rotation=True, validation="test", stride=(18, 18, 4)),
    "Pancreas": dict(directory="Pancreas", patch=(96, 96, 96), labels=6,
                     rotation=False, validation="test", stride=(16, 16, 16)),
    "BraTS19": dict(directory="brats19/Flods", patch=(96, 96, 96), labels=25,
                    rotation=True, validation="val", stride=(16, 16, 16)),
}


def read_ids(root, split):
    ids = [line.strip() for line in (Path(root) / f"{split}.list").read_text().splitlines()
           if line.strip()]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError(f"Empty or duplicated IDs in {root}/{split}.list")
    return ids


def case_path(root, dataset, case):
    root = Path(root)
    if dataset == "LA":
        return root / "2018LA_Seg_TrainingSet" / case / "mri_norm2.h5"
    if dataset == "Pancreas":
        # The local archive contains normalized volumes with an explicit suffix.
        return root / "Pancreas_h5" / f"{case}_norm.h5"
    return root / f"{case}.h5"


def load_case(path):
    with h5py.File(path, "r") as handle:
        image = handle["image"][:].astype(np.float32)
        label = handle["label"][:].astype(np.int64)
    if image.ndim != 3 or image.shape != label.shape:
        raise ValueError(f"Expected matching single-channel 3D image/label: {path}")
    if not np.isfinite(image).all() or not np.isin(label, [0, 1]).all():
        raise ValueError(f"Nonfinite image or nonbinary label: {path}")
    return image, label


def split_manifest(root, dataset, labelnum):
    records = {}
    for split in ("train", "val", "test"):
        path = Path(root) / f"{split}.list"
        if path.is_file():
            ids = read_ids(root, split)
            missing = [str(case_path(root, dataset, case)) for case in ids
                       if not case_path(root, dataset, case).is_file()]
            if missing:
                raise FileNotFoundError(f"Missing {split} volumes: {missing[:5]}")
            records[split] = dict(path=str(path), ids=ids,
                                  sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    train = records["train"]["ids"]
    for split in ("val", "test"):
        if split in records and set(train) & set(records[split]["ids"]):
            raise ValueError(f"Train/{split} overlap")
    validation = SPECS[dataset]["validation"]
    return dict(dataset=dataset, labeled_ids=train[:labelnum],
                unlabeled_ids=train[labelnum:], splits=records,
                validation_split=validation,
                validation_test_same_list=validation == "test")


class VolumeDataset(Dataset):
    def __init__(self, root, dataset, patch_size, boundary_cache=None, labeled_count=0):
        self.root, self.dataset = Path(root), dataset
        self.ids = read_ids(root, "train")
        self.patch_size = tuple(patch_size)
        self.boundary_cache = Path(boundary_cache) if boundary_cache is not None else None
        self.labeled_count = labeled_count

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        image, label = load_case(case_path(self.root, self.dataset, self.ids[index]))
        band = None
        if self.boundary_cache is not None:
            band = (np.load(self.boundary_cache / f"{index:03d}.npy", allow_pickle=False)
                    if index < self.labeled_count else np.zeros(label.shape, dtype=np.uint8))
            if band.shape != label.shape:
                raise ValueError("Cached band shape does not match full-volume label")
        if SPECS[self.dataset]["rotation"]:
            k = np.random.randint(0, 4)
            axis = np.random.randint(0, 2)
            image = np.flip(np.rot90(image, k), axis=axis).copy()
            label = np.flip(np.rot90(label, k), axis=axis).copy()
            if band is not None:
                band = np.flip(np.rot90(band, k), axis=axis).copy()
        # Preserve the original crop padding and uniform random crop behavior.
        if any(size <= patch for size, patch in zip(label.shape, self.patch_size)):
            pads = [max((patch - size) // 2 + 3, 0)
                    for size, patch in zip(label.shape, self.patch_size)]
            padding = [(pad, pad) for pad in pads]
            image, label = np.pad(image, padding), np.pad(label, padding)
            if band is not None:
                band = np.pad(band, padding)
        starts = [np.random.randint(0, size - patch + 1)
                  for size, patch in zip(label.shape, self.patch_size)]
        crop = tuple(slice(start, start + patch)
                     for start, patch in zip(starts, self.patch_size))
        result = dict(image=torch.from_numpy(np.ascontiguousarray(image[crop][None])),
                      label=torch.from_numpy(np.ascontiguousarray(label[crop])))
        if band is not None:
            result["boundary_band"] = torch.from_numpy(np.ascontiguousarray(band[crop]))
        return result


class TwoStreamBatchSampler(Sampler):
    def __init__(self, labeled, unlabeled, batch_size, labeled_bs):
        self.labeled, self.unlabeled = labeled, unlabeled
        self.labeled_bs, self.unlabeled_bs = labeled_bs, batch_size - labeled_bs
        if min(self.labeled_bs, self.unlabeled_bs) <= 0:
            raise ValueError("A batch must contain labeled and unlabeled cases")
        if len(labeled) < self.labeled_bs or len(unlabeled) < self.unlabeled_bs:
            raise ValueError("Not enough cases for the requested two-stream batch")

    def __iter__(self):
        labeled = iter(np.random.permutation(self.labeled))
        unlabeled = itertools.chain.from_iterable(
            np.random.permutation(self.unlabeled) for _ in itertools.count())
        return (tuple(left) + tuple(right) for left, right in zip(
            zip(*[labeled] * self.labeled_bs), zip(*[unlabeled] * self.unlabeled_bs)))

    def __len__(self):
        return len(self.labeled) // self.labeled_bs
