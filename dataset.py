"""
Dataset class for IQ signal data stored in HDF5 format
(e.g. Real_usrp_39000_11class_raw.h5).

The HDF5 file is expected to contain:
  - 'iq'             : (N, L, 2) array of IQ samples (I and Q channels).
  - 'label'          : (N,) integer class labels.
  - 'snr'            : (N,) or (N, 1) SNR value per sample.
  - 'identification' : (N,) integer capture / device IDs.
  - 'iq_min'         : (2,) global min for I and Q (used for normalization).
  - 'iq_max'         : (2,) global max for I and Q (used for normalization).
  - 'snr_min'        : scalar minimum SNR in the dataset.
  - 'snr_max'        : scalar maximum SNR in the dataset.

Typical use:

    from dataset import IQDataset, get_dataloaders

    ds = IQDataset("Real_usrp_39000_11class_raw.h5")
    ds.summary()
    train_loader, val_loader, test_loader = get_dataloaders(ds, batch_size=64)
"""

import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset


# Edit these to match your real class names if you have them.
DEFAULT_CLASS_NAMES = [
    "class_0", "class_1", "class_2", "class_3", "class_4", "class_5",
    "class_6", "class_7", "class_8", "class_9", "class_10",
]


class IQDataset(Dataset):
    """Minimal IQ dataset reader for Real_usrp_*.h5 files."""

    def __init__(
        self,
        h5_path,
        snr_range=None,
        id_ranges=None,
        max_samples=None,
        signal_length=None,
        normalize=True,
        class_names=None,
    ):
        """
        Args:
            h5_path:       Path to the .h5 file.
            snr_range:     Tuple (snr_min, snr_max). Keeps samples where
                           snr_min <= snr < snr_max. None disables filtering.
            id_ranges:     List of inclusive (start, end) ranges on the
                           'identification' field, e.g. [(1, 200), (250, 300)].
                           None disables filtering.
            max_samples:   Optional cap on number of raw samples loaded.
            signal_length: Optional override of the time-axis length.
                           Defaults to the full length stored in the file.
            normalize:     If True, scale IQ to [-1, 1] using stored iq_min/iq_max.
            class_names:   Optional list of class name strings.
        """
        assert os.path.exists(h5_path), f"Dataset not found at: {h5_path}"
        self.h5_path = h5_path
        self.normalize = normalize
        self.class_names = class_names or DEFAULT_CLASS_NAMES

        with h5py.File(h5_path, "r") as f:
            n = f["iq"].shape[0]
            if max_samples is not None:
                n = min(n, max_samples)

            self.iq = f["iq"][:n]                              # (N, L, 2)
            self.labels = f["label"][:n]
            self.snr = np.squeeze(f["snr"][:n])
            self.identification = f["identification"][:n]
            self.iq_min = f["iq_min"][:]
            self.iq_max = f["iq_max"][:]
            self.snr_min = f["snr_min"][()]
            self.snr_max = f["snr_max"][()]

        # Make 1-D integer / float arrays
        self.labels = np.asarray(self.labels).astype(np.int64).reshape(-1)
        self.identification = np.asarray(self.identification).astype(np.int64).reshape(-1)
        self.snr = np.asarray(self.snr).astype(np.float32).reshape(-1)

        # Optional filters
        if snr_range is not None:
            self._filter_by_snr(*snr_range)
        if id_ranges is not None:
            self._filter_by_id_ranges(id_ranges)

        full_len = self.iq.shape[1]
        self.signal_length = signal_length or full_len
        assert self.signal_length <= full_len, (
            f"signal_length={self.signal_length} > stored length {full_len}"
        )

    # ---------- filtering ---------- #

    def _filter_by_snr(self, snr_min, snr_max):
        mask = (self.snr >= snr_min) & (self.snr < snr_max)
        self._apply_mask(mask)

    def _filter_by_id_ranges(self, id_ranges):
        mask = np.zeros_like(self.identification, dtype=bool)
        for start, end in id_ranges:
            mask |= (self.identification >= start) & (self.identification <= end)
        self._apply_mask(mask)

    def _apply_mask(self, mask):
        self.iq = self.iq[mask]
        self.labels = self.labels[mask]
        self.snr = self.snr[mask]
        self.identification = self.identification[mask]

    # ---------- normalization ---------- #

    def _normalize(self, iq):
        # iq shape: (2, L); iq_min / iq_max shape: (2,)
        iq_min = self.iq_min.reshape(2, 1)
        iq_max = self.iq_max.reshape(2, 1)
        iq = (iq - iq_min) / (iq_max - iq_min)
        return iq * 2.0 - 1.0  # scale to [-1, 1]

    # ---------- torch Dataset interface ---------- #

    def __len__(self):
        return self.iq.shape[0]

    def __getitem__(self, index):
        # Stored as (L, 2); convert to (2, L) so channel dim comes first.
        x = self.iq[index].T.astype(np.float32)          # (2, L)
        x = x[:, : self.signal_length]
        if self.normalize:
            x = self._normalize(x)

        return {
            "iq": torch.from_numpy(x),                   # shape (2, L)
            "label": int(self.labels[index]),
            "snr": float(self.snr[index]),
            "identification": int(self.identification[index]),
        }

    # ---------- summary ---------- #

    def summary(self):
        lines = [
            f"File: {self.h5_path}",
            f"Samples (after filters): {len(self)}",
            f"Signal length used: {self.signal_length} "
            f"(full length in file: {self.iq.shape[1]})",
            f"IQ min  (I, Q): ({self.iq_min[0]:.4f}, {self.iq_min[1]:.4f})",
            f"IQ max  (I, Q): ({self.iq_max[0]:.4f}, {self.iq_max[1]:.4f})",
            f"SNR range in file: [{self.snr_min}, {self.snr_max}]",
            f"Identification range: "
            f"[{int(self.identification.min())}, {int(self.identification.max())}]",
            "",
            "Class distribution:",
        ]
        labels, counts = np.unique(self.labels, return_counts=True)
        for lbl, c in zip(labels, counts):
            name = self.class_names[lbl] if lbl < len(self.class_names) else f"class_{lbl}"
            lines.append(f"  [{lbl}] {name}: {c}")

        lines.append("")
        lines.append("SNR distribution:")
        snrs, counts = np.unique(self.snr, return_counts=True)
        for s, c in zip(snrs, counts):
            lines.append(f"  SNR {s}: {c}")

        text = "\n".join(lines)
        print(text)
        return text


def get_dataloaders(
    dataset,
    batch_size=64,
    train_ratio=0.7,
    val_ratio=0.1,
    shuffle=True,
    num_workers=0,
    seed=42,
):
    """
    Split a dataset into train / val / test DataLoaders.

    Args:
        dataset:      An IQDataset (or any torch Dataset).
        batch_size:   Batch size for all loaders.
        train_ratio:  Fraction used for training.
        val_ratio:    Fraction used for validation. Test gets the rest.
        shuffle:      Shuffle indices before splitting + shuffle train loader.
        num_workers:  Workers for DataLoader.
        seed:         Random seed for the index shuffle.
    """
    assert 0 < train_ratio < 1 and 0 < val_ratio < 1
    assert train_ratio + val_ratio < 1, "train_ratio + val_ratio must be < 1"

    n = len(dataset)
    indices = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    train_end = int(train_ratio * n)
    val_end = train_end + int(val_ratio * n)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    test_set = Subset(dataset, test_idx)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    print(f"Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}")
    return train_loader, val_loader, test_loader
