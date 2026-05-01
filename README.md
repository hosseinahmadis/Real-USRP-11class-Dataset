# Real USRP IQ Dataset (11 classes)

A small PyTorch dataset wrapper around an HDF5 file containing real USRP IQ
recordings across 11 modulation classes (`Real_usrp_39000_11class_raw.h5`).

The data was collected at a single SNR value (**SNR = 10 dB**) with a
transmitter–receiver distance of approximately **5 meters**.

## Download the dataset

The `.h5` file is too large to host on GitHub. Download it from Google Drive:

<<<<<<< HEAD
**Link:** https://drive.google.com/file/d/1JVvdrUe_moXxzDadsngZ478VTlfnu48k/view?usp=drive_link
=======
**Link:** [https://drive.google.com/file/d/1TEjx4_JN4r6YiWmTDbxPyeodIuhNiWgF/view?usp=drive_link](https://drive.google.com/file/d/1JVvdrUe_moXxzDadsngZ478VTlfnu48k/view?usp=drive_link)
>>>>>>> ce8537f2bb9a18622d9de615a62d10b293fcae12

Place the downloaded file in the same folder as `dataset.py` before running anything.

## Contact

For more information, questions, or access requests, please email
**ha168@uakron.edu**.

## Contents

```
.
├── Real_usrp_39000_11class_raw.h5   # the dataset (not in this repo, see below)
├── dataset.py                       # IQDataset class + get_dataloaders helper
├── test_dataset.ipynb               # quick test & visualization notebook
└── README.md
```

The `.h5` file is not tracked in git because of its size — see the
**Download the dataset** section above.

## Requirements

- Python 3.8+
- `numpy`
- `h5py`
- `torch`
- `matplotlib` (only for the notebook)

```bash
pip install numpy h5py torch matplotlib jupyter
```

## HDF5 file layout

The dataset class expects the following keys inside the `.h5` file:

| Key              | Shape       | Description                                  |
|------------------|-------------|----------------------------------------------|
| `iq`             | `(N, L, 2)` | IQ samples, last dim is `[I, Q]`             |
| `label`          | `(N,)`      | Integer class label per sample               |
| `snr`            | `(N,)`      | SNR value per sample                         |
| `identification` | `(N,)`      | Capture / device ID per sample               |
| `iq_min`         | `(2,)`      | Global min for I and Q (for normalization)   |
| `iq_max`         | `(2,)`      | Global max for I and Q (for normalization)   |
| `snr_min`        | scalar      | Minimum SNR in the file                      |
| `snr_max`        | scalar      | Maximum SNR in the file                      |

## Quick start

```python
from dataset import IQDataset, get_dataloaders

# Load the dataset
dataset = IQDataset("Real_usrp_39000_11class_raw.h5")
dataset.summary()

# One sample
sample = dataset[0]
print(sample["iq"].shape)        # (2, L)
print(sample["label"], sample["snr"], sample["identification"])

# Train / val / test DataLoaders (default split: 70 / 10 / 20)
train_loader, val_loader, test_loader = get_dataloaders(dataset, batch_size=64)

for batch in train_loader:
    iq      = batch["iq"]              # (B, 2, L)
    labels  = batch["label"]           # (B,)
    snr     = batch["snr"]             # (B,)
    ids     = batch["identification"]  # (B,)
    break
```

## Optional filters

You can filter by SNR or by `identification` ranges directly in the constructor:

```python
# Keep only samples with SNR in [10, 30)
ds = IQDataset("Real_usrp_39000_11class_raw.h5", snr_range=(10, 30))

# Keep only certain identification ranges
ds = IQDataset(
    "Real_usrp_39000_11class_raw.h5",
    id_ranges=[(1, 200), (250, 300)],
)

# Limit how many raw samples are loaded, and trim the signal length
ds = IQDataset(
    "Real_usrp_39000_11class_raw.h5",
    max_samples=10000,
    signal_length=512,
)
```

## Sample format

Each call to `dataset[i]` returns a dictionary:

| Key              | Type           | Description                       |
|------------------|----------------|-----------------------------------|
| `iq`             | `torch.Tensor` | Shape `(2, L)`, `float32`         |
| `label`          | `int`          | Class label                       |
| `snr`            | `float`        | SNR in dB                         |
| `identification` | `int`          | Capture / device ID               |

By default, IQ values are normalized to `[-1, 1]` using `iq_min` / `iq_max`
stored in the file. Pass `normalize=False` to disable.

## Notebook

Open `test_dataset.ipynb` for an end-to-end check:

1. Inspect raw HDF5 keys and shapes.
2. Load the dataset and print a summary.
3. Plot I/Q waveforms (one per class).
4. Constellation diagrams (I vs Q).
5. Class and SNR distribution bar charts.
6. Examples of SNR / identification filters.
7. Build DataLoaders and inspect one batch.

## Citation

If you use this dataset or code in your research, please cite the paper for
which this dataset was originally collected:

> H. Ahmadi, Y. Zhang, A. Li, T. Li, and Y. Zhang,
> "BadAMC: A Model-Agnostic Digital Backdoor Attack for Automatic Modulation
> Classification in Crowdsourced Platforms,"
> *IEEE Transactions on Networking*, Early Access, Apr. 2026.
> doi: [10.1109/TON.2026.3688147](https://doi.org/10.1109/TON.2026.3688147).

BibTeX:

```bibtex
@article{ahmadi2026badamc,
  title   = {{BadAMC}: A Model-Agnostic Digital Backdoor Attack for Automatic
             Modulation Classification in Crowdsourced Platforms},
  author  = {Ahmadi, Hossein and Zhang, Yan and Li, Ang and Li, Tao and Zhang, Yanchao},
  journal = {IEEE Transactions on Networking},
  year    = {2026},
  note    = {Early Access},
  issn    = {2998-4157},
  doi     = {10.1109/TON.2026.3688147},
  url     = {https://doi.org/10.1109/TON.2026.3688147}
}
```

## License

Specify your license here (e.g. MIT).
