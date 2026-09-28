# Complete backend manifest

This bundle is designed for the production pipeline currently configured on `gpu-node`.

## Pipeline

- Lung-nodule inference: MANet
- Airway inference: nnU-Net v2 Dataset520
- Airway quality control and repair
- Navigation route generation
- UI/stage-4 package generation
- FastAPI upload, status, logs, and package-download endpoints

## Runtime captured from the GPU server

- GPU: NVIDIA H100 PCIe, driver 580.159.04
- nnU-Net environment: Python 3.10.20, PyTorch 2.12.1+cu130, nnUNetv2 2.8.1
- MANet environment: Python 3.10.14, PyTorch 2.4.1, CUDA 12.1

## Required model files

- MANet: `results/cross_val_test/Fold_0_mask/model/170.ckpt` (about 82 MB), exposed as `last.ckpt`
- Airway: Dataset520 `fold_0/checkpoint_final.pth` (about 239 MB), plus `plans.json` and `dataset.json`

Training datasets, old checkpoints, logs, and historical outputs are intentionally excluded.
