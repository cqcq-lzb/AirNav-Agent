#!/usr/bin/env bash
set -euo pipefail

docker run --rm --gpus all airway-navigation-gpu-backend:2.0 bash -lc '
set -e
/data0/home/wcq/.conda/envs/perm_test/bin/python -c "import torch, nnunetv2, SimpleITK; print(\"nnUNet CUDA\", torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
/data0/home/wcq/.conda/envs/nodulenet_h100/bin/python -c "import torch, SimpleITK; print(\"MANet CUDA\", torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
test -f /data2/home/wcq/projects/MANet-main/results/cross_val_test/Fold_0_mask/model/last.ckpt
test -f /data2/home/wcq/nnUNet/nnUNet_results/Dataset520_AirRC_AirwayLumen/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth
test -f /data2/home/wcq/nnUNet/nav_project/engineering_demo/run_stage4_server_pipeline.py
echo "GPU runtime, models, and pipeline files are present."
'
