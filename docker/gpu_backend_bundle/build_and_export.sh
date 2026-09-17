#!/usr/bin/env bash
set -euo pipefail

bundle_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$bundle_dir"

test -f build_context/engineering_demo/run_full_pipeline.py
test -f build_context/manet/results/cross_val_test/Fold_0_mask/model/170.ckpt
test -f build_context/nnUNet_results/Dataset520_AirRC_AirwayLumen/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth

docker compose -f docker-compose.gpu.yml build
docker run --rm --gpus all airway-navigation-gpu-backend:2.0 \
  /data0/home/wcq/.conda/envs/perm_test/bin/python -c \
  "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
docker save -o airway-navigation-gpu-backend-2.0.tar airway-navigation-gpu-backend:2.0
sha256sum airway-navigation-gpu-backend-2.0.tar > airway-navigation-gpu-backend-2.0.tar.sha256
printf '%s\n' "Exported $bundle_dir/airway-navigation-gpu-backend-2.0.tar"
