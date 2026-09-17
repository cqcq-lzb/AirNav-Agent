#!/usr/bin/env bash
set -euo pipefail

bundle_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$bundle_dir"

test -f build_context/engineering_demo/run_full_pipeline.py
test -f build_context/manet/results/cross_val_test/Fold_0_mask/model/170.ckpt
test -f build_context/nnUNet_results/Dataset520_AirRC_AirwayLumen/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth

cuda_base_image=${CUDA_BASE_IMAGE:-swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04}
gpu_device_id=${NVIDIA_VISIBLE_DEVICES:-3}
archive=airway-navigation-h100.tar.gz

docker build --build-arg "CUDA_BASE_IMAGE=$cuda_base_image" -t airway-navigation:latest .
docker run --rm --gpus "device=$gpu_device_id" airway-navigation:latest \
  /data0/home/wcq/.conda/envs/perm_test/bin/python -c \
  "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
docker save airway-navigation:latest | gzip -1 > "$archive.partial"
mv "$archive.partial" "$archive"
sha256sum "$archive" > "$archive.sha256"
printf '%s\n' "Exported $bundle_dir/$archive"
