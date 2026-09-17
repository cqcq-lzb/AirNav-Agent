#!/usr/bin/env bash
set -euo pipefail

bundle_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
context_dir="$bundle_dir/build_context"
engineering_source=/data2/home/wcq/nnUNet/nav_project/engineering_demo
manet_source=/data2/home/wcq/projects/MANet-main
results_source=/data2/home/wcq/nnUNet/nnUNet_results
timestamp=$(date +%Y%m%d_%H%M%S)

if [[ -e "$context_dir" ]]; then
  mv "$context_dir" "${context_dir}.backup_${timestamp}"
fi

engineering_target="$context_dir/engineering_demo"
manet_target="$context_dir/manet"
results_target="$context_dir/nnUNet_results"
mkdir -p "$engineering_target" "$manet_target" "$results_target"

engineering_files=(
  airway_qc.py
  demo_model_config.json
  optimizer_config.json
  optimize_segmentations.py
  prepare_stage4_case_package.py
  run_case_inference.py
  run_case_navigation.py
  run_case_navigation_branchaware_v4.py
  run_case_navigation_candidates_v3.py
  run_case_navigation_multicost_v2.py
  run_case_qc.py
  run_full_pipeline.py
  run_optimized_nnunet.py
  run_stage4_server_pipeline.py
  target_guided_airway_repair.py
)

for name in "${engineering_files[@]}"; do
  test -f "$engineering_source/$name"
  cp -a "$engineering_source/$name" "$engineering_target/$name"
done

cp -a "$manet_source/run_manet_navigation_inference.py" "$manet_target/"
cp -a "$manet_source/scripts" "$manet_target/"
find "$manet_target" -type d -name __pycache__ -prune -exec rm -rf {} +
find "$manet_target" -type f \( -name '*.pyc' -o -name '*.bak*' -o -name '*.before_*' \) -delete
find "$manet_target" -type f -name '.___init__.py' -delete

manet_model_dir="$manet_target/results/cross_val_test/Fold_0_mask/model"
mkdir -p "$manet_model_dir"
manet_checkpoint=$(readlink -f "$manet_source/results/cross_val_test/Fold_0_mask/model/last.ckpt")
test -f "$manet_checkpoint"
cp -a "$manet_checkpoint" "$manet_model_dir/170.ckpt"
ln -s 170.ckpt "$manet_model_dir/last.ckpt"

relative_model=Dataset520_AirRC_AirwayLumen/nnUNetTrainer__nnUNetPlans__3d_fullres
nnunet_source="$results_source/$relative_model"
nnunet_target="$results_target/$relative_model"
mkdir -p "$nnunet_target/fold_0"
cp -a "$nnunet_source/dataset.json" "$nnunet_target/"
cp -a "$nnunet_source/plans.json" "$nnunet_target/"
cp -a "$nnunet_source/fold_0/checkpoint_final.pth" "$nnunet_target/fold_0/"

printf '%s\n' "Prepared complete backend build context: $context_dir"
du -sh "$context_dir"
