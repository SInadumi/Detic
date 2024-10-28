#!/usr/bin/env bash

set -euxo pipefail
readonly ROOT_DIR="/mnt/home/shun-i/dev/cl_mmref/data/f30k_ent_jp"

for CONFIG_FILE in "Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size" "Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size"
do
    poetry run python -u tools/extract_region_features.py \
        --root-dir ${ROOT_DIR} \
        --dataset-name f30k_ent_jp \
        --output-file-name ${CONFIG_FILE}_thresh-30 \
        --config-file "configs/${CONFIG_FILE}.yaml" \
        --vocabulary lvis \
        --confidence-threshold 0.3 \
        --opts \
        MODEL.WEIGHTS "models/${CONFIG_FILE}.pth" \
        SEED 10
done

echo "done!"
