#!/usr/bin/env bash

set -euxo pipefail
readonly ROOT_DIR="path/to/cl_mmref/data/jcre3"

for CONFIG_FILE in "Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size" "Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size"
do
    poetry run python -u tools/extract_region_features.py \
        --root-dir ${ROOT_DIR} \
        --dataset-name jcre3 \
        --output-file-name ${CONFIG_FILE} \
        --config-file "configs/${CONFIG_FILE}.yaml" \
        --vocabulary lvis \
        --confidence-threshold 0 \
        --nms-threshold 0.5 \
        --opts \
        MODEL.WEIGHTS "models/${CONFIG_FILE}.pth" \
        SEED 10
done

echo "done!"
