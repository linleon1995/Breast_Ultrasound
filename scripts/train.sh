gpu_ids=0


CUDA_VISIBLE_DEVICES=$gpu_ids python ./train.py \
    --config ./config/_2dunet_512_config.yml \