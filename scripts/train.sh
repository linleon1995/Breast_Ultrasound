gpu_ids=0


CUDA_VISIBLE_DEVICES=$gpu_ids python ./train.py \
    --config ./config/_2dunet_seg_train_config.yml \