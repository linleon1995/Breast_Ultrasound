gpu_ids=0


CUDA_VISIBLE_DEVICES=$gpu_ids python ./classification.py \
    --config ./config/_2dunet_cls_train_config.yml \


CUDA_VISIBLE_DEVICES=$gpu_ids python ./classification.py \
    --config ./config/_2dunet_cls_train_config.yml \


CUDA_VISIBLE_DEVICES=$gpu_ids python ./classification.py \
    --config ./config/_2dunet_cls_train_config.yml \