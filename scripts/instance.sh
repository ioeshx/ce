CUDA_VISIBLE_DEVICES=0 python train_erase_null.py \
    --target_concepts "Snoopy, Mickey, Spongebob" --anchor_concepts "" \
    --retain_path "data/instance.csv" --heads "concept"