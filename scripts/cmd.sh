python erase.py \
    --target_concepts "Snoopy, Mickey, Spongebob" \
    --anchor_concepts "" \
    --retain_path "data/instance.csv" \
    --header "concept" \
    --params V \
    --aug_num 0 \
    --disable_filter \


CUDA_VISIBLE_DEVICES=0 python sample.py \
    --erase_type 'instance' \
    --target_concept 'Snoopy, Mickey, Spongebob' \
    --contents 'Snoopy, Mickey, Spongebob, Pikachu, Hello Kitty' \
    --edit_ckpt "logs/checkpoints/20260304-122127-Snoopy_Mickey_Spongebob_3-to_null.pt" \
    --mode 'original, edit' \
    --num_samples 10 --batch_size 10 \
    --save_root 'results/mask-naive-k0.8'