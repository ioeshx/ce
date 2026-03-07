save_path='ckpt/only_null_proj'
sample_save_root='result/only_null_proj'
ckpt_meta=$(mktemp)

export CUDA_VISIBLE_DEVICES=0

python erase.py \
    --target_concepts "Snoopy, Mickey, Spongebob" \
    --anchor_concepts "" \
    --retain_path "data/instance.csv" \
    --header "concept" \
    --params V \
    --aug_num 0 \
    --disable_filter \
    --save_path ${save_path} \
    --ckpt_path_file "${ckpt_meta}"

edit_ckpt=$(cat "${ckpt_meta}")
rm -f "${ckpt_meta}"

python sample.py \
    --erase_type 'instance' \
    --target_concept 'Snoopy, Mickey, Spongebob' \
    --contents 'Snoopy, Mickey, Spongebob, Pikachu, Hello Kitty' \
    --edit_ckpt "${edit_ckpt}" \
    --mode 'original, edit' \
    --num_samples 10 --batch_size 10 \
    --save_root ${sample_save_root}