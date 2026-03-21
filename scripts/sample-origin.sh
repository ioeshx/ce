#!/usr/bin/env sh

export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=1

target_concepts="only_origin"

script_name=$(basename "$0" .sh)
anchor_slug=$(printf '%s' "$anchor_concepts" | tr ' ' '_')
sample_save_root="result/${script_name}_${anchor_slug}"



python sample.py \
    --erase_type 'instance' \
    --target_concept "${target_concepts}" \
    --contents 'Snoopy, Mickey, Spongebob, Pikachu, Hello Kitty' \
    --mode 'original' \
    --num_samples 10 --batch_size 10 \
    --save_root ${sample_save_root}


python sample.py \
    --erase_type 'style' \
    --target_concept "${target_concepts}" \
    --contents 'Van Gogh, Picasso, Monet, Paul Gauguin, Caravaggio' \
    --mode 'original' \
    --num_samples 10 --batch_size 10 \
    --save_root ${sample_save_root}

python sample2.py \
    --contents 'coco' \
    --target_concept "${target_concepts}" \
    --mode 'original' \
    --num_samples 10 --batch_size 10 \
    --save_root ${sample_save_root}
