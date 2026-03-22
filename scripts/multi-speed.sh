#!/usr/bin/env sh
# 擦除来自CIFAR100的大量（50，75，1000）概念，然后用coco测试生成效果

start_seconds=$(date +%s)

export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=1

trim_spaces() {
    s="$1"
    s="${s#"${s%%[![:space:]]*}"}"
    s="${s%"${s##*[![:space:]]}"}"
    printf '%s' "$s"
}
coco_csv="data/mscoco.csv"
benchmark_py='../ce-benchmark/ce-benchmark.py'
prompts_csv='/path/to/prompts.csv'

# target_concepts="imagenet100"
# anchor_concepts=""
# erase_type="instance"
# retain_path="data/instance.csv"
# contents="Snoopy, Mickey, Spongebob, Pikachu, Hello Kitty"
# type="instance"


##### multi #####
for target_concepts in "CIFAR10" "CIFAR30" "CIFAR50" "CIFAR75" "CIFAR100"; do

    anchor_concepts=""
    erase_type="instance"
    retain_path="data/instance.csv"
    contents="coco"
    type="instance"

    echo "=======[${type}]========"
    echo "Target Concepts: ${target_concepts}"
    echo "Anchor Concepts: ${anchor_concepts}"
    echo "Contents: ${contents}"

    script_name=$(basename "$0" .sh)
    anchor_slug=$(printf '%s' "$anchor_concepts" | tr ' ' '_')
    save_path="ckpt/${script_name}_${type}"
    sample_save_root="result/${script_name}_${type}"

    ckpt_meta=$(mktemp)

    python erase-my.py \
        --target_concepts "${target_concepts}" \
        --anchor_concepts "${anchor_concepts}" \
        --retain_path "${retain_path}" \
        --header "concept" \
        --params V \
        --save_path ${save_path} \
        --ckpt_path_file "${ckpt_meta}"

    edit_ckpt=$(cat "${ckpt_meta}")
    rm -f "${ckpt_meta}"

    echo "[INFO] Running sample2.py for coco 1k..."
    python sample2.py \
        --target_concept "${target_concepts}" \
        --contents "coco" \
        --edit_ckpt "${edit_ckpt}" \
        --mode 'original, edit' \
        --batch_size 16 \
        --save_root ${sample_save_root}
        # --coco_max_num 100


    # Expected structure
    #   ${sample_save_root}/${target_group}/${content}/[original,edit]
    target_group=$(echo "${target_concepts}" | sed 's/, /_/g; s/,/_/g; s/ /_/g')
    images_root_candidate="${sample_save_root}/${target_group}"

    if [ ! -d "${images_root_candidate}" ]; then
        echo "[WARN] Not found: ${images_root_candidate}"
        echo "[WARN] Falling back to images root: ${sample_save_root}"
        images_root_candidate="${sample_save_root}"
    fi

    echo "[INFO] Benchmarking coco 1k from sample2.py..."
    coco_content="coco"
    image_root="${images_root_candidate}/${coco_content}/edit"
    fid_ref="${images_root_candidate}/${coco_content}/original"
    lpips_original="${images_root_candidate}/${coco_content}/original"
    lpips_edited="${images_root_candidate}/${coco_content}/edit"
    output_json="${images_root_candidate}/${coco_content}/summary.json"

    if [ ! -d "${lpips_original}" ] || [ ! -d "${lpips_edited}" ]; then
        echo "[WARN] Skip ${coco_content}: missing ${lpips_original} or ${lpips_edited}"
    else
        echo "[INFO] Benchmarking content: ${coco_content}"
        python "${benchmark_py}" \
            --metrics lpips aesthetic \
            --images-root "${image_root}" \
            --lpips-original "${lpips_original}" \
            --lpips-edited "${lpips_edited}" \
            --output-json "${output_json}"

        python util/clip_score_cal.py \
            --contents "coco" \
            --root_path "${images_root_candidate}/" \
            --pretrained_path "${images_root_candidate}/"
    fi
done


end_seconds=$(date +%s)
total_seconds=$((end_seconds - start_seconds))

hours=$((total_seconds / 3600))
minutes=$(( (total_seconds % 3600) / 60 ))
seconds=$((total_seconds % 60))

# 补零（比如把 2 变成 02）
formatted_time=$(printf "%02d:%02d:%02d" $hours $minutes $seconds)

# 打印结果
echo "====================================="
echo "run time: $formatted_time"
echo "====================================="