#!/usr/bin/env sh

start_seconds=$(date +%s)

export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0

trim_spaces() {
    s="$1"
    s="${s#"${s%%[![:space:]]*}"}"
    s="${s%"${s##*[![:space:]]}"}"
    printf '%s' "$s"
}
coco_csv="data/mscoco.csv"
benchmark_py='/home/shx/code/ce-benchmark/ce-benchmark.py'
prompts_csv='/path/to/prompts.csv'


##### instance #####
# for target_concepts in "Monet"; do
for target_concepts in "Snoopy" "Snoopy, Mickey" "Snoopy, Mickey, Spongebob" "Van Gogh" "Picasso" "Monet"; do
    # target_concepts="Snoopy, Mickey, Spongebob"
    # anchor_concepts=""
    # retain_path="data/instance.csv"
    # contents="Snoopy, Mickey, Spongebob, Pikachu, Hello Kitty"
    if [ "$target_concepts" = "Snoopy" ] || [ "$target_concepts" = "Snoopy, Mickey" ] || [ "$target_concepts" = "Snoopy, Mickey, Spongebob" ] ; then
        anchor_concepts=""
        erase_type="instance"
        retain_path="data/instance.csv"
        contents="Hello Kitty, Mickey, Pikachu, Snoopy, Spongebob"
    elif [ "$target_concepts" = "bed" ] || [ "$target_concepts" = "smartphone" ] || [ "$target_concepts" = "apple" ] || [ "$target_concepts" = "car" ] || [ "$target_concepts" = "book" ]; then
        anchor_concepts=""
        erase_type="object"
        retain_path="data/object.csv"
        contents="bed, smartphone, apple, car, book"
    else
        anchor_concepts="art"
        erase_type="style"
        retain_path="data/style.csv"
        contents="Van Gogh, Picasso, Monet, Paul Gauguin, Caravaggio"
    fi
    
    echo "=======[${erase_type}]========"
    echo "Target Concepts: ${target_concepts}"
    echo "Anchor Concepts: ${anchor_concepts}"
    echo "Contents: ${contents}"

    script_name=$(basename "$0" .sh)
    anchor_slug=$(printf '%s' "$anchor_concepts" | tr ' ' '_')
    save_path="ckpt/${script_name}_${erase_type}"
    sample_save_root="result/${script_name}_${erase_type}"

    ckpt_meta=$(mktemp)

    python erase-my-2.py \
        --target_concepts "${target_concepts}" \
        --anchor_concepts "${anchor_concepts}" \
        --retain_path "${retain_path}" \
        --header "concept" \
        --params V \
        --save_path ${save_path} \
        --ckpt_path_file "${ckpt_meta}" \
        --mapEoT


    edit_ckpt=$(cat "${ckpt_meta}")
    rm -f "${ckpt_meta}"

    echo "[INFO] Running sample.py for instance..."
    python sample.py \
        --erase_type "${erase_type}" \
        --target_concept "${target_concepts}" \
        --contents "${contents}" \
        --edit_ckpt "${edit_ckpt}" \
        --mode 'original, edit' \
        --num_samples 10 --batch_size 10 \
        --save_root ${sample_save_root}

    echo "[INFO] Running sample2.py for coco 1k..."
    python sample2.py \
        --target_concept "${target_concepts}" \
        --contents "coco" \
        --edit_ckpt "${edit_ckpt}" \
        --mode 'original, edit' \
        --batch_size 16 \
        --save_root ${sample_save_root}
        # --coco_max_num 100

    # Expected structure:
    #   ${sample_save_root}/${target_group}/${content}/{original,edit}
    # target_group=$(echo "${target_concepts}" | sed 's/, /_/g; s/,/_/g; s/ /_/g')
    target_group=$(printf '%s' "${target_concepts}" | awk -F',' '
        {
            for (i = 1; i <= NF; i++) {
                gsub(/^[[:space:]]+|[[:space:]]+$/, "", $i)
                parts[i] = $i
            }
            out = parts[1]
            for (i = 2; i <= NF; i++) out = out "_" parts[i]
            print out
        }
    ')


    images_root_candidate="${sample_save_root}/${target_group}"

    if [ ! -d "${images_root_candidate}" ]; then
        echo "[WARN] Not found: ${images_root_candidate}"
        echo "[WARN] Falling back to images root: ${sample_save_root}"
        images_root_candidate="${sample_save_root}"
    fi

    old_ifs="$IFS"
    IFS=','
    set -f
    for raw_content in ${contents}; do
        content=$(trim_spaces "${raw_content}")
        image_root="${images_root_candidate}/${content}/edit"
        fid_ref="${images_root_candidate}/${content}/original"
        lpips_original="${images_root_candidate}/${content}/original"
        lpips_edited="${images_root_candidate}/${content}/edit"
        output_json="${images_root_candidate}/${content}/summary.json"

        if [ ! -d "${lpips_original}" ] || [ ! -d "${lpips_edited}" ]; then
            echo "[WARN] Skip ${content}: missing ${lpips_original} or ${lpips_edited}"
            continue
        fi

        echo "[INFO] Benchmarking content: ${content}"
        python "${benchmark_py}" \
            --metrics lpips aesthetic \
            --images-root "${image_root}" \
            --fid-ref "${fid_ref}" \
            --prompts-csv "${prompts_csv}" \
            --lpips-original "${lpips_original}" \
            --lpips-edited "${lpips_edited}" \
            --output-json "${output_json}" \
            --prompt_from_filename
            # --metrics fid clip lpips aesthetic \
        python util/clip_score_cal.py \
            --contents "${content}" \
            --root_path "${images_root_candidate}/" \
            --pretrained_path "${images_root_candidate}/" \
            --version "openai/clip-vit-large-patch14"
    done
        set +f
        IFS="$old_ifs"


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
            --pretrained_path "${images_root_candidate}/" \
            --version "openai/clip-vit-large-patch14"

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