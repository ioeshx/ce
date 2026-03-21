#!/usr/bin/env sh

export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0

target_concepts="Snoopy, Mickey, Spongebob"
anchor_concepts=""
k=10
boundary_topk=5
retian_path="data/instance.csv"
contents="Snoopy, Mickey, Spongebob, Pikachu, Hello Kitty"

script_name=$(basename "$0" .sh)
anchor_slug=$(printf '%s' "$anchor_concepts" | tr ' ' '_')
save_path="ckpt/${script_name}_${anchor_slug}"
sample_save_root="result/${script_name}_${anchor_slug}"

benchmark_py='/home/shx/code/ce-benchmark/ce-benchmark.py'
# Keep these as needed for your local benchmark assets.
# prompts_csv='/path/to/prompts.csv'

ckpt_meta=$(mktemp)

python erase-origin-elastic.py \
    --target_concepts "${target_concepts}" \
    --anchor_concepts "${anchor_concepts}" \
    --retain_path "${retian_path}" \
    --header "concept" \
    --params V \
    --save_path ${save_path} \
    --ckpt_path_file "${ckpt_meta}" \
    --elastic_calibration \
    --lambda_2 500.0 \

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

trim_spaces() {
    s="$1"
    s="${s#"${s%%[![:space:]]*}"}"
    s="${s%"${s##*[![:space:]]}"}"
    printf '%s' "$s"
}


# Expected structure:
#   ${sample_save_root}/${target_group}/${content}/{original,edit}
target_group=$(echo "${target_concepts}" | sed 's/, /_/g; s/,/_/g; s/ /_/g')
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
    image_root="${images_root_candidate}/${content}/original"
    fid_ref="${images_root_candidate}/${content}/edit"
    lpips_original="${images_root_candidate}/${content}/original"
    lpips_edited="${images_root_candidate}/${content}/edit"
    output_json="${images_root_candidate}/${content}/summary.json"

    if [ ! -d "${lpips_original}" ] || [ ! -d "${lpips_edited}" ]; then
        echo "[WARN] Skip ${content}: missing ${lpips_original} or ${lpips_edited}"
        continue
    fi

    echo "[INFO] Benchmarking content: ${content}"
    python "${benchmark_py}" \
        --metrics clip \
        --images-root "${image_root}" \
        --fid-ref "${fid_ref}" \
        --prompts-csv "${prompts_csv}" \
        --lpips-original "${lpips_original}" \
        --lpips-edited "${lpips_edited}" \
        --output-json "${output_json}" \
        --prompt_from_filename
done
    set +f
    IFS="$old_ifs"