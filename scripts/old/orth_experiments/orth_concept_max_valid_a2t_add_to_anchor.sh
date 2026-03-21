#!/usr/bin/env bash

export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=1

target="Snoopy"
anchor="cartoon character"
save_root="result/orth_max_valid"

benchmark_py='/home/shx/code/ce-benchmark/ce-benchmark.py'

echo "[INFO] Running Orthogonal Projection Experiment: orth_concept_max_valid_a2t_add_to_anchor"

python orth_exp.py \
    --save_root "${save_root}" \
    --target "${target}" \
    --anchor "${anchor}" \
    --proj_direction "a2t" \
    --gen_mode "add_to_anchor" \
    --use_concept_as_prompt \
    --num_samples 100 \
    --proj_length "max_valid"

anchor_slug=$(printf '%s' "$anchor" | tr ' ' '_')
res_dir="${save_root}/${target}_to_${anchor_slug}_proj_a2t_add_to_anchor"

image_root="${res_dir}/projected"
fid_ref="${res_dir}/target_original"
lpips_original="${res_dir}/target_original"
lpips_edited="${res_dir}/projected"
output_json="${res_dir}/summary.json"

echo "[INFO] Benchmarking: ${res_dir}"
python "${benchmark_py}" \
    --metrics fid clip lpips aesthetic \
    --images-root "${image_root}" \
    --fid-ref "${fid_ref}" \
    --lpips-original "${lpips_original}" \
    --lpips-edited "${lpips_edited}" \
    --output-json "${output_json}" \
    --prompt_from_filename

echo "[DONE] orth_concept_max_valid_a2t_add_to_anchor"
