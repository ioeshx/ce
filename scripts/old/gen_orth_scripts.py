import os

directions = ['t2a', 'a2t']
gen_modes = ['proj_only', 'add_to_anchor']
proj_lengths = ['all', 'max_valid']

scripts_dir = "/home/shx/code/ce/scripts/orth_experiments"
os.makedirs(scripts_dir, exist_ok=True)

base_template = """#!/usr/bin/env bash

export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0

target="Snoopy"
anchor="dog"

benchmark_py='/home/shx/code/ce-benchmark/ce-benchmark.py'

echo "[INFO] Running Orthogonal Projection Experiment: {script_name}"

python orth_exp.py \\
    --target "${{target}}" \\
    --anchor "${{anchor}}" \\
    --proj_direction "{direction}" \\
    --gen_mode "{gen_mode}" {use_concept_arg}{proj_length_arg}

res_dir="result/robust_PCA_pilot/${{target}}_to_${{anchor}}_proj_{direction}_{gen_mode}"

image_root="${{res_dir}}/projected"
fid_ref="${{res_dir}}/target_original"
lpips_original="${{res_dir}}/target_original"
lpips_edited="${{res_dir}}/projected"
output_json="${{res_dir}}/summary.json"

echo "[INFO] Benchmarking: ${{res_dir}}"
python "${{benchmark_py}}" \\
    --metrics fid clip lpips aesthetic \\
    --images-root "${{image_root}}" \\
    --fid-ref "${{fid_ref}}" \\
    --lpips-original "${{lpips_original}}" \\
    --lpips-edited "${{lpips_edited}}" \\
    --output-json "${{output_json}}" \\
    --prompt_from_filename

echo "[DONE] {script_name}"
"""

for d in directions:
    for g in gen_modes:
        # Template ones
        script_name = f"orth_template_{d}_{g}"
        content = base_template.format(
            script_name=script_name,
            direction=d,
            gen_mode=g,
            use_concept_arg="",
            proj_length_arg=""
        )
        with open(os.path.join(scripts_dir, f"{script_name}.sh"), 'w') as f:
            f.write(content)
        
        # Concept ones
        for pl in proj_lengths:
            script_name = f"orth_concept_{pl}_{d}_{g}"
            content = base_template.format(
                script_name=script_name,
                direction=d,
                gen_mode=g,
                use_concept_arg="\\\n    --use_concept_as_prompt ",
                proj_length_arg=f"\\\n    --proj_length \"{pl}\""
            )
            with open(os.path.join(scripts_dir, f"{script_name}.sh"), 'w') as f:
                f.write(content)

print(f"Generated 12 scripts in {scripts_dir}")
