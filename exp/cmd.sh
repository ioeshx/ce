python exp/prompt_token_variants.py --replace_mode to_eot --device "cuda:1"

python exp/prompt_token_variants.py --replace_mode concept_span

python exp/prompt_token_variants.py --prompt_template "a photo of {} flying in the sky" \
    --replace_mode single \
    --save_dir "result/ca/single"
python exp/prompt_token_variants.py --prompt_template "a photo of {} flying in the sky" --replace_mode concept_span 
python exp/prompt_token_variants.py --prompt_template "a photo of {} flying in the sky" \
    --replace_mode to_eot \
    --device "cuda:1"