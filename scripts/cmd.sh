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

nohup ./scripts/a2t-avg.sh > ./logs/a2t-avg.log
nohup ./scripts/a2t-sum.sh > ./logs/a2t-sum.log
nohup ./scripts/a2t-only-avg.sh > ./logs/a2t-only-avg.log
nohup ./scripts/a2t-only-sum.sh > ./logs/a2t-only-sum.log

nohup ./scripts/t2a-avg.sh > ./logs/t2a-avg.log
nohup ./scripts/t2a-sum.sh > ./logs/t2a-sum.log
nohup ./scripts/t2a-only-avg.sh > ./logs/t2a-only-avg.log
nohup ./scripts/t2a-only-sum.sh > ./logs/t2a-only-sum.log