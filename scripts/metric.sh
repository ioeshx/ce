TARGET_DIR="./result" # 在这里修改你要遍历的父文件夹路径

for dir in "$TARGET_DIR"/*/; do
    dir=${dir%/} # 移除路径末尾的斜杠
    echo "================================"
    echo "Processing $dir ..."
    
    python util/clip_score_cal.py \
        --contents "Hello Kitty, Mickey, Pikachu, Snoopy, Spongebob" \
        --root_path "${dir}/Snoopy_Mickey_Spongebob" \
        --pretrained_path "${dir}/Snoopy_Mickey_Spongebob"
done