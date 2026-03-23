import torch
import clip
from util.template import imagenet_templates
import random

word = "dog"

# 初始化clip的tokenizer
from clip.simple_tokenizer import SimpleTokenizer
tokenizer = SimpleTokenizer()

# 选取两个不同的模板并格式化句子
i = random.randint(0, len(imagenet_templates) - 1)
j = random.randint(0, len(imagenet_templates) - 1)
# sentence1 = imagenet_templates[i].format(word)
# sentence2 = imagenet_templates[j].format(word)
sentence1 = ""
sentence2 = ""

# 获取Token IDs (提取第0个batch的数据)
tokens1 = clip.tokenize(sentence1)[0]
tokens2 = clip.tokenize(sentence2)[0]

def print_tokens(sentence, tokens):
    print(f"Sentence: '{sentence}'")
    print(f"{'Token ID':<10} | {'Token String':<20}")
    print("-" * 35)
    for token_id in tokens:
        token_id = token_id.item()
        # if token_id == 0:  # 跳过padding token (<|endoftext|>后的0)
        #     continue
        # 从decoder获取对应的字符
        token_str = tokenizer.decoder.get(token_id, '')
        print(f"{token_id:<10} | '{token_str}'")
    print("=" * 50)

print_tokens(sentence1, tokens1)
print_tokens(sentence2, tokens2)

    