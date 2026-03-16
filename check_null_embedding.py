import os
import torch
from diffusers import StableDiffusionPipeline

# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

def check_null_text_embedding():
    print("Loading model...")
    # 使用与 erase.py 中一致的模型作为默认测试
    model_id = "CompVis/stable-diffusion-v1-4"
    pipeline = StableDiffusionPipeline.from_pretrained(model_id)
    
    tokenizer = pipeline.tokenizer
    text_encoder = pipeline.text_encoder
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    text_encoder.to(device)

    # 1. 对空字符串进行 tokenize
    null_prompt = ""
    text_inputs = tokenizer(
        null_prompt, 
        padding="max_length", 
        max_length=tokenizer.model_max_length, 
        truncation=True, 
        return_tensors="pt"
    ).to(device)
    
    print("\n[Input IDs]:")
    print(text_inputs.input_ids)
    print("Shape:", text_inputs.input_ids.shape)
    
    # 2. 获取 text encoder 隐层状态输出
    with torch.no_grad():
        embeddings = text_encoder(text_inputs.input_ids).last_hidden_state
        
    print("\n[Embeddings]:")
    print("Shape:", embeddings.shape)
    
    # 3. 检查是否全0
    is_all_zero = torch.all(embeddings == 0).item()
    print(f"\nIs the embedding matrix completely zeros? => {is_all_zero}")
    
    # 打印部分数据以供观察
    print("\nSnippet of embeddings (first token, first 10 dims):")
    print(embeddings[0, 0, :10])
    
    print("\nSnippet of embeddings (second token, first 10 dims):")
    print(embeddings[0, 1, :10])
    
    # 打印矩阵统计信息
    print(f"\nMax value: {embeddings.max().item():.4f}")
    print(f"Min value: {embeddings.min().item():.4f}")
    print(f"Mean: {embeddings.mean().item():.4f}")
    print(f"L2 Norm: {torch.norm(embeddings).item():.4f}")

if __name__ == "__main__":
    check_null_text_embedding()
