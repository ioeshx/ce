import argparse
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
from diffusers import StableDiffusionPipeline

def load_sd_pipeline(model_path, dtype, device):
    torch_dtype = torch.float16 if dtype == "fp16" else torch.float32

    # 兼容目录模型和单文件模型（.safetensors / .ckpt）
    if os.path.isfile(model_path) and model_path.endswith((".safetensors", ".ckpt")):
        pipe = StableDiffusionPipeline.from_single_file(
            model_path,
            torch_dtype=torch_dtype,
            safety_checker=None,
        )
    else:
        pipe = StableDiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            safety_checker=None,
        )

    pipe = pipe.to(device)
    return pipe

def get_weight_tensor(maybe_linear):
    # 常见线性层
    if hasattr(maybe_linear, "weight"):
        return maybe_linear.weight
    # 一些 LoRA 包装层
    if hasattr(maybe_linear, "base_layer") and hasattr(maybe_linear.base_layer, "weight"):
        return maybe_linear.base_layer.weight
    return None

def find_cross_attention_kv(unet):
    kv_items = []

    for module_name, module in unet.named_modules():
        # 在 SD UNet 中，attn2 才是 cross-attention
        if module_name.endswith("attn2") and hasattr(module, "to_k") and hasattr(module, "to_v"):
            k_w = get_weight_tensor(module.to_k)
            v_w = get_weight_tensor(module.to_v)

            if k_w is not None and v_w is not None:
                kv_items.append(
                    {
                        "name": module_name,
                        "k_shape": tuple(k_w.shape),
                        "v_shape": tuple(v_w.shape),
                    }
                )

    return kv_items

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="CompVis/stable-diffusion-v1-4", help="")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "fp32"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--print_detail", default=True, action="store_true", help="")
    args = parser.parse_args()

    pipe = load_sd_pipeline(args.model_path, args.dtype, args.device)
    kv_items = find_cross_attention_kv(pipe.unet)

    k_count = len(kv_items)
    v_count = len(kv_items)

    print("=" * 60)
    print("Cross-Attention K/V 统计结果")
    print(f"模型: {args.model_path}")
    print(f"K 矩阵数量: {k_count}")
    print(f"V 矩阵数量: {v_count}")
    print(f"K+V 总数: {k_count + v_count}")
    print("=" * 60)

    if args.print_detail:
        for i, item in enumerate(kv_items, 1):
            print(f"[{i}] {item['name']}")
            print(f"    K shape: {item['k_shape']}, V shape: {item['v_shape']}")

if __name__ == "__main__":
    main()