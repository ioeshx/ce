from random import random
import os
import csv
import math
import argparse
from typing import Dict, List, Tuple

import diffusers
import torch
import numpy as np
from PIL import Image

try:
    from transformers import CLIPModel, CLIPProcessor
except Exception:
    CLIPModel = None
    CLIPProcessor = None

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def _get_text_embs(prompts: List[str], pipeline: diffusers.StableDiffusionPipeline, device: torch.device) -> torch.Tensor:
    tokenizer = pipeline.tokenizer
    text_encoder = pipeline.text_encoder
    inputs = tokenizer(
        prompts,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    hidden = text_encoder(input_ids).last_hidden_state
    last_indices = attention_mask.sum(1) - 2
    batch_idx = torch.arange(hidden.size(0), device=device)
    return hidden[batch_idx, last_indices]


def _parse_list_arg(value: str) -> List[str]:
    if value is None:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _load_retain_prompts(retain_path: str, heads: str, max_num: int) -> List[str]:
    if not retain_path or not heads:
        return []
    heads_list = [h.strip() for h in heads.split(",") if h.strip()]
    retain_prompts: List[str] = []
    with open(retain_path, "r", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            for head in heads_list:
                if head in row and row[head].strip():
                    retain_prompts.append(row[head].strip())
            if max_num and len(retain_prompts) >= max_num:
                break
    return retain_prompts[:max_num] if max_num else retain_prompts


def _null_space_projector(C: torch.Tensor, eps: float) -> torch.Tensor:
    if C.numel() == 0:
        return torch.eye(C.size(1), device=C.device)
    _, S, Vh = torch.linalg.svd(C, full_matrices=True)
    rank = int((S > eps).sum().item())
    if rank >= Vh.size(0):
        return torch.zeros((Vh.size(1), Vh.size(1)), device=C.device)
    N = Vh[rank:].T
    return N @ N.T


def _cosine_scores(matrix: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    matrix_norm = matrix / (matrix.norm(dim=0, keepdim=True) + 1e-8)
    vector_norm = vector / (vector.norm() + 1e-8)
    return (matrix_norm.T @ vector_norm).squeeze(-1)


def _select_topk_columns(U: torch.Tensor, scores: torch.Tensor, top_k: int) -> torch.Tensor:
    if top_k <= 0 or top_k >= U.size(1):
        return U
    idx = torch.topk(scores, k=top_k, largest=True).indices
    return U[:, idx]


def _projector_from_basis(B: torch.Tensor, dim: int) -> torch.Tensor:
    if B.numel() == 0:
        return torch.zeros((dim, dim), device=B.device)
    return B @ B.T


def _fit_clip_to_sd_map(
    pipeline: diffusers.StableDiffusionPipeline,
    prompts: List[str],
    device: torch.device,
    num_images: int,
    seed: int,
    clip_model_id: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if CLIPModel is None or CLIPProcessor is None:
        raise RuntimeError("transformers is required for CLIP features. Please install transformers.")

    clip_model = CLIPModel.from_pretrained(clip_model_id).to(device)
    clip_processor = CLIPProcessor.from_pretrained(clip_model_id)

    images: List[Image.Image] = []
    used_prompts: List[str] = []
    generator = torch.Generator(device=device).manual_seed(seed)
    for prompt in prompts:
        for _ in range(num_images):
            image = pipeline(prompt=prompt, num_inference_steps=25, generator=generator).images[0]
            images.append(image)
            used_prompts.append(prompt)

    text_embs = _get_text_embs(used_prompts, pipeline, device)
    clip_inputs = clip_processor(images=images, return_tensors="pt").to(device)
    with torch.no_grad():
        image_features = clip_model.get_image_features(**clip_inputs)
    image_features = image_features / (image_features.norm(dim=1, keepdim=True) + 1e-8)

    E = text_embs.T
    F = image_features.T
    reg = 1e-4 * torch.eye(F.size(0), device=device)
    B = E @ F.T @ torch.inverse(F @ F.T + reg)
    return B, image_features.mean(0)


def _get_layer_group(layer_name: str) -> str:
    if "down_blocks" in layer_name:
        return "low"
    if "up_blocks" in layer_name or "mid_block" in layer_name:
        return "high"
    return "unknown"


def erase(pipeline: diffusers.StableDiffusionPipeline, args: argparse.Namespace):
    device = pipeline.device
    target_concepts = _parse_list_arg(args.target_concepts)
    guided_concepts = _parse_list_arg(args.guided_concepts)
    if len(guided_concepts) == 1 and len(target_concepts) > 1:
        guided_concepts = guided_concepts * len(target_concepts)
    if len(target_concepts) != len(guided_concepts):
        raise ValueError("target_concepts and guided_concepts must align in length.")

    retain_prompts = []
    # 读取retain prompts的逻辑：如果指定了retain_path和header，就从csv中读取对应列的提示词；如果指定了preserve_concepts，就把这些概念也加入retain prompts；如果两者都没有指定，则retain prompts为空
    if args.retain_path and args.header:
        retain_prompts = _load_retain_prompts(args.retain_path, args.header, args.retain_max)
    if args.preserve_concepts:
        retain_prompts.extend(_parse_list_arg(args.preserve_concepts))

    target_embs = _get_text_embs(target_concepts, pipeline, device)
    anchor_embs = _get_text_embs(guided_concepts, pipeline, device)
    retain_embs = _get_text_embs(retain_prompts, pipeline, device) if retain_prompts else torch.empty(0, target_embs.size(1), device=device)

    C_list = [target_embs[i] - anchor_embs[i] for i in range(len(target_concepts))]
    C = torch.stack(C_list, dim=0)
    I = torch.eye(target_embs.size(1), device=device)

    clip_map = None
    clip_target = None
    if args.use_clip:
        clip_map, clip_target = _fit_clip_to_sd_map(
            pipeline=pipeline,
            prompts=target_concepts,
            device=device,
            num_images=args.clip_samples,
            seed=args.seed,
            clip_model_id=args.clip_model_id,
        )

    edit_dict: Dict[str, torch.Tensor] = {}
    for name, param in pipeline.unet.state_dict().items():
        if "attn2.to_k" not in name and "attn2.to_v" not in name:
            continue

        layer_group = _get_layer_group(name)
        if layer_group == "low":
            threshold = args.low_threshold
            alpha_k = args.low_alpha_k
            beta_k = args.low_beta_k
            alpha_v = args.low_alpha_v
            beta_v = args.low_beta_v
        else:
            threshold = args.high_threshold
            alpha_k = args.high_alpha_k
            beta_k = args.high_beta_k
            alpha_v = args.high_alpha_v
            beta_v = args.high_beta_v

        # 根据SPEED计算的erase weight
        # 分解为null space和orthogonal space两部分，分别乘以不同的系数进行调整
        W = param.to(device)
        T = target_embs.T @ target_embs
        A = anchor_embs.T @ target_embs
        erase_weight = W @ (A - T) @ torch.inverse(I + T)
        delta_base = args.erase_scale * erase_weight

        P_null = _null_space_projector(C, threshold)
        delta_null = delta_base @ P_null
        delta_orth = delta_base @ (I - P_null)

        if "attn2.to_k" in name:
            if retain_embs.numel() == 0:
                combined = target_embs
                retain_part = target_embs[:0]
            else:
                combined = torch.cat([target_embs, retain_embs], dim=0)
                retain_part = retain_embs

            Kall = W @ combined.T
            U, S, Vh = torch.linalg.svd(Kall, full_matrices=False)
            num_target = target_embs.size(0)
            target_scores = Vh[:, :num_target].abs().mean(dim=1)
            Ue = _select_topk_columns(U, target_scores, args.k_topk)
            P_ske = _projector_from_basis(Ue, I.size(0))

            if retain_part.numel() == 0:
                P_skr = I - P_ske
            else:
                Ur = _select_topk_columns(U, Vh[:, num_target:].abs().mean(dim=1), args.k_topk)
                P_skr = _projector_from_basis(Ur, I.size(0))

            delta = (
                alpha_k * (delta_null @ P_ske) +
                beta_k * (delta_orth @ P_ske) +
                args.preserve_scale * (delta_null @ P_skr)
            )
        else:
            if clip_map is None or clip_target is None:
                raise RuntimeError("CLIP mapping is required for V-subspace. Use --use_clip.")

            vgt_sd = clip_map @ clip_target
            vgt_sd = vgt_sd / (vgt_sd.norm() + 1e-8)
            Vvecs = W @ target_embs.T
            scores = _cosine_scores(Vvecs, vgt_sd)
            Uv, _, _ = torch.linalg.svd(Vvecs, full_matrices=False)
            Ve = _select_topk_columns(Uv, scores, args.v_topk)
            P_sve = _projector_from_basis(Ve, I.size(0))
            P_sve_perp = I - P_sve

            delta = (
                alpha_v * (delta_null @ P_sve) +
                beta_v * (delta_orth @ P_sve) +
                args.preserve_scale * (delta_null @ P_sve_perp)
            )

        edit_dict[name] = (W + delta).to(param.device)

    unet_state = pipeline.unet.state_dict()
    unet_state.update(edit_dict)
    pipeline.unet.load_state_dict(unet_state)

    os.makedirs(args.save_path, exist_ok=True)
    edit_path = os.path.join(args.save_path, "edited_unet.pt")
    torch.save(edit_dict, edit_path)

    target_out = os.path.join(args.save_path, "samples", "target")
    retain_out = os.path.join(args.save_path, "samples", "retain")
    os.makedirs(target_out, exist_ok=True)
    os.makedirs(retain_out, exist_ok=True)

    generator = torch.Generator(device=device).manual_seed(args.seed)
    for prompt in target_concepts:
        image = pipeline(prompt=prompt, num_inference_steps=args.sample_steps, generator=generator).images[0]
        image.save(os.path.join(target_out, f"{prompt.replace(' ', '_')}.png"))

    for prompt in retain_prompts[: args.retain_sample_max]:
        image = pipeline(prompt=prompt, num_inference_steps=args.sample_steps, generator=generator).images[0]
        image.save(os.path.join(retain_out, f"{prompt.replace(' ', '_')}.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Base Config
    parser.add_argument('--sd_ckpt', help='base version for stable diffusion', type=str, default='CompVis/stable-diffusion-v1-4')
    parser.add_argument('--save_path', type=str, default="./save/")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--dtype', type=str, choices=['float32', 'float16'], default='float16')
    # Erase Config
    parser.add_argument('--target_concepts', type=str, required=True)
    parser.add_argument('--guided_concepts', type=str, required=True)
    parser.add_argument('--preserve_concepts', type=str, default=None)
    # Hyperparameters
    parser.add_argument('--erase_scale', help='scale to erase concepts', type=float, required=False, default=1)
    parser.add_argument('--preserve_scale', help='scale to preserve concepts', type=float, required=False, default=None)
    parser.add_argument('--retain_path', type=str, default=None) # 指定csv
    parser.add_argument('--header', type=str, default=None)  # 用于指定读取csv的哪些列作为保留提示词
    parser.add_argument('--retain_max', type=int, default=50)   # 从csv中读取的保留提示词的最大数量，超过这个数量就截断，默认50；如果不指定retain_path或heads，则这个参数不起作用
    parser.add_argument('--retain_sample_max', type=int, default=10)    # 从保留提示词中生成的样本图片的最大数量，超过这个数量就截断，默认10；如果不指定retain_path或heads，则这个参数不起作用

    parser.add_argument('--null_eps', type=float, default=1e-6)
    parser.add_argument('--low_threshold', type=float, default=1e-2)
    parser.add_argument('--high_threshold', type=float, default=1e-4)

    parser.add_argument('--k_topk', type=int, default=4)
    parser.add_argument('--v_topk', type=int, default=4)
    parser.add_argument('--low_alpha_k', type=float, default=1.0)
    parser.add_argument('--low_beta_k', type=float, default=0.2)
    parser.add_argument('--high_alpha_k', type=float, default=1.2)
    parser.add_argument('--high_beta_k', type=float, default=0.5)
    parser.add_argument('--low_alpha_v', type=float, default=1.0)
    parser.add_argument('--low_beta_v', type=float, default=0.2)
    parser.add_argument('--high_alpha_v', type=float, default=1.2)
    parser.add_argument('--high_beta_v', type=float, default=0.5)

    parser.add_argument('--use_clip', action='store_true', default=False)
    parser.add_argument('--clip_model_id', type=str, default='openai/clip-vit-base-patch32')
    parser.add_argument('--clip_samples', type=int, default=2)
    parser.add_argument('--sample_steps', type=int, default=25)

    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device(args.device)
    dtype = torch.float16 if args.dtype == 'float16' else torch.float32
    pipeline = diffusers.StableDiffusionPipeline.from_pretrained(args.sd_ckpt, torch_dtype=dtype).to(device)
    if args.preserve_scale is None:
        args.preserve_scale = 0.0

    erase(pipeline, args)
