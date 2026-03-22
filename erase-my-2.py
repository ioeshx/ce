# exccesive erase + selective layer update

import os, re, pdb
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import time
import math
from collections import defaultdict
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from kmeans_pytorch import kmeans
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import random

from util.utils import str2bool
from util.template import imagenet_templates, imagenet_templates_extend, imagenet_classes, CIFAR100_classes


def seed_everything(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def build_probe_prompt(base_prompt, concept):
    if not base_prompt:
        return concept
    if '{}' in base_prompt:
        return base_prompt.format(concept)
    return base_prompt

def get_token_id(prompt, tokenizer=None, return_ids_only=True):
    token_ids = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    return token_ids.input_ids if return_ids_only else token_ids


def get_subject_token_indices(tokenized_inputs, use_all_tokens=False):
    valid_len = int(tokenized_inputs.attention_mask[0].sum().item())
    if use_all_tokens:
        return list(range(1, max(valid_len - 1, 1)))
    return [max(valid_len - 2, 0)]


class CrossAttentionScoreProbe:
    class ScoreCaptureProcessor:
        def __init__(self, layer_name, base_processor, token_indices, score_cache, active_getter):
            self.layer_name = layer_name
            self.base_processor = base_processor
            self.token_indices = token_indices
            self.score_cache = score_cache
            self.active_getter = active_getter

        def _prepare_states(self, tensor):
            if tensor.dim() == 4:
                b, c, h, w = tensor.shape
                return tensor.view(b, c, h * w).transpose(1, 2)
            return tensor

        def _compute_score(self, attn, hidden_states, encoder_hidden_states):
            hidden_states = self._prepare_states(hidden_states)
            encoder_hidden_states = self._prepare_states(encoder_hidden_states)

            if getattr(attn, "norm_cross", False):
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

            query = attn.to_q(hidden_states)
            key = attn.to_k(encoder_hidden_states)

            query = attn.head_to_batch_dim(query)
            key = attn.head_to_batch_dim(key)

            scores = torch.bmm(query, key.transpose(-1, -2)) * attn.scale
            probs = torch.softmax(scores, dim=-1)

            max_tokens = probs.shape[-1]
            valid_token_indices = [idx for idx in self.token_indices if 0 <= idx < max_tokens]
            if len(valid_token_indices) == 0:
                return

            token_scores = probs[:, :, valid_token_indices]
            self.score_cache[self.layer_name].append(float(token_scores.mean().item()))

        def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, *args, **kwargs):
            output = self.base_processor(
                attn,
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                temb=temb,
                *args,
                **kwargs,
            )

            if self.active_getter() and encoder_hidden_states is not None and len(self.token_indices) > 0:
                try:
                    self._compute_score(attn, hidden_states, encoder_hidden_states)
                except Exception:
                    # Probe failures should not break model editing.
                    pass

            return output

    def __init__(self, unet, target_token_indices):
        self.unet = unet
        self.target_token_indices = target_token_indices
        self.original_processors = {}
        self.layer_step_cache = defaultdict(list)
        self.layer_history = defaultdict(list)
        self.active = False

    def register(self):
        for module_name, module in self.unet.named_modules():
            if module_name.endswith("attn2") and hasattr(module, "processor"):
                self.original_processors[module_name] = module.processor
                module.processor = self.ScoreCaptureProcessor(
                    layer_name=module_name,
                    base_processor=module.processor,
                    token_indices=self.target_token_indices,
                    score_cache=self.layer_step_cache,
                    active_getter=lambda: self.active,
                )

    def set_active(self, active):
        self.active = active

    def finalize_step(self):
        if not self.active:
            self.layer_step_cache.clear()
            return
        for layer_name, scores in self.layer_step_cache.items():
            if len(scores) > 0:
                self.layer_history[layer_name].append(float(np.mean(scores)))
        self.layer_step_cache.clear()

    def restore(self):
        for module_name, module in self.unet.named_modules():
            if module_name in self.original_processors:
                module.processor = self.original_processors[module_name]
        self.original_processors.clear()


@torch.no_grad()
def probe_attention_scores(args, pipeline, prompt, token_indices, device="cuda"):
    scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    scheduler.set_timesteps(args.probe_steps)
    timesteps = scheduler.timesteps

    start_idx = int(len(timesteps) * args.window_start_ratio)
    end_idx = int(len(timesteps) * args.window_end_ratio)
    end_idx = max(end_idx, start_idx + 1)

    tokenized = get_token_id(prompt, pipeline.tokenizer, return_ids_only=False)
    text_embeddings = pipeline.text_encoder(tokenized.input_ids.to(device)).last_hidden_state
    latents = torch.randn(
        1,
        pipeline.unet.in_channels,
        pipeline.unet.config.sample_size,
        pipeline.unet.config.sample_size,
        device=device,
        dtype=pipeline.unet.dtype,
    )

    if args.probe_seed >= 0:
        gen = torch.Generator(device=device).manual_seed(args.probe_seed)
        latents = torch.randn(
            1,
            pipeline.unet.in_channels,
            pipeline.unet.config.sample_size,
            pipeline.unet.config.sample_size,
            generator=gen,
            device=device,
            dtype=pipeline.unet.dtype,
        )

    probe = CrossAttentionScoreProbe(pipeline.unet, token_indices)
    probe.register()
    try:
        for step_idx, timestep in enumerate(timesteps):
            in_window = start_idx <= step_idx < end_idx
            probe.set_active(in_window)
            latent_model_input = scheduler.scale_model_input(latents, timestep)
            noise_pred = pipeline.unet(
                latent_model_input,
                timestep,
                encoder_hidden_states=text_embeddings,
            ).sample
            latents = scheduler.step(noise_pred, timestep, latents).prev_sample
            probe.finalize_step()
    finally:
        probe.restore()

    if args.temporal_agg == 'max':
        layer_scores = {k: max(v) for k, v in probe.layer_history.items() if len(v) > 0}
    else:
        layer_scores = {k: float(np.mean(v)) for k, v in probe.layer_history.items() if len(v) > 0}

    return layer_scores


@torch.no_grad()
def build_dynamic_layer_mask(args, pipeline, target_concepts, anchor_concepts, device="cuda"):

    prompt_template = "a photo of {}." # imagenet_templates[0] if args.probe_prompt_a == '' else args.probe_prompt_a
    per_layer_scores = defaultdict(list)

    for concept_idx, target_concept in enumerate(target_concepts):
        prompt_a = build_probe_prompt(prompt_template, target_concept)
        prompt_a_tokens = get_token_id(prompt_a, pipeline.tokenizer, return_ids_only=False)
        token_idx_a = get_subject_token_indices(prompt_a_tokens, use_all_tokens=args.target_all_tokens)
        scores_a = probe_attention_scores(args, pipeline, prompt_a, token_idx_a, device=device)

        if args.score_mode == 'residual':
            anchor_concept = anchor_concepts[concept_idx] if concept_idx < len(anchor_concepts) else anchor_concepts[0]
            prompt_b = build_probe_prompt(prompt_template, anchor_concept)
            prompt_b_tokens = get_token_id(prompt_b, pipeline.tokenizer, return_ids_only=False)
            token_idx_b = get_subject_token_indices(prompt_b_tokens, use_all_tokens=args.target_all_tokens)
            scores_b = probe_attention_scores(args, pipeline, prompt_b, token_idx_b, device=device)
            all_layers = set(scores_a.keys()) | set(scores_b.keys())
            concept_scores = {layer: abs(scores_a.get(layer, 0.0) - scores_b.get(layer, 0.0)) for layer in all_layers}
        else:
            concept_scores = scores_a

        for layer_name, score in concept_scores.items():
            per_layer_scores[layer_name].append(float(score))

    raw_scores = {layer_name: float(np.mean(scores)) for layer_name, scores in per_layer_scores.items() if len(scores) > 0}
    print(f"[Dynamic Mask] Averaged probing over {len(target_concepts)} target concepts.")
    return raw_scores


def build_layer_gammas(edit_dict, raw_scores, mask_strategy='top_k', topk_ratio=0.7):
    layer_names = sorted({k.rsplit('.to_', 1)[0] for k in edit_dict.keys()})
    if len(layer_names) == 0:
        return {}

    scored_pairs = [(name, float(raw_scores.get(name, 0.0))) for name in layer_names]
    gammas = {}

    if mask_strategy == 'min_max':
        values = [x[1] for x in scored_pairs]
        s_min, s_max = min(values), max(values)
        denom = max(s_max - s_min, 1e-12)
        for name, score in scored_pairs:
            gammas[name] = (score - s_min) / denom
    else:
        k = max(1, int(math.ceil(len(layer_names) * topk_ratio)))
        ranked = sorted(scored_pairs, key=lambda x: x[1], reverse=True)
        selected = {name for name, _ in ranked[:k]}
        for name, _ in ranked:
            gammas[name] = 1.0 if name in selected else 0.0

    return gammas

# Speed专属
def generate_perturbed_embs(ret_embs, P, erase_weight, num_per_sample, mini_batch=8):
    ret_embs = ret_embs.squeeze(1)
    out_embs, norm_list = [], []
    for i in range(0, ret_embs.size(0), mini_batch):
        mini_ret_embs = ret_embs[i:i + mini_batch]
        for _ in range(num_per_sample):
            noise = torch.randn_like(mini_ret_embs)
            perturbed_embs = mini_ret_embs + noise @ P
            out_embs.append(perturbed_embs)
            norm_list.append(torch.matmul(perturbed_embs, erase_weight.T).norm(dim=1))
    out_embs = torch.cat(out_embs, dim=0)
    norm_list = torch.cat(norm_list, dim=0)
    return out_embs[norm_list > norm_list.mean()].unsqueeze(1) # shape: [Num, 1, 768]


@torch.no_grad()
def edit_model(args, pipeline, target_concepts, anchor_concepts, retain_texts, baseline=None, chunk_size=128, emb_size=768, device="cuda"):

    # decide which matrix to edit and prepare null inputs
    I = torch.eye(emb_size, device=device)
    if args.params == 'KV':
        edit_dict = {k: v for k, v in pipeline.unet.state_dict().items() if 'attn2.to_k' in k or 'attn2.to_v' in k}
    elif args.params == 'V':
        edit_dict = {k: v for k, v in pipeline.unet.state_dict().items() if 'attn2.to_v' in k}
    elif args.params == 'K':
        edit_dict = {k: v for k, v in pipeline.unet.state_dict().items() if 'attn2.to_k' in k}

    if args.enable_dynamic_mask:
        print("[Dynamic Mask] Probing cross-attention scores in golden window...")
        raw_scores = build_dynamic_layer_mask(
            args=args,
            pipeline=pipeline,
            target_concepts=target_concepts,
            anchor_concepts=anchor_concepts,
            device=device,
        )
        print("raw_scores: ", raw_scores)
        layer_gamma = build_layer_gammas(
            edit_dict=edit_dict,
            raw_scores=raw_scores,
            mask_strategy=args.mask_strategy,
            topk_ratio=args.mask_topk_ratio,
        )
        active_layer_num = sum(1 for x in layer_gamma.values() if x > 0)
        print(f"[Dynamic Mask] Probed layers: {len(raw_scores)}, editable layers: {len(layer_gamma)}, active layers: {active_layer_num}")
    else:
        layer_gamma = {k.rsplit('.to_', 1)[0]: 1.0 for k in edit_dict.keys()}

    if baseline in ['SPEED']:
        null_inputs = get_token_id('', pipeline.tokenizer, return_ids_only=False)
        null_hidden = pipeline.text_encoder(null_inputs.input_ids.to(device)).last_hidden_state[0] # [0] index -> [77, 768]
        cluster_ids, cluster_centers = kmeans(X=null_hidden[1:], num_clusters=3, distance='euclidean', device='cuda')
        K2 = torch.cat([null_hidden[[0], :], cluster_centers.to(device)], dim=0).T # shape: [768, 4]
        I2 = torch.eye(len(K2.T), device=device) # shape: [4, 4]
        print("k2 shape: ", K2.shape)
        print("I2 shape: ", I2.shape)
    else:
        raise ValueError("Invalid baseline")

    # region [Target and Anchor]
    # nudity 使用所有token，其他概念使用last subject token
    # 擦除多个概念时，取它们的平均作为最终的擦除目标；如果是单个概念，则直接使用该概念的文本嵌入作为擦除目标
    sum_anchor_target, sum_target_target = [], []
    all_target_embs_list = []
    all_anchor_embs_list = [] # 【新增】专门记录 anchor
    for i in range(0, len(target_concepts)):
        target_inputs = get_token_id(target_concepts[i], pipeline.tokenizer, return_ids_only=False)
        target_embs = pipeline.text_encoder(target_inputs.input_ids.to(device)).last_hidden_state[0] # [77, 768]
        anchor_inputs = get_token_id(anchor_concepts[i], pipeline.tokenizer, return_ids_only=False)
        anchor_embs = pipeline.text_encoder(anchor_inputs.input_ids.to(device)).last_hidden_state[0] # [77, 768]
        if target_concepts == ['nudity']:
            target_embs = target_embs[1:, :]  # all tokens
            anchor_embs = anchor_embs[1:, :]  # all tokens
        elif args.target_all_tokens:
            print("Using all tokens for target.")
            target_embs = target_embs[0:(target_inputs.attention_mask[0].sum().item() - 1), :]  # all subject tokens [num_valid_tokens, 768]
        else:
            print("Using last subject token for target and anchor.")
            target_embs = target_embs[[(target_inputs.attention_mask[0].sum().item() - 2)], :]  # last subject token [1,768]
        
        if args.zero_anchor:
            print(f"Enable Anchor-Free Zeroing for concept: {target_concepts[i]}")
            anchor_embs = torch.zeros_like(target_embs)
        else:
            anchor_embs = anchor_embs[[(anchor_inputs.attention_mask[0].sum().item() - 2)], :]  # last subject token
        
        if args.mapping2context:
            print("Mapping to prompt context.")
            current_template = imagenet_templates_extend if args.anchor_using_extend  else imagenet_templates
            for j in range(0, len(current_template)):
                anchor_prompt = current_template[j].format(anchor_concepts[i])
                anchor_inputs = get_token_id(anchor_prompt, pipeline.tokenizer, return_ids_only=False)
                anchor_embs = pipeline.text_encoder(anchor_inputs.input_ids.to(device)).last_hidden_state[0] # [77, 768]
                if target_concepts == ['nudity']:
                    anchor_embs = anchor_embs[1:, :]  # all tokens
                if args.target_all_tokens:
                    anchor_embs = anchor_embs[0:(anchor_inputs.attention_mask[0].sum().item() - 2), :].mean(dim=0, keepdim=True)
                    anchor_embs = anchor_embs.repeat(target_embs.size(0), 1)  # repeat to match target_embs shape for later calculations
                else:
                    anchor_embs = anchor_embs[0:(anchor_inputs.attention_mask[0].sum().item() - 2), :].mean(dim=0, keepdim=True)  # average of all subject tokens [1,768]

        all_target_embs_list.append(target_embs)
        all_anchor_embs_list.append(anchor_embs) # 【新增】
        sum_target_target.append(target_embs.T @ target_embs)
        sum_anchor_target.append(anchor_embs.T @ target_embs)
    ## 这里用了平均，shape of both: [768, 768]
    sum_target_target, sum_anchor_target = torch.stack(sum_target_target).mean(0), torch.stack(sum_anchor_target).mean(0)
    # 【新增】构建 C1 与 C_null 矩阵
    # 按照公式，将单个 [1, 768] 的特征堆叠翻转成 [768, num_targets] 以进行矩阵乘法
    C1 = torch.cat(all_target_embs_list, dim=0).T 
    C_null = torch.cat(all_anchor_embs_list, dim=0).T
    # endregion


    # region [Retain]
    # retain为空，则使用一个全零输入的文本嵌入作为retain；否则使用retain文本中最后一个subject token的文本嵌入作为retain
    last_ret_embs = []
    retain_texts = [text for text in retain_texts if not any(re.search(r'\b' + re.escape(concept.lower()) + r'\b', text.lower()) for concept in target_concepts)]
    # print("retain texts: ", retain_texts)
    assert len(retain_texts) + len(target_concepts) == len(set(retain_texts + target_concepts)) # 保证不重合
    for j in range(0, len(retain_texts), chunk_size):
        ret_inputs = get_token_id(retain_texts[j:j + chunk_size], pipeline.tokenizer, return_ids_only=False)
        ret_embs = pipeline.text_encoder(ret_inputs.input_ids.to(device)).last_hidden_state # [chunk_size, 77, 768]
        if retain_texts == ['']:
            last_ret_embs.append(ret_embs[:, 1:, :].permute(1, 0, 2))   # shape: [76, 1, 768]
        else:
            last_subject_indices = ret_inputs.attention_mask.sum(1) - 2
            last_ret_embs.append(ret_embs[torch.arange(ret_embs.size(0)), last_subject_indices].unsqueeze(1)) # shape: [chunk_size, 1, 768]
    last_ret_embs = torch.cat(last_ret_embs)
    last_ret_embs = last_ret_embs[torch.randperm(last_ret_embs.size(0))]  # shuffle, shape [total_retain_num, 1, 768]


    if args.aug_num > 0 and not args.disable_filter:
        print("Enable IPF and DPA")
    else:
        print("No IPF or DPA")
    
    for (layer_name, layer_weight) in tqdm(edit_dict.items(), desc="Model Editing"):

        erase_weight = layer_weight @ (sum_anchor_target - sum_target_target) @ (I + sum_target_target).inverse()        
        (U0, S0, V0) = torch.svd(layer_weight)
        P0_min = V0[:, -1:] @ V0[:, -1:].T
        # INFLUENCE-BASED PRIOR FILTERING
        if not args.disable_filter:
            print("Enable IPF")
            weight_norm_init = torch.matmul(last_ret_embs.squeeze(1), erase_weight.T).norm(dim=1)
            layer_ret_embs = last_ret_embs[weight_norm_init > weight_norm_init.mean()]
        else:
             layer_ret_embs = last_ret_embs
        layer_ret_embs = last_ret_embs
        
        # retain矩阵也是取平均的方式进行编辑
        sum_ret_ret, valid_num = [], 0
        print("Enable DPA with aug_num =", args.aug_num)
        for j in range(0, len(layer_ret_embs), chunk_size):
            chunk_ret_embs = layer_ret_embs[j:j + chunk_size]
            if args.aug_num > 0:
                # Directed Prior Augmentation (DPA) --- 只对retain矩阵进行增强
                chunk_ret_embs = torch.cat(
                    [chunk_ret_embs, generate_perturbed_embs(chunk_ret_embs, P0_min, erase_weight, num_per_sample=args.aug_num)], dim=0
                )
            valid_num += chunk_ret_embs.shape[0]
            sum_ret_ret.append(chunk_ret_embs.squeeze(1).T @ chunk_ret_embs.squeeze(1))
        sum_ret_ret = torch.stack(sum_ret_ret, dim=0).sum(0) / valid_num

        U, S, V = torch.svd(sum_ret_ret)
        P = U[:, S < args.threshold] @ U[:, S < args.threshold].T
        M = (sum_target_target @ P + args.retain_scale * I).inverse()

        if baseline == 'SPEED':      
            delta_weight = layer_weight @ (sum_anchor_target - sum_target_target) @ P @ (I - M @ K2 @ (K2.T @ P @ M @ K2 + args.lamb * I2).inverse() @ K2.T @ P) @ M
            
            if args.elastic_calibration:
                print("Enable SPEED with Elastic Calibration") 
                # 1. 照常计算：计算出带有 IEC 保护的基础更新矩阵 Delta_SPEED
                delta_SPEED = layer_weight @ (sum_anchor_target - sum_target_target) @ P @ (I - M @ K2 @ (K2.T @ P @ M @ K2 + args.lamb * I2).inverse() @ K2.T @ P) @ M
                
                # 2. 阻力感知：获得残余能量常数 K_IEC 
                # 这里的 lambda_1 用现有的 retain_scale 替代
                res_term = delta_SPEED @ C1 + layer_weight @ (C1 - C_null)
                K_IEC = torch.norm(res_term, p='fro')**2 + args.retain_scale * torch.norm(delta_SPEED, p='fro')**2
                
                # 3. 智能离合：计算弹性系数，并回乘获得最终的 Delta_weight
                alpha_star = args.elastic_scale / (K_IEC + args.elastic_scale)
                delta_weight = alpha_star * delta_SPEED
                
                # 打印监测日志（观察每次更新被拉高了多少阻力）
                print(f"Layer {layer_name} - K_IEC: {K_IEC.item():.4f}, Alpha: {alpha_star.item():.4f}")
        elif baseline == 'my':
            # 我的实现禁用IPF和DPA，直接使用最基本的投影公式
            # $$\Delta P = W(C_*C_1^\top - C_1C_1^\top) P (C_1C_1^\top P + I)^{-1}$$
            print("Enable My Method")
            delta_weight = layer_weight @ (sum_anchor_target - sum_target_target) @ P @ (sum_target_target @ P + I).inverse()

        gamma = layer_gamma.get(layer_name.rsplit('.to_', 1)[0], 0.0)
        delta_weight = delta_weight * gamma
        
        edit_dict[layer_name] = layer_weight + delta_weight

    print(f"Current model status: Edited {str(target_concepts)} into {str(anchor_concepts)}")
    return edit_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Base Config
    parser.add_argument('--sd_ckpt', help='base version for stable diffusion', type=str, default='CompVis/stable-diffusion-v1-4')
    parser.add_argument('--save_path', type=str, default=None)
    parser.add_argument('--file_name', type=str, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--disable_fixed_seed', default=False, action='store_true')
    parser.add_argument('--ckpt_path_file', type=str, default=None)
    # Erase Config
    parser.add_argument('--target_concepts', type=str, required=True)
    parser.add_argument('--anchor_concepts', type=str, required=True)
    parser.add_argument('--retain_path', type=str, default=None)
    parser.add_argument('--header', type=str, default=None)
    parser.add_argument('--baseline', type=str, default='SPEED')
    # Hyperparameters
    parser.add_argument('--params', type=str, default='V')
    parser.add_argument('--aug_num', type=int, default=10)
    parser.add_argument('--threshold', type=float, default=1e-1)
    parser.add_argument('--retain_scale', type=float, default=1.0) # not used, acts as lambda_1
    parser.add_argument('--lamb', type=float, default=0.0)  # not used
    parser.add_argument('--disable_filter', action='store_true', default=False)

    # target token length
    parser.add_argument('--target_all_tokens', action='store_true', default=False, help="using all tokens for target concept(s) instead of just the last subject token")
    # elastic
    parser.add_argument('--elastic_calibration', action='store_true', default=False, help="Whether to enable Elastic Calibration")
    parser.add_argument('--elastic_scale', type=float, default=10.0) # 【新增】弹性超参 elastic_scale
    # mapping to context
    parser.add_argument('--mapping2context', action='store_true', default=False, help="Whether to map the anchor concept to prompt context")
    parser.add_argument('--anchor_using_extend', action='store_true', default=False)
    # zero_anchor
    parser.add_argument('--zero_anchor', action='store_true', default=False, help="Whether to use a zero vector as the anchor concept representation (only valid when mapping2context is False)")
    # attention-score-based dynamic masking
    parser.add_argument('--enable_dynamic_mask', action='store_true', default=False, help="Enable attention-map based layer scoring and masking")
    parser.add_argument('--score_mode', type=str, default='absolute', choices=['absolute', 'residual'])
    parser.add_argument('--probe_steps', type=int, default=20, help="Total denoising steps used by probing")
    parser.add_argument('--window_start_ratio', type=float, default=0.3, help="Golden window start ratio in [0,1)")
    parser.add_argument('--window_end_ratio', type=float, default=0.6, help="Golden window end ratio in (0,1]")
    parser.add_argument('--temporal_agg', type=str, default='mean', choices=['mean', 'max'])
    parser.add_argument('--mask_strategy', type=str, default='top_k', choices=['top_k', 'min_max'])
    parser.add_argument('--mask_topk_ratio', type=float, default=0.9, help="Top-k ratio when mask_strategy=top_k")
    parser.add_argument('--probe_seed', type=int, default=0, help="Random seed for probing; set -1 to disable fixed probe seed")

    args = parser.parse_args()
    print("[Arguments]")
    for key, value in vars(args).items():
        print(f"{key}={value}")

    assert 0.0 <= args.window_start_ratio < args.window_end_ratio <= 1.0, "window ratios must satisfy 0 <= start < end <= 1"
    assert 0.0 < args.mask_topk_ratio <= 1.0, "mask_topk_ratio should be in (0, 1]"
    
    device = torch.device("cuda")
    if args.disable_fixed_seed:
        print("====Warning: Fixed seed is disabled.====")
    else:
        seed_everything(args.seed)
    
    if "CIFAR" in args.target_concepts:
        print("Using CIFAR-100 classes as target concepts.")
        num_classes = int(args.target_concepts.replace("CIFAR", ""))
        target_concepts = CIFAR100_classes[:num_classes]
    else:
        target_concepts = [con.strip() for con in args.target_concepts.split(',')]
    anchor_concepts = args.anchor_concepts
    retain_path = args.retain_path
    
    file_suffix = "_".join(target_concepts[:5]) + f"_{len(target_concepts)}"  # The filename only displays the first 5 target concepts in multi-concept erasure
    anchor_concepts = [x.strip() for x in anchor_concepts.split(',')]
    if len(anchor_concepts) == 1:
        anchor_concepts = anchor_concepts * len(target_concepts)
        if anchor_concepts[0] == "":
            file_suffix += '-to_null'
        else:
            file_suffix += f'-to_{anchor_concepts[0]}'
    else:
        assert len(target_concepts) == len(anchor_concepts), "length of anchor_concepts should be 1 or the same as target_concepts"
        file_suffix += f'-to_{anchor_concepts[0]}_etc'

    retain_texts = []
    if retain_path is not None:
        assert retain_path.endswith('.csv')
        df = pd.read_csv(retain_path)
        for head in args.header.split(','):
            retain_texts += df[head.strip()].unique().tolist()
    else:
        retain_texts.append("")

    pipeline = StableDiffusionPipeline.from_pretrained(args.sd_ckpt).to(device)

    edit_dict = edit_model(
        args=args,
        pipeline=pipeline, 
        target_concepts=target_concepts, 
        anchor_concepts=anchor_concepts, 
        retain_texts=retain_texts, 
        baseline=args.baseline, 
        device=device, 
    )

    save_path = args.save_path or "logs/checkpoints"
    file_name = args.file_name or f"{time.strftime('%Y%m%d-%H%M%S')}-{file_suffix}"
    os.makedirs(save_path, exist_ok=True)
    ckpt_path = os.path.join(save_path, f"{file_name}.pt")
    torch.save(edit_dict, ckpt_path)
    
    if args.ckpt_path_file:
        with open(args.ckpt_path_file, 'w') as f:
            f.write(ckpt_path)
    print(f"[CKPT_PATH]{ckpt_path}")
    

