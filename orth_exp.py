import os
import re
import argparse
from typing import Dict

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
from tqdm import tqdm
from PIL import Image, ImageDraw
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler

from util.template import template_dict
from util.utils import seed_everything, get_token_id, get_token, get_textencoding, process_img


def diffusion(
    unet,
    scheduler,
    latents,
    text_embeddings,
    total_timesteps,
    start_timesteps=0,
    guidance_scale=7.5,
    show_progress=False,
    desc=None,
):
    scheduler.set_timesteps(total_timesteps)
    for timestep in tqdm(scheduler.timesteps[start_timesteps:total_timesteps], desc=desc, show_progress=show_progress):
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, timestep)

        noise_pred = unet(
            latent_model_input,
            timestep,
            encoder_hidden_states=text_embeddings,
        ).sample

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        latents = scheduler.step(noise_pred, timestep, latents).prev_sample

    return latents


def _sanitize_filename(s: str) -> str:
    return re.sub(r'[^\w\s-]', '', s).strip().replace(' ', '_')


def _decode_latents_to_image(latent, vae) -> Image.Image:
    decoded = vae.decode(latent.unsqueeze(0) / vae.config.scaling_factor, return_dict=False)[0]
    return process_img(decoded)


def _combine_1x3(images: Dict[str, Image.Image], labels: Dict[str, str]) -> Image.Image:
    keys = [
        'target_original',
        'anchor_original',
        'projected',
    ]
    w, h = images[keys[0]].size
    canvas = Image.new('RGB', (w * 3, h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    positions = {
        'target_original': (0, 0),
        'anchor_original': (w, 0),
        'projected': (w * 2, 0),
    }

    for k in keys:
        x, y = positions[k]
        canvas.paste(images[k], (x, y))
        draw.rectangle((x, y, x + w, y + 20), fill=(0, 0, 0))
        draw.text((x + 6, y + 4), labels[k], fill=(255, 255, 255))

    return canvas


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_root', type=str,required=True)
    parser.add_argument('--sd_ckpt', type=str, default='CompVis/stable-diffusion-v1-4')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--erase_type', type=str, default='instance', choices=['instance', 'style', 'celebrity'])
    parser.add_argument('--target', type=str, required=True)
    parser.add_argument('--anchor', type=str, required=True)
    parser.add_argument('--num_templates', type=int, default=80)
    parser.add_argument('--num_samples', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--guidance_scale', type=float, default=7.5)
    parser.add_argument('--total_timesteps', type=int, default=20)
    parser.add_argument('--show_progress', action='store_true', default=False)

    # orthogonal projection related args
    parser.add_argument('--use_concept_as_prompt', action='store_true', default=False)
    parser.add_argument('--extract_concept_tokens', action='store_true', default=False)
    parser.add_argument('--proj_direction', type=str, default='t2a', choices=['t2a', 'a2t'])
    parser.add_argument('--gen_mode', type=str, default='proj_only', choices=['proj_only', 'add_to_anchor'])
    parser.add_argument('--proj_length', type=str, default='all', choices=['all', 'max_valid'])

    args = parser.parse_args()

    assert args.num_samples >= args.batch_size and args.num_samples % args.batch_size == 0, (
        'num_samples should be a multiple of batch_size.'
    )

    print('[Arguments]')
    for key, value in vars(args).items():
        print(f'{key}={value}')

    seed_everything(args.seed, deterministic=True)

    pipe = DiffusionPipeline.from_pretrained(
        args.sd_ckpt,
        safety_checker=None,
        torch_dtype=torch.float16,
    ).to('cuda')
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    unet, tokenizer, text_encoder, vae = pipe.unet, pipe.tokenizer, pipe.text_encoder, pipe.vae
    uncond_embedding = get_textencoding(get_token('', tokenizer), text_encoder)

    if args.use_concept_as_prompt:
        templates = ['{}']
    else:
        templates = template_dict[args.erase_type][:args.num_templates]

    root = os.path.join(
        args.save_root,
        f"{_sanitize_filename(args.target)}_to_{_sanitize_filename(args.anchor)}_proj_{args.proj_direction}_{args.gen_mode}",
    )
    mode_dirs = {
        'target_original': os.path.join(root, 'target_original'),
        'anchor_original': os.path.join(root, 'anchor_original'),
        'projected': os.path.join(root, 'projected'),
        'compare': os.path.join(root, 'compare'),
    }
    for p in mode_dirs.values():
        os.makedirs(p, exist_ok=True)

    labels = {
        'target_original': 'target-original',
        'anchor_original': 'anchor-original',
        'projected': 'projected-embedding',
    }

    # Pre-calculate projection if use_concept_as_prompt
    precalc_t_emb_final = None
    precalc_a_emb_final = None
    precalc_target_emb = None
    precalc_anchor_emb = None

    if args.use_concept_as_prompt:
        tmpl = templates[0]
        target_prompt = tmpl.format(args.target)
        anchor_prompt = tmpl.format(args.anchor)
        
        target_token = get_token_id(target_prompt, tokenizer, False)
        anchor_token = get_token_id(anchor_prompt, tokenizer, False)
        target_emb = get_textencoding(target_token.input_ids, text_encoder)[0]
        anchor_emb = get_textencoding(anchor_token.input_ids, text_encoder)[0]
        
        tar_last_token_idx = target_token.attention_mask[0].sum().item() - 2
        anc_last_token_idx = anchor_token.attention_mask[0].sum().item() - 2

        t_emb_final = target_emb.clone()
        a_emb_final = anchor_emb.clone()
        
        if args.proj_length == 'max_valid':
            max_len = max(target_token.attention_mask[0].sum().item(), anchor_token.attention_mask[0].sum().item())
            t_feat = target_emb[1:max_len]
            a_feat = anchor_emb[1:max_len]
        else:
            t_feat = target_emb[1:]
            a_feat = anchor_emb[1:]
            
        if args.proj_direction == 't2a':
            coeff = t_feat.float() @ a_feat.float().T @ torch.linalg.pinv(a_feat.float() @ a_feat.float().T)
            proj = (coeff @ a_feat.float()).to(t_feat.dtype)
            
            if args.gen_mode == 'proj_only':
                new_feat = proj
            else:
                new_feat = a_feat + proj
                
            if args.proj_length == 'max_valid':
                t_emb_final[1:max_len] = new_feat
            else:
                t_emb_final[1:] = new_feat
        else:
            coeff = a_feat.float() @ t_feat.float().T @ torch.linalg.pinv(t_feat.float() @ t_feat.float().T)
            proj = (coeff @ t_feat.float()).to(t_feat.dtype)
            
            if args.gen_mode == 'proj_only':
                new_feat = proj
            else:
                new_feat = a_feat + proj

            if args.proj_length == 'max_valid':
                a_emb_final[1:max_len] = new_feat
            else:
                a_emb_final[1:] = new_feat
                
        precalc_t_emb_final = t_emb_final
        precalc_a_emb_final = a_emb_final
        precalc_target_emb = target_emb
        precalc_anchor_emb = anchor_emb

    repeats = args.num_samples // args.batch_size
    for rep in range(repeats):
        latents = torch.randn(args.batch_size, 4, 64, 64).to(pipe.device, dtype=pipe.dtype)

        for t_idx, tmpl in enumerate(templates):
            if args.use_concept_as_prompt:
                target_emb = precalc_target_emb
                anchor_emb = precalc_anchor_emb
                t_emb_final = precalc_t_emb_final
                a_emb_final = precalc_a_emb_final
            else:
                target_prompt = tmpl.format(args.target)
                anchor_prompt = tmpl.format(args.anchor)
                
                target_token = get_token_id(target_prompt, tokenizer, False)
                anchor_token = get_token_id(anchor_prompt, tokenizer, False)
                target_emb = get_textencoding(target_token.input_ids, text_encoder)[0]
                anchor_emb = get_textencoding(anchor_token.input_ids, text_encoder)[0]
                
                tar_last_token_idx = target_token.attention_mask[0].sum().item() - 2
                anc_last_token_idx = anchor_token.attention_mask[0].sum().item() - 2

                t_emb_final = target_emb.clone()
                a_emb_final = anchor_emb.clone()
                
                if args.extract_concept_tokens:
                    prefix = tmpl.split('{}')[0]
                    prefix_toks = get_token_id(prefix, tokenizer, False)
                    # number of valid tokens in prefix
                    prefix_len = prefix_toks.attention_mask[0].sum().item() - 2
                    
                    t_start = 1 + prefix_len
                    t_end = tar_last_token_idx + 1
                    a_start = 1 + prefix_len
                    a_end = anc_last_token_idx + 1
                    
                    t_feat_raw = target_emb[t_start:t_end]
                    a_feat_raw = anchor_emb[a_start:a_end]
                    
                    max_concept_len = max(t_feat_raw.shape[0], a_feat_raw.shape[0])
                    
                    if t_feat_raw.shape[0] < max_concept_len:
                        pad_len = max_concept_len - t_feat_raw.shape[0]
                        pad_embs = target_emb[tar_last_token_idx + 2 : tar_last_token_idx + 2 + pad_len]
                        t_feat = torch.cat([t_feat_raw, pad_embs], dim=0)
                    else:
                        t_feat = t_feat_raw
                        
                    if a_feat_raw.shape[0] < max_concept_len:
                        pad_len = max_concept_len - a_feat_raw.shape[0]
                        pad_embs = anchor_emb[anc_last_token_idx + 2 : anc_last_token_idx + 2 + pad_len]
                        a_feat = torch.cat([a_feat_raw, pad_embs], dim=0)
                    else:
                        a_feat = a_feat_raw
                else:
                    t_feat = target_emb[tar_last_token_idx:tar_last_token_idx+1]
                    a_feat = anchor_emb[anc_last_token_idx:anc_last_token_idx+1]
                    
                if args.proj_direction == 't2a':
                    # project target to anchor
                    coeff = t_feat.float() @ a_feat.float().T @ torch.linalg.pinv(a_feat.float() @ a_feat.float().T)
                    proj = (coeff @ a_feat.float()).to(t_feat.dtype)
                    
                    if args.gen_mode == 'proj_only':
                        new_feat = proj
                    else: # add_to_anchor
                        new_feat = a_feat + proj
                        
                    if args.extract_concept_tokens:
                        t_emb_final[t_start:t_end] = new_feat[:t_feat_raw.shape[0]]
                    else:
                        t_emb_final[tar_last_token_idx:tar_last_token_idx+1] = new_feat
                else:
                    # project anchor to target
                    coeff = a_feat.float() @ t_feat.float().T @ torch.linalg.pinv(t_feat.float() @ t_feat.float().T)
                    proj = (coeff @ t_feat.float()).to(t_feat.dtype)
                    
                    if args.gen_mode == 'proj_only':
                        new_feat = proj
                    else: # add_to_anchor
                        new_feat = a_feat + proj

                    if args.extract_concept_tokens:
                        a_emb_final[a_start:a_end] = new_feat[:a_feat_raw.shape[0]]
                    else:
                        a_emb_final[anc_last_token_idx:anc_last_token_idx+1] = new_feat

            target_emb_batched = target_emb.unsqueeze(0)
            anchor_emb_batched = anchor_emb.unsqueeze(0)
            
            # The "projected" embedding will either be taking the place of target or anchor
            if args.proj_direction == 't2a':
                proj_emb_batched = t_emb_final.unsqueeze(0)
            else:
                proj_emb_batched = a_emb_final.unsqueeze(0)

            text_emb_target_original = torch.cat(
                [uncond_embedding] * args.batch_size + [target_emb_batched] * args.batch_size,
                dim=0,
            )
            text_emb_anchor_original = torch.cat(
                [uncond_embedding] * args.batch_size + [anchor_emb_batched] * args.batch_size,
                dim=0,
            )
            text_emb_projected = torch.cat(
                [uncond_embedding] * args.batch_size + [proj_emb_batched] * args.batch_size,
                dim=0,
            )

            save_latents = {
                'target_original': diffusion(
                    unet=unet,
                    scheduler=pipe.scheduler,
                    latents=latents,
                    text_embeddings=text_emb_target_original,
                    total_timesteps=args.total_timesteps,
                    guidance_scale=args.guidance_scale,
                    show_progress=args.show_progress,
                    desc=f'{t_idx} | target original',
                ),
                'anchor_original': diffusion(
                    unet=unet,
                    scheduler=pipe.scheduler,
                    latents=latents,
                    text_embeddings=text_emb_anchor_original,
                    total_timesteps=args.total_timesteps,
                    guidance_scale=args.guidance_scale,
                    show_progress=args.show_progress,
                    desc=f'{t_idx} | anchor original',
                ),
                'projected': diffusion(
                    unet=unet,
                    scheduler=pipe.scheduler,
                    latents=latents,
                    text_embeddings=text_emb_projected,
                    total_timesteps=args.total_timesteps,
                    guidance_scale=args.guidance_scale,
                    show_progress=args.show_progress,
                    desc=f'{t_idx} | projected',
                ),
            }

            for b in range(args.batch_size):
                global_idx = rep * args.batch_size + b
                
                # Format to `prompt_number` to match CE benchmark requirements
                if args.use_concept_as_prompt:
                    # When using concept, the prompt is essentially target
                    prompt_str = args.target
                else:
                    prompt_str = tmpl.format(args.target)
                    
                file_stub = f'{_sanitize_filename(prompt_str)[:80]}_{global_idx:03d}'

                decoded_imgs = {
                    name: _decode_latents_to_image(save_latents[name][b], vae)
                    for name in save_latents
                }

                for mode_name, img in decoded_imgs.items():
                    # For original generation, usually 'target_original' is the reference
                    img.save(os.path.join(mode_dirs[mode_name], f'{file_stub}.png'))

                compare_img = _combine_1x3(decoded_imgs, labels)
                # Keep compare in same format or similar format
                compare_img.save(os.path.join(mode_dirs['compare'], f'{file_stub}.jpg'))

    print(f'[DONE] Saved pilot experiment results to: {root}')


if __name__ == '__main__':
    main()

        


