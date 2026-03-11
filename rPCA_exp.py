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
from util.rpca import robust_pca_target_anchor
from util.utils import seed_everything, get_token_id, get_token, get_textencoding, process_img


def diffusion(
    unet,
    scheduler,
    latents,
    text_embeddings,
    total_timesteps,
    start_timesteps=0,
    guidance_scale=7.5,
    desc=None,
):
    scheduler.set_timesteps(total_timesteps)
    for timestep in tqdm(scheduler.timesteps[start_timesteps:total_timesteps], desc=desc):
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


def _combine_2x2(images: Dict[str, Image.Image], labels: Dict[str, str]) -> Image.Image:
    keys = [
        'target_original',
        'target_rpca',
        'anchor_original',
        'anchor_rpca',
    ]
    w, h = images[keys[0]].size
    canvas = Image.new('RGB', (w * 2, h * 2), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    positions = {
        'target_original': (0, 0),
        'target_rpca': (w, 0),
        'anchor_original': (0, h),
        'anchor_rpca': (w, h),
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
    parser.add_argument('--save_root', type=str, default='result/robust_PCA_pilot')
    parser.add_argument('--sd_ckpt', type=str, default='CompVis/stable-diffusion-v1-4')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--erase_type', type=str, default='instance', choices=['instance', 'style', 'celebrity'])
    parser.add_argument('--target', type=str, required=True)
    parser.add_argument('--anchor', type=str, required=True)
    parser.add_argument('--num_templates', type=int, default=20)
    parser.add_argument('--num_samples', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--guidance_scale', type=float, default=7.5)
    parser.add_argument('--total_timesteps', type=int, default=20)
    parser.add_argument('--rpca_lam', type=float)
    parser.add_argument('--rpca_max_iter', type=int, default=1000)
    parser.add_argument('--rpca_tol', type=float, default=1e-7)
    parser.add_argument('--enable_full_prompt', action='store_true', default=False)

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

    templates = template_dict[args.erase_type][:args.num_templates]

    root = os.path.join(
        args.save_root,
        f"{_sanitize_filename(args.target)}_to_{_sanitize_filename(args.anchor)}",
    )
    mode_dirs = {
        'target_original': os.path.join(root, 'target_original'),
        'target_rpca': os.path.join(root, 'target_rpca'),
        'anchor_original': os.path.join(root, 'anchor_original'),
        'anchor_rpca': os.path.join(root, 'anchor_rpca'),
        'compare': os.path.join(root, 'compare'),
    }
    for p in mode_dirs.values():
        os.makedirs(p, exist_ok=True)

    labels = {
        'target_original': 'target-original',
        'target_rpca': 'target-rpca-full-prompt',
        'anchor_original': 'anchor-original',
        'anchor_rpca': 'anchor-rpca-full-prompt',
    }

    repeats = args.num_samples // args.batch_size
    for rep in range(repeats):
        latents = torch.randn(args.batch_size, 4, 64, 64).to(pipe.device, dtype=pipe.dtype)

        for t_idx, tmpl in enumerate(templates):
            target_prompt = tmpl.format(args.target)
            anchor_prompt = tmpl.format(args.anchor)
            # Keep [77, 768] so RPCA is computed on the full prompt embedding.
            target_token = get_token_id(target_prompt, tokenizer, False)
            anchor_token = get_token_id(anchor_prompt, tokenizer, False)
            target_emb = get_textencoding(target_token.input_ids, text_encoder)[0]
            anchor_emb = get_textencoding(anchor_token.input_ids, text_encoder)[0]
            # Perform RPCA on the last token embedding
            tar_last_token_idx = target_token.attention_mask[0].sum().item() - 2
            anc_last_token_idx = anchor_token.attention_mask[0].sum().item() - 2
            t_emb = target_emb[tar_last_token_idx].unsqueeze(0).clone()
            a_emb = anchor_emb[anc_last_token_idx].unsqueeze(0).clone()
            if args.enable_full_prompt:
                t_emb = target_emb.clone()
                a_emb = anchor_emb.clone()

                
            rpca_result = robust_pca_target_anchor(
                t_emb,
                a_emb,
                lam=args.rpca_lam,
                max_iter=args.rpca_max_iter,
                tol=args.rpca_tol,
                device=str(pipe.device),
                dtype=torch.float32,
            )

            t_emb_rpca = rpca_result['L_target'].to(target_emb.dtype)
            a_emb_rpca = rpca_result['L_anchor'].to(anchor_emb.dtype)
            print("After RPCA")
            rel_target = torch.norm(t_emb - t_emb_rpca) / (torch.norm(t_emb) + 1e-12)
            rel_anchor = torch.norm(a_emb - a_emb_rpca) / (torch.norm(a_emb) + 1e-12)
            cos_target = torch.cosine_similarity(
                t_emb.reshape(1, -1),
                t_emb_rpca.reshape(1, -1),
                dim=1,).item()
            cos_anchor = torch.cosine_similarity(
                a_emb.reshape(1, -1),
                a_emb_rpca.reshape(1, -1),
                dim=1,).item()
            print(f"Rel abs: target={rel_target.item():.6f}, anchor={rel_anchor.item():.6f}")
            print(f"Cosine sim: target={cos_target:.6f}, anchor={cos_anchor:.6f}")
            
            t_emb_final = target_emb.clone()
            t_emb_final[tar_last_token_idx] = t_emb_rpca
            a_emb_final = anchor_emb.clone()
            a_emb_final[anc_last_token_idx] = a_emb_rpca
            if args.enable_full_prompt:
                t_emb_final= t_emb_rpca
                a_emb_final = a_emb_rpca

            target_emb_batched = target_emb.unsqueeze(0)
            anchor_emb_batched = anchor_emb.unsqueeze(0)
            target_emb_rpca_batched = t_emb_final.unsqueeze(0)
            anchor_emb_rpca_batched = a_emb_final.unsqueeze(0)

            text_emb_target_original = torch.cat(
                [uncond_embedding] * args.batch_size + [target_emb_batched] * args.batch_size,
                dim=0,
            )
            text_emb_target_rpca = torch.cat(
                [uncond_embedding] * args.batch_size + [target_emb_rpca_batched] * args.batch_size,
                dim=0,
            )
            text_emb_anchor_original = torch.cat(
                [uncond_embedding] * args.batch_size + [anchor_emb_batched] * args.batch_size,
                dim=0,
            )
            text_emb_anchor_rpca = torch.cat(
                [uncond_embedding] * args.batch_size + [anchor_emb_rpca_batched] * args.batch_size,
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
                    desc=f'{t_idx} | target original',
                ),
                'target_rpca': diffusion(
                    unet=unet,
                    scheduler=pipe.scheduler,
                    latents=latents,
                    text_embeddings=text_emb_target_rpca,
                    total_timesteps=args.total_timesteps,
                    guidance_scale=args.guidance_scale,
                    desc=f'{t_idx} | target rpca',
                ),
                'anchor_original': diffusion(
                    unet=unet,
                    scheduler=pipe.scheduler,
                    latents=latents,
                    text_embeddings=text_emb_anchor_original,
                    total_timesteps=args.total_timesteps,
                    guidance_scale=args.guidance_scale,
                    desc=f'{t_idx} | anchor original',
                ),
                'anchor_rpca': diffusion(
                    unet=unet,
                    scheduler=pipe.scheduler,
                    latents=latents,
                    text_embeddings=text_emb_anchor_rpca,
                    total_timesteps=args.total_timesteps,
                    guidance_scale=args.guidance_scale,
                    desc=f'{t_idx} | anchor rpca',
                ),
            }

            for b in range(args.batch_size):
                global_idx = rep * args.batch_size + b
                file_stub = f'{t_idx:03d}_{_sanitize_filename(tmpl)[:80]}_{global_idx:03d}'

                decoded_imgs = {
                    name: _decode_latents_to_image(save_latents[name][b], vae)
                    for name in save_latents
                }

                for mode_name, img in decoded_imgs.items():
                    img.save(os.path.join(mode_dirs[mode_name], f'{file_stub}.png'))

                compare_img = _combine_2x2(decoded_imgs, labels)
                compare_img.save(os.path.join(mode_dirs['compare'], f'{file_stub}.jpg'))

    print(f'[DONE] Saved pilot experiment results to: {root}')


if __name__ == '__main__':
    main()

# import diffusers
# import torch

# from util.template import template_dict


# if __name__ == "__main__":
#     model_id = "CompVis/stable-diffusion-v1-4"
#     pipe = diffusers.StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
#     pipe.to("cuda")
    
#     target_prompt = template_dict['instance']['target'].format(concept='snoopy')
#     anchor_prompt = template_dict['instance']['anchor'].format(concept='')

#     target_inputs = pipe.tokenizer(target_prompt, padding="max_length", max_length=pipe.tokenizer.model_max_length, truncation=True, return_tensors="pt")
#     anchor_inputs = pipe.tokenizer(anchor_prompt, padding="max_length", max_length=pipe.tokenizer.model_max_length, truncation=True, return_tensors="pt")
#     with torch.no_grad():
#         target_embs = pipe.text_encoder(target_inputs.input_ids.to(pipe.text_encoder.device)).last_hidden_state[0]
#         anchor_embs = pipe.text_encoder(anchor_inputs.input_ids.to(pipe.text_encoder.device)).last_hidden_state[0]

#         tar_last_token = target_inputs[[(target_inputs.attention_mask[0].sum().item() - 2)], :]
#         anc_last_token = anchor_inputs[[(anchor_inputs.attention_mask[0].sum().item() - 2)], :]

#         from util.rpca import robust_pca_target_anchor
#         rpca_result = robust_pca_target_anchor(tar_last_token, anc_last_token)
#         print("R-PCA Result:")
#         target_embs[target_inputs.attention_mask[0].sum().item() - 2, :] = rpca_result['L_target'].to(pipe.text_encoder.device)
#         anchor_embs[anchor_inputs.attention_mask[0].sum().item() - 2, :] = rpca_result['L_anchor'].to(pipe.text_encoder.device)

        


