import os
import argparse
import re
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
from PIL import Image, ImageDraw
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

from template import imagenet_templates


def build_prompt_embeds(pipe, prompt: str):
    toks = pipe.tokenizer(
        prompt,
        padding="max_length",
        truncation=True,
        max_length=pipe.tokenizer.model_max_length,
        return_tensors="pt",
    )
    input_ids = toks.input_ids.to(pipe.device)
    attn_mask = toks.attention_mask.to(pipe.device)
    embeds = pipe.text_encoder(input_ids)[0]
    return embeds, attn_mask, input_ids


def find_subsequence_start(sequence, subseq):
    if len(subseq) == 0 or len(subseq) > len(sequence):
        return -1
    for i in range(len(sequence) - len(subseq) + 1):
        if sequence[i : i + len(subseq)] == subseq:
            return i
    return -1


def replace_concept_token_with_prefix_avg(
    embeds: torch.Tensor,
    attn_mask: torch.Tensor,
    prompt_ids: torch.Tensor,
    concept_ids,
):
    # CLIP token layout: [BOS] content... [EOS] [PAD] ...
    valid_len = int(attn_mask[0].sum().item())
    content_start = 1
    content_end = max(content_start, valid_len - 1)  # exclude EOS
    if content_end <= content_start:
        return embeds.clone(), -1

    content_ids = prompt_ids[0, content_start:content_end].tolist()
    start_in_content = find_subsequence_start(content_ids, concept_ids)
    if start_in_content < 0:
        return embeds.clone(), -1

    concept_pos = content_start + start_in_content
    if concept_pos <= content_start:
        return embeds.clone(), concept_pos

    prefix_avg = embeds[0, content_start:concept_pos].mean(dim=0)
    out = embeds.clone()
    out[0, concept_pos] = prefix_avg
    return out, concept_pos


def replace_concept_span_with_prefix_avg(
    embeds: torch.Tensor,
    attn_mask: torch.Tensor,
    prompt_ids: torch.Tensor,
    concept_ids,
):
    # Replace all concept tokens with mean of all tokens before concept.
    valid_len = int(attn_mask[0].sum().item())
    content_start = 1
    content_end = max(content_start, valid_len - 1)  # exclude EOS
    if content_end <= content_start:
        return embeds.clone(), -1

    content_ids = prompt_ids[0, content_start:content_end].tolist()
    start_in_content = find_subsequence_start(content_ids, concept_ids)
    if start_in_content < 0:
        return embeds.clone(), -1

    concept_pos = content_start + start_in_content
    concept_end = min(content_end, concept_pos + len(concept_ids))
    if concept_pos <= content_start:
        return embeds.clone(), concept_pos

    prefix_avg = embeds[0, content_start:concept_pos].mean(dim=0)
    out = embeds.clone()
    out[0, concept_pos:concept_end] = prefix_avg
    return out, concept_pos


def replace_from_concept_to_eot_with_prefix_avg(
    embeds: torch.Tensor,
    attn_mask: torch.Tensor,
    prompt_ids: torch.Tensor,
    concept_ids,
):
    # Replace from concept start through EOS with mean of tokens before concept.
    valid_len = int(attn_mask[0].sum().item())
    content_start = 1
    content_end = max(content_start, valid_len - 1)  # exclude EOS
    if content_end <= content_start:
        return embeds.clone(), -1

    content_ids = prompt_ids[0, content_start:content_end].tolist()
    start_in_content = find_subsequence_start(content_ids, concept_ids)
    if start_in_content < 0:
        return embeds.clone(), -1

    concept_pos = content_start + start_in_content
    if concept_pos <= content_start:
        return embeds.clone(), concept_pos

    prefix_avg = embeds[0, content_start:concept_pos].mean(dim=0)
    out = embeds.clone()
    out[0, concept_pos:valid_len] = prefix_avg
    return out, concept_pos


def build_third_variant_embeds(
    mode: str,
    embeds: torch.Tensor,
    attn_mask: torch.Tensor,
    prompt_ids: torch.Tensor,
    concept_ids,
):
    if mode == "single":
        return replace_concept_token_with_prefix_avg(embeds, attn_mask, prompt_ids, concept_ids)
    if mode == "concept_span":
        return replace_concept_span_with_prefix_avg(embeds, attn_mask, prompt_ids, concept_ids)
    if mode == "to_eot":
        return replace_from_concept_to_eot_with_prefix_avg(embeds, attn_mask, prompt_ids, concept_ids)
    raise ValueError(f"Unknown replace_mode: {mode}")


def generate_one(pipe, prompt_embeds, negative_embeds, latents, steps, guidance):
    out = pipe(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_embeds,
        num_inference_steps=steps,
        guidance_scale=guidance,
        latents=latents,
        output_type="pil",
    )
    return out.images[0]


def generate_batch(pipe, prompt_embeds, negative_embeds, latents, steps, guidance):
    bsz = latents.shape[0]
    out = pipe(
        prompt_embeds=prompt_embeds.repeat(bsz, 1, 1),
        negative_prompt_embeds=negative_embeds.repeat(bsz, 1, 1),
        num_inference_steps=steps,
        guidance_scale=guidance,
        latents=latents,
        output_type="pil",
    )
    return out.images


def concat_1x3(imgs):
    w, h = imgs[0].size
    header_h = 36
    canvas = Image.new("RGB", (w * 3, h + header_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    titles = ["erase", "null", "avg-last"]

    for i, title in enumerate(titles):
        x0 = i * w
        x1 = (i + 1) * w
        draw.rectangle((x0, 0, x1, header_h), fill=(0, 0, 0))
        draw.text((x0 + 10, 10), title, fill=(255, 255, 255))

    for i, im in enumerate(imgs):
        canvas.paste(im, (i * w, header_h))
    return canvas


def sanitize_filename(s: str) -> str:
    s = re.sub(r"[^\w\s-]", "", s).strip().replace(" ", "_")
    return s[:120] if s else "empty"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_template", type=str, default=None, help="Optional single template override")
    parser.add_argument("--num_templates", type=int, default=10)
    parser.add_argument("--target", type=str, default="snoopy")
    parser.add_argument("--anchor", type=str, default="")
    parser.add_argument("--save_dir", type=str, default="result/ca")
    parser.add_argument("--num_per_template", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--model_id", type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--guidance", type=float, default=7.5)
    parser.add_argument(
        "--replace_mode",
        type=str,
        default="single",
        choices=["single", "concept_span", "to_eot"],
        help="single: replace first concept token; concept_span: replace all concept tokens; to_eot: replace concept->EOS",
    )
    args = parser.parse_args()
    args.save_dir = os.path.join(args.save_dir, args.replace_mode)

    if args.batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if args.num_per_template <= 0:
        raise ValueError("num_per_template must be > 0")

    os.makedirs(args.save_dir, exist_ok=True)

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(args.model_id, torch_dtype=dtype, safety_checker=None)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(args.device)

    templates = [args.prompt_template] if args.prompt_template else imagenet_templates[:args.num_templates]
    concept_ids = pipe.tokenizer(args.target, add_special_tokens=False)["input_ids"]
    if len(concept_ids) == 0:
        raise ValueError("target concept tokenization is empty")

    negative_embeds, _, _ = build_prompt_embeds(pipe, "")

    global_seed_offset = 0
    for tidx, tmpl in enumerate(templates):
        if "{}" not in tmpl:
            continue

        prompt_erase = tmpl.format(args.target)
        prompt_null = tmpl.format(args.anchor)

        erase_embeds, erase_mask, erase_ids = build_prompt_embeds(pipe, prompt_erase)
        null_embeds, _, _ = build_prompt_embeds(pipe, prompt_null)
        avg_concept_embeds, replaced_pos = build_third_variant_embeds(
            args.replace_mode,
            erase_embeds,
            erase_mask,
            erase_ids,
            concept_ids,
        )

        tmpl_tag = f"tmpl_{tidx:03d}_{sanitize_filename(tmpl)}"

        done = 0
        while done < args.num_per_template:
            cur_bsz = min(args.batch_size, args.num_per_template - done)
            seed_base = args.seed + global_seed_offset
            generator = torch.Generator(device=pipe.device).manual_seed(seed_base)
            latents = torch.randn(
                (cur_bsz, pipe.unet.in_channels, 64, 64),
                generator=generator,
                device=pipe.device,
                dtype=pipe.unet.dtype,
            )

            imgs_erase = generate_batch(pipe, erase_embeds, negative_embeds, latents.clone(), args.steps, args.guidance)
            imgs_null = generate_batch(pipe, null_embeds, negative_embeds, latents.clone(), args.steps, args.guidance)
            imgs_avg = generate_batch(pipe, avg_concept_embeds, negative_embeds, latents.clone(), args.steps, args.guidance)

            for bi in range(cur_bsz):
                idx = done + bi
                imgs_erase[bi].save(os.path.join(args.save_dir, f"{tmpl_tag}_{idx:04d}_erase.png"))
                imgs_null[bi].save(os.path.join(args.save_dir, f"{tmpl_tag}_{idx:04d}_null.png"))
                imgs_avg[bi].save(os.path.join(args.save_dir, f"{tmpl_tag}_{idx:04d}_avg_last.png"))
                merged = concat_1x3([imgs_erase[bi], imgs_null[bi], imgs_avg[bi]])
                merged.save(os.path.join(args.save_dir, f"{tmpl_tag}_{idx:04d}_concat.jpg"))

            done += cur_bsz
            global_seed_offset += cur_bsz

        print(f"[Template {tidx:03d}] saved {args.num_per_template} samples to: {args.save_dir}")
        print(f"erase prompt: {prompt_erase}")
        print(f"null  prompt: {prompt_null}")
        print(f"concept replaced token position: {replaced_pos}")
        print(f"replace mode: {args.replace_mode}")

    print(f"Done. All results saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
