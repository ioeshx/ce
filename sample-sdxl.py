import warnings
warnings.filterwarnings("ignore")
import os, sys, pdb
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import re
import copy
import argparse
from PIL import Image
from tqdm import tqdm

import torch
from diffusers import AutoPipelineForText2Image, DiffusionPipeline, DPMSolverMultistepScheduler

from util.template import template_dict
from util.utils import process_img, seed_everything

## 使用template_dict中的prompt模板


def generate_images(pipe, prompts, steps, guidance, show_progress=True):
    out = pipe(
        prompt=prompts,
        num_inference_steps=steps,
        guidance_scale=guidance,
        output_type='pil',
    )
    return out.images


@torch.no_grad()
def main():

    parser = argparse.ArgumentParser()
    # Base Config
    parser.add_argument('--save_root', type=str, default='')
    parser.add_argument('--sd_ckpt', type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument('--seed', type=int, default=0)
    # Sampling Config
    parser.add_argument('--mode', type=str, default='original', help='original, edit')
    parser.add_argument('--guidance_scale', type=float, default=7.5)
    parser.add_argument('--total_timesteps', type=int, default=20, help='The total timesteps of the sampling process')
    parser.add_argument('--num_samples', type=int, default=10, help='The number of samples per prompt to generate' )
    parser.add_argument('--batch_size', type=int, default=10, help='The batch size of the sampling process')
    parser.add_argument('--prompts', type=str, default=None)
    # disable progress bar
    parser.add_argument('--disable_progress_bar', action='store_true', default=True, help='Disable tqdm and diffusers progress bars')
    # Erasing Config
    parser.add_argument('--erase_type', type=str, default='', help='instance, style, celebrity')
    parser.add_argument('--target_concept', type=str, default='')
    parser.add_argument('--contents', type=str, default='')
    parser.add_argument('--edit_ckpt', type=str, default=None)
    parser.add_argument("--disable_fixed_seed", action='store_true', default=False, help="Disable fixed seed for sampling.")
    args = parser.parse_args()
    assert args.num_samples >= args.batch_size and args.num_samples % args.batch_size == 0, "num_samples should be a multiple of batch_size."
    print("[Arguments]")
    for key, value in vars(args).items():
        print(f"{key}={value}")


    bs = args.batch_size
    mode_list = args.mode.replace(' ', '').split(',')
    only_original = mode_list == ['original']
    only_edit = mode_list == ['edit']

    # region [If certain concept is already sampled, then skip it.]
    concept_list, concept_list_tmp = [], [item.strip() for item in args.contents.split(',')]
    if 'edit' in mode_list:
        for concept in concept_list_tmp:
            check_path = os.path.join(args.save_root, args.target_concept.replace(', ', '_'), concept, 'edit')
            os.makedirs(check_path, exist_ok=True)
            if len(os.listdir(check_path)) != len(template_dict[args.erase_type]) * args.num_samples:
                concept_list.append(concept)
    else:
        concept_list = concept_list_tmp
    if len(concept_list) == 0: sys.exit()
    # endregion

    # region [Prepare Models]
    try:
        pipe = AutoPipelineForText2Image.from_pretrained(args.sd_ckpt, torch_dtype=torch.float16).to('cuda')
    except Exception:
        pipe = DiffusionPipeline.from_pretrained(args.sd_ckpt, torch_dtype=torch.float16).to('cuda')
    if args.disable_progress_bar:
        pipe.set_progress_bar_config(disable=True)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    if only_edit:
        edit_path = args.edit_ckpt or os.path.join("logs/checkpoints", sorted(os.listdir("logs/checkpoints"))[-1])
        pipe.unet.load_state_dict(torch.load(edit_path, map_location='cpu'), strict=False)
        pipe_edit = None
    elif 'edit' in mode_list:
        pipe_edit = copy.deepcopy(pipe)
        edit_path = args.edit_ckpt or os.path.join("logs/checkpoints", sorted(os.listdir("logs/checkpoints"))[-1])
        pipe_edit.unet.load_state_dict(torch.load(edit_path, map_location='cpu'), strict=False)
    else:
        pipe_edit = None
    # endregion

    # Sampling process
    if not args.disable_fixed_seed:
        seed_everything(args.seed, True)
    
    if args.prompts is None:
        prompt_list = [[x.format(concept) for x in template_dict[args.erase_type]] for concept in concept_list]
    else:
        prompt_list = [[x.format(concept) for x in args.prompts.split(';')] for concept in concept_list]
    for i in range(int(args.num_samples // bs)):
        for concept, prompts in zip(concept_list, prompt_list):
            for count, prompt in enumerate(prompts):

                save_images = {}
                batch_prompts = [prompt] * bs

                if 'original' in mode_list:
                    save_images['original'] = generate_images(
                        pipe=pipe,
                        prompts=batch_prompts,
                        steps=args.total_timesteps,
                        guidance=args.guidance_scale,
                        show_progress=not args.disable_progress_bar,
                    )
                if 'edit' in mode_list:
                    edit_pipe = pipe if only_edit else pipe_edit
                    save_images['edit'] = generate_images(
                        pipe=edit_pipe,
                        prompts=batch_prompts,
                        steps=args.total_timesteps,
                        guidance=args.guidance_scale,
                        show_progress=not args.disable_progress_bar,
                    )
                                        
                save_path = os.path.join(args.save_root, args.target_concept.replace(', ', '_'), concept)
                for mode in mode_list: os.makedirs(os.path.join(save_path, mode), exist_ok=True)
                if len(mode_list) > 1: os.makedirs(os.path.join(save_path, 'combine'), exist_ok=True)

                decoded_imgs = save_images

                # Save images
                def combine_images_horizontally(Images):
                    widths, heights = zip(*(img.size for img in Images))
                    new_img = Image.new('RGB', (sum(widths), max(heights)))
                    for i, img in enumerate(Images): new_img.paste(img, (sum(widths[:i]), 0))
                    return new_img
                for idx in range(len(decoded_imgs[mode_list[0]])):
                    save_filename = re.sub(r'[^\w\s]', '', prompt).replace(', ', '_') + f"_{int(idx + bs * i)}.png"
                    images_to_combine = []
                    for mode in mode_list: 
                        decoded_imgs[mode][idx].save(os.path.join(save_path, mode, save_filename))
                        images_to_combine.append(decoded_imgs[mode][idx])
                    if len(mode_list) > 1:
                        img_combined = combine_images_horizontally(images_to_combine)
                        img_combined.save(os.path.join(save_path, 'combine', save_filename.replace('.png', '.jpg')))


if __name__ == '__main__':
    main()