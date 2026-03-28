import os, sys, pdb
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import re
import copy
import argparse
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from diffusers import AutoPipelineForText2Image, DiffusionPipeline, DPMSolverMultistepScheduler

from util.template import template_dict
from util.utils import process_img

# 使用自定义的csv文件生成

def generate_images(pipe, prompts, steps, guidance):
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
    parser.add_argument('--coco_max_num', type=int, default=1000, help='The max number of coco samples to generate')
    # Erasing Config
    parser.add_argument('--erase_type', type=str, default='', help='instance, style, celebrity')
    parser.add_argument('--target_concept', type=str, default='')
    parser.add_argument('--contents', type=str, default='')
    parser.add_argument('--edit_ckpt', type=str, required=False)
    args = parser.parse_args()

    bs = args.batch_size
    mode_list = args.mode.replace(' ', '').split(',')

    # region [Prepare Models]
    try:
        pipe = AutoPipelineForText2Image.from_pretrained(args.sd_ckpt, torch_dtype=torch.float16).to('cuda')
    except Exception:
        pipe = DiffusionPipeline.from_pretrained(args.sd_ckpt, torch_dtype=torch.float16).to('cuda')
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    if 'edit' in mode_list:
        pipe_edit = copy.deepcopy(pipe)
        pipe_edit.unet.load_state_dict(torch.load(args.edit_ckpt, map_location='cpu'), strict=False)
    # endregion

    # Sampling process
    if args.contents == 'nudity':
        dataset = AdaDataset(data_path='data/i2p_benchmark.csv')
    elif args.contents == 'coco':
        dataset = AdaDataset(data_path ='data/mscoco.csv', seed=args.seed, guidance_scale=args.guidance_scale, max_num=args.coco_max_num)
    elif 'erase' in args.contents or 'retain' in args.contents:
        dataset = AdaDataset(data_path =f'data/{args.erase_type}.csv', guidance_scale=args.guidance_scale)
    dataloader = DataLoader(dataset, batch_size=bs, drop_last=False)

    for content in args.contents.split(', '):
        for count, data in enumerate(dataloader):
            if content in ['erase', 'retain'] and content != data['type'][0]:
                continue

            save_images = {}

            curr_bs = len(data['prompt'])
            batch_prompts = list(data['prompt'])

            if 'original' in mode_list:
                save_images['original'] = generate_images(
                    pipe=pipe,
                    prompts=batch_prompts,
                    steps=args.total_timesteps,
                    guidance=args.guidance_scale,
                )
            if 'edit' in mode_list:
                save_images['edit'] = generate_images(
                    pipe=pipe_edit,
                    prompts=batch_prompts,
                    steps=args.total_timesteps,
                    guidance=args.guidance_scale,
                )
            
            save_path = os.path.join(args.save_root, args.target_concept.replace(', ', '_'), content)
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
                if content == 'coco':
                    save_filename = f'COCO_val2014_{int(data["idx"][idx]):012}.png'
                elif content == 'nudity':
                    save_filename = f'{data["idx"][idx]}_' + re.sub(r'[^\w\s]', '', data["prompt"][idx]).replace(' ', '_')[:100]+ f"_{int(idx + bs * idx)}.png"
                elif content in ['erase', 'retain']:
                    save_filename = re.sub(r'[^\w\s]', '', data["prompt"][idx]).replace(' ', '_') + f'_{data["idx"][idx]}.png'
                images_to_combine = []
                for mode in mode_list: 
                    decoded_imgs[mode][idx].save(os.path.join(save_path, mode, save_filename))
                    images_to_combine.append(decoded_imgs[mode][idx])
                if len(mode_list) > 1:
                    img_combined = combine_images_horizontally(images_to_combine)
                    img_combined.save(os.path.join(save_path, 'combine', save_filename.replace('.png', '.jpg')))


class AdaDataset(Dataset):
    def __init__(self, data_path, seed=None, guidance_scale=None, max_num=100000):
        self.data_path = data_path
        self.data = pd.read_csv(data_path)
        if 'coco' in data_path:
            self.prompt_list = list(self.data['text'])[:max_num]
            self.idx = list(self.data['image_id'])[:max_num]
            self.seed = [seed] * len(self.prompt_list)
            self.guidance_scale = [guidance_scale] * len(self.prompt_list)
        elif 'i2p' in data_path:
            # self.idx = list(range(1000, 1500))[:max_num]
            self.idx = list(range(4703))[:max_num]
            self.data = self.data.iloc[self.idx]
            self.prompt_list = list(self.data['prompt'])
            self.seed = list(self.data['sd_seed'])
            self.guidance_scale = list(self.data['sd_guidance_scale'])
        elif '_celebrity' in data_path:
            self.prompt_list = list(self.data['text'])[:max_num]
            self.idx = list(self.data['id'])[:max_num]
            self.seed = list(self.data['seed'])
            self.guidance_scale = [guidance_scale] * len(self.prompt_list)
            self.type = list(self.data['type'])[:max_num]

    def __getitem__(self, idx):
        item = {
            'prompt': self.prompt_list[idx],
            'idx': self.idx[idx],
            'seed': self.seed[idx],
            'guidance': self.guidance_scale[idx],
        }
        if '_celebrity' in self.data_path:
            item['type'] = self.type[idx]
        return item
    
    def __len__(self):
        return len(self.prompt_list)


if __name__ == '__main__':
    main()