import os, re, pdb
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import time
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from kmeans_pytorch import kmeans
from diffusers import StableDiffusionPipeline
import random

from util.utils import str2bool


def seed_everything(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_token_id(prompt, tokenizer=None, return_ids_only=True):
    token_ids = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    return token_ids.input_ids if return_ids_only else token_ids

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

    if baseline in ['SPEED']:
        null_inputs = get_token_id('', pipeline.tokenizer, return_ids_only=False)
        null_hidden = pipeline.text_encoder(null_inputs.input_ids.to(device)).last_hidden_state[0] # [0] index -> [77, 768]
        cluster_ids, cluster_centers = kmeans(X=null_hidden[1:], num_clusters=3, distance='euclidean', device='cuda')
        K2 = torch.cat([null_hidden[[0], :], cluster_centers.to(device)], dim=0).T # shape: [768, 4]
        I2 = torch.eye(len(K2.T), device=device) # shape: [4, 4]
    else:
        raise ValueError("Invalid baseline")

    # region [Target and Anchor]
    # nudity 使用所有token，其他概念使用last subject token
    # 擦除多个概念时，取它们的平均作为最终的擦除目标；如果是单个概念，则直接使用该概念的文本嵌入作为擦除目标
    if args.enable_target_proj2_anchor:
        print("Enable Common = target projection to anchor.")
    if args.pabs:
        print("Enable Principal Angle between Subspaces (PABS) analysis.")
        from util.pabs import compute_principal_angles
        t_token = get_token_id(target_concepts, pipeline.tokenizer, False) # [len(target_concepts), 77]
        a_token = get_token_id(anchor_concepts, pipeline.tokenizer, False) # [len(anchor_concepts), 77]
        t_embs = pipeline.text_encoder(t_token.input_ids.to(device)).last_hidden_state[0] # [len(target_concepts), 77, 768]
        a_embs = pipeline.text_encoder(a_token.input_ids.to(device)).last_hidden_state[0] # [len(anchor_concepts), 77, 768]
        t_last_embs = t_embs[t_token.attention_mask[0].sum().item()-2, :].unsqueeze(1) # shape: [len(target_concepts), 1, 768]
        a_last_embs = a_embs[a_token.attention_mask[0].sum().item()-2, :].unsqueeze(1) # shape: [len(anchor_concepts), 1, 768]
        cos_theta, angles_degree, P_target, P_anchor = compute_principal_angles(t_last_embs, a_last_embs)
        anchor_final_embes = P_anchor[0].unsqueeze(0) # shape: [1, 768]
        print("diff between original anchor and PABS anchor: ", torch.norm(a_last_embs.squeeze(1) - anchor_final_embes) / torch.norm(a_last_embs.squeeze(1)))
        print("shape of anchor_final_embes: ", anchor_final_embes.shape)
        print("Cosine of principal angles:", cos_theta.detach().cpu().numpy())


    sum_anchor_target, sum_target_target = [], []
    all_target_embs_list = []
    for i in range(0, len(target_concepts)):
        target_inputs = get_token_id(target_concepts[i], pipeline.tokenizer, return_ids_only=False)
        target_embs = pipeline.text_encoder(target_inputs.input_ids.to(device)).last_hidden_state[0] # [77, 768]
        anchor_inputs = get_token_id(anchor_concepts[i], pipeline.tokenizer, return_ids_only=False)
        anchor_embs = pipeline.text_encoder(anchor_inputs.input_ids.to(device)).last_hidden_state[0] # [77, 768]
        if target_concepts == ['nudity']:
            target_embs = target_embs[1:, :]  # all tokens
            anchor_embs = anchor_embs[1:, :]  # all tokens
        if args.max_valid_tokens:
            print("Enable all token editing")
            t_last_index = target_inputs.attention_mask[0].sum().item() - 2
            a_last_index = anchor_inputs.attention_mask[0].sum().item() - 2
            max_valid_idx = max(t_last_index, a_last_index)  

            target_embs = target_embs[1:max_valid_idx+1, :]  # all tokens
            anchor_embs = anchor_embs[1:max_valid_idx+1, :]  # all tokens
        else:
            target_embs = target_embs[[(target_inputs.attention_mask[0].sum().item() - 2)], :]  # last subject token [1,768]
            anchor_embs = anchor_embs[[(anchor_inputs.attention_mask[0].sum().item() - 2)], :]  # last subject token
        # $$P = C_{anchor}(C_{anchor}^\top C_{anchor})^{-1}C_{anchor}^\top$$
        
        # region [project]
        tar_origin = target_embs.clone()
        anch_origin = anchor_embs.clone()

        project2a_coeff = target_embs @ anchor_embs.T @ (anchor_embs @ anchor_embs.T).inverse()  # shape: [1, 77] or [1, 1]
        project2t_coeff = anchor_embs @ target_embs.T @ (target_embs @ target_embs.T).inverse()  # shape: [1, 77] or [1, 1]
        if(args.max_valid_tokens):
            project2a_coeff = torch.diag(project2a_coeff).unsqueeze(1)  # shape: [num_valid_tokens]
            project2t_coeff = torch.diag(project2t_coeff).unsqueeze(1)  # shape: [num_valid_tokens]
            print("Project2a coeff shape: ", project2a_coeff.shape)
            print("Project2t coeff shape: ", project2t_coeff.shape)

        anchor_proj = anchor_embs * project2a_coeff
        tar_proj = target_embs * project2t_coeff
        if args.t2a:
            print("Enable t2a+anchor")
            anchor_embs = anchor_embs + anchor_proj
        elif args.a2t:
            print("Enable a2t+anchor")
            anchor_embs = anchor_embs + tar_proj
        elif args.t2a_only:
            print("Enable t2a only")
            anchor_embs = anchor_proj
        elif args.a2t_only:
            print("Enable a2t only")
            anchor_embs = tar_proj

        if args.robust_PCA:
            print("Enable Robust PCA")
            from util.rpca import robust_pca_target_anchor
            rpca_result = robust_pca_target_anchor(target_embs, anchor_embs)
            # target_embs = rpca_result['L_target'].to(device)
            anchor_embs = rpca_result['L_anchor'].to(device)

            print("After RPCA")
            print("Rel abs: target={}, anchor={}".format(torch.norm(tar_origin - target_embs) / torch.norm(tar_origin), 
                                                                torch.norm(anch_origin - anchor_embs) / torch.norm(anch_origin)))
            print("cosine sim: target={}, anchor={}".format(torch.cosine_similarity(tar_origin, target_embs), torch.cosine_similarity(anch_origin, anchor_embs)))
        if args.pabs:
            anchor_embs = anchor_final_embes
          
        all_target_embs_list.append(target_embs)
        sum_target_target.append(target_embs.T @ target_embs)
        sum_anchor_target.append(anchor_embs.T @ target_embs)
    ## 这里用了平均
    sum_target_target, sum_anchor_target = torch.stack(sum_target_target).mean(0), torch.stack(sum_anchor_target).mean(0)
    # shape of both: [768, 768]
    # endregion


    # region [Retain]
    # retain为空，则使用一个全零输入的文本嵌入作为retain；否则使用retain文本中最后一个subject token的文本嵌入作为retain
    last_ret_embs = []
    retain_texts = [text for text in retain_texts if not any(re.search(r'\b' + re.escape(concept.lower()) + r'\b', text.lower()) for concept in target_concepts)]
    assert len(retain_texts) + len(target_concepts) == len(set(retain_texts + target_concepts))
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

    # Hard-Boundary Retain Augmentation
    if args.hard_boundary_aug:
        C0_sq = last_ret_embs.squeeze(1) # [total_retain_num, 768]
        
        if args.boundary_per_concept:
            augmented_samples = []
            for t_embs in all_target_embs_list:
                C1_local = t_embs.mean(dim=0, keepdim=True) # [1, 768]
                
                C0_norm = torch.nn.functional.normalize(C0_sq, p=2, dim=1)
                C1_norm = torch.nn.functional.normalize(C1_local, p=2, dim=1)
                sim = torch.matmul(C0_norm, C1_norm.T).squeeze(1) 
                
                topk = min(args.boundary_topk, C0_sq.size(0))
                if topk > 0:
                    hard_indices = torch.topk(sim, topk).indices
                    C0_hard = C0_sq[hard_indices]
                    diff = C0_hard - C1_local
                    diff_norm = torch.norm(diff, p=2, dim=1, keepdim=True) + 1e-8
                    C0_enhanced = C0_hard + args.boundary_gamma * (diff / diff_norm)
                    augmented_samples.append(C0_enhanced)
            
            if augmented_samples:
                all_augmented = torch.cat(augmented_samples, dim=0)
                last_ret_embs = torch.cat([last_ret_embs, all_augmented.unsqueeze(1)], dim=0)
                print(f"[INFO] Applied Hard-Boundary Augmentation: Added {all_augmented.size(0)} augmented samples toward boundary (Per Concept). Gamma: {args.boundary_gamma}")
        else:
            C1_global = torch.cat(all_target_embs_list, dim=0).mean(dim=0, keepdim=True) # [1, 768]
            
            # Calculate cosine similarity
            C0_norm = torch.nn.functional.normalize(C0_sq, p=2, dim=1)
            C1_norm = torch.nn.functional.normalize(C1_global, p=2, dim=1)
            sim = torch.matmul(C0_norm, C1_norm.T).squeeze(1) # [total_retain_num]
            
            # Find Top-K nearest
            topk = min(args.boundary_topk, C0_sq.size(0))
            if topk > 0:
                hard_indices = torch.topk(sim, topk).indices
                
                C0_hard = C0_sq[hard_indices] # [topk, 768]
                
                # Enhance: C0_enhanced = C0 + gamma * (C0 - C1) / ||C0 - C1||
                diff = C0_hard - C1_global
                diff_norm = torch.norm(diff, p=2, dim=1, keepdim=True) + 1e-8
                C0_enhanced = C0_hard + args.boundary_gamma * (diff / diff_norm)
                
                last_ret_embs = torch.cat([last_ret_embs, C0_enhanced.unsqueeze(1)], dim=0)
                print(f"[INFO] Applied Hard-Boundary Augmentation: Added {topk} augmented samples toward boundary. Gamma: {args.boundary_gamma}")

    # Semantic Manifold Interpolation
    if args.manifold_interp and args.interp_samples > 0:
        current_num = last_ret_embs.size(0)
        if current_num > 1:
            idx1 = torch.randint(0, current_num, (args.interp_samples,))
            idx2 = torch.randint(0, current_num, (args.interp_samples,))
            
            theta = torch.rand((args.interp_samples, 1, 1), device=device)
            
            C0_i = last_ret_embs[idx1]
            C0_j = last_ret_embs[idx2]
            
            C0_interp = theta * C0_i + (1 - theta) * C0_j
            last_ret_embs = torch.cat([last_ret_embs, C0_interp], dim=0)
            print(f"[INFO] Applied Semantic Manifold Interpolation: Added {args.interp_samples} interpolated retain samples.")

    # 在后面也做了平均，所以retain的处理方式和target/anchor是一致的，都是取文本嵌入的平均作为最终的retain矩阵；如果retain文本为空，则使用全零输入的文本嵌入（同样取平均）作为retain矩阵
    # endregion
    
    ret_ret_embs = last_ret_embs.squeeze(1).T @ last_ret_embs.squeeze(1)

    if args.aug_num > 0 and not args.disable_filter:
        print("Enable IPF and DPA")
    else:
        print("No IPF or DPA")
    
    for (layer_name, layer_weight) in tqdm(edit_dict.items(), desc="Model Editing"):

        erase_weight = layer_weight @ (sum_anchor_target - sum_target_target) @ (I + sum_target_target).inverse()        

        (U0, S0, V0) = torch.svd(layer_weight)
        P0_min = V0[:, -1:] @ V0[:, -1:].T
        # INFLUENCE-BASED PRIOR FILTERING
        # if args.aug_num > 0 and not args.disable_filter:
        #     # prior shift filtering
        #     weight_norm_init = torch.matmul(last_ret_embs.squeeze(1), erase_weight.T).norm(dim=1)
        #     layer_ret_embs = last_ret_embs[weight_norm_init > weight_norm_init.mean()]
        # else:
        #      layer_ret_embs = last_ret_embs
        layer_ret_embs = last_ret_embs
        
        # retain矩阵也是取平均的方式进行编辑
        sum_ret_ret, valid_num = [], 0
        for j in range(0, len(layer_ret_embs), chunk_size):
            chunk_ret_embs = layer_ret_embs[j:j + chunk_size]
            # if args.aug_num > 0:
            #     chunk_ret_embs = torch.cat(
            #         [chunk_ret_embs, generate_perturbed_embs(chunk_ret_embs, P0_min, erase_weight, num_per_sample=args.aug_num)], dim=0
            #     )
            valid_num += chunk_ret_embs.shape[0]
            sum_ret_ret.append((chunk_ret_embs.transpose(1, 2) @ chunk_ret_embs).sum(0))
        sum_ret_ret = torch.stack(sum_ret_ret, dim=0).sum(0) / valid_num


        U, S, V = torch.svd(sum_ret_ret)
        P = U[:, S < args.threshold] @ U[:, S < args.threshold].T
        
        # $$\Delta P = W(C_*C_1^\top - C_1C_1^\top) P (C_1C_1^\top P + I)^{-1}$$
        delta_weight = layer_weight @ (sum_anchor_target - sum_target_target) @ P @ (sum_target_target @ P + I).inverse()
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
    parser.add_argument('--retain_scale', type=float, default=1.0) # not used
    parser.add_argument('--lamb', type=float, default=0.0)  # not used
    parser.add_argument('--disable_filter', action='store_true', default=False)
    # null project2retain + my ideas
    parser.add_argument('--enable_target_proj2_anchor', action='store_true', default=False)
    # robust PCA
    parser.add_argument('--robust_PCA', action="store_true", default=False)
    parser.add_argument('--rpca_lam', type=float)  # only used when robust_PCA is True
    # principal angle between subspaces
    parser.add_argument('--pabs', action='store_true', default=False)
    # boundary enhancement
    parser.add_argument('--hard_boundary_aug', action='store_true', default=False)
    parser.add_argument('--boundary_topk', type=int, default=10)
    parser.add_argument('--boundary_gamma', type=float, default=0.1)
    parser.add_argument('--boundary_per_concept', action='store_true', default=False)
    # semantic manifold interpolation
    parser.add_argument('--manifold_interp', action='store_true', default=False)
    parser.add_argument('--interp_samples', type=int, default=100)
    # t2a/a2t
    mutually_excl_group = parser.add_mutually_exclusive_group(required=False)
    mutually_excl_group.add_argument('--t2a', action='store_true', default=False)
    mutually_excl_group.add_argument('--a2t', action='store_true', default=False)
    mutually_excl_group.add_argument('--t2a_only', action='store_true', default=False)
    mutually_excl_group.add_argument('--a2t_only', action='store_true', default=False)

    parser.add_argument('--all_token', action='store_true', default=False)
    parser.add_argument('--max_valid_tokens', action='store_true', default=False, help="")

    args = parser.parse_args()
    print("[Arguments]")
    for key, value in vars(args).items():
        print(f"{key}={value}")
    

    device = torch.device("cuda")
    seed_everything(args.seed)

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
        assert len(target_concepts) == len(anchor_concepts)
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
    

