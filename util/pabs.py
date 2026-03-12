# principal angle between subspaces

import os, re, pdb
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import math
import torch
import diffusers
from diffusers import StableDiffusionPipeline
from util.utils import get_token_id, get_textencoding, get_token

def compute_principal_angles(C_target, C_anchor):
    """
    计算两个特征矩阵子空间的主角度，并提取对齐后的主向量（共同特征）。
    
    参数:
        C_target: 目标概念特征矩阵, shape [n, 768]
        C_anchor: 锚点概念特征矩阵, shape [n, 768]
        
    返回:
        cos_theta: 主角度的余弦值 (奇异值), shape [n]
        angles_degree: 主角度的角度值 (度)
        P_target: 目标空间中的主向量 (重组后的特征), shape [n, 768]
        P_anchor: 锚点空间中的主向量 (重组后的特征), shape [n, 768]
    """
    # ---------------------------------------------------------
    # 第一步：转置矩阵，使列向量代表基向量
    # 在线性代数中，通常对矩阵的“列空间”求正交基
    # 形状变换：[n, 768] -> [768, n]
    # ---------------------------------------------------------
    A = C_target.T
    B = C_anchor.T
    
    # ---------------------------------------------------------
    # 第二步：QR 分解提取标准正交基
    # Qa 和 Qb 的形状为 [768, n]，它们的列向量相互正交且模长为 1
    # ---------------------------------------------------------
    Qa, Ra = torch.linalg.qr(A)
    Qb, Rb = torch.linalg.qr(B)
    
    # ---------------------------------------------------------
    # 第三步：计算两个正交基的内积矩阵 M
    # M 的形状为 [n, n]
    # ---------------------------------------------------------
    M = torch.matmul(Qa.T, Qb)
    
    # ---------------------------------------------------------
    # 第四步：对内积矩阵进行奇异值分解 (SVD)
    # U 形状 [n, n], S 形状 [n], Vh 形状 [n, n]
    # ---------------------------------------------------------
    U, S, Vh = torch.linalg.svd(M)
    V = Vh.T # 转置得到 V
    
    # ---------------------------------------------------------
    # 第五步：处理奇异值得到主角度
    # 奇异值 S 就是主角度的余弦值。
    # 注意：浮点误差可能导致 S 出现 1.000001，直接求 arccos 会报 NaN，必须 Clamp
    # ---------------------------------------------------------
    cos_theta = torch.clamp(S, min=-1.0, max=1.0)
    angles_rad = torch.acos(cos_theta)
    angles_degree = angles_rad * (180.0 / math.pi)
    
    # ---------------------------------------------------------
    # 第六步：计算子空间中对应的“主向量” (提取共同特征)
    # P_target 形状: [768, n] -> 转置回 [n, 768]
    # ---------------------------------------------------------
    P_target = torch.matmul(Qa, U).T 
    P_anchor = torch.matmul(Qb, V).T
    
    return cos_theta, angles_degree, P_target, P_anchor

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = pipeline = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to(device)

    target_concepts = ["Snoopy", "Micky", "SpongeBob"]
    anchor_concepts = [""] * 3
    t_emb = []
    a_emb = []
    
    for i in range(len(target_concepts)):
        target_inputs = get_token_id(target_concepts[i], pipeline.tokenizer, return_ids_only=False)
        target_embs = pipeline.text_encoder(target_inputs.input_ids.to(device)).last_hidden_state[0] # [77, 768]
        anchor_inputs = get_token_id(anchor_concepts[i], pipeline.tokenizer, return_ids_only=False)
        anchor_embs = pipeline.text_encoder(anchor_inputs.input_ids.to(device)).last_hidden_state[0] # [77, 768]
        t_last_token_index = target_inputs.attention_mask.sum() - 2 # -1 是因为索引从0开始，另一个-1是因为最后一个token是特殊的结束符
        a_last_token_index = anchor_inputs.attention_mask.sum() - 2

        t_emb.append(target_embs[t_last_token_index]) # shape: [768]
        a_emb.append(anchor_embs[a_last_token_index])
    
    t_emb = torch.stack(t_emb) # [3, 768]
    a_emb = torch.stack(a_emb) # [3, 768]
    print("Target embeddings shape:", t_emb.shape)
    print("Anchor embeddings shape:", a_emb.shape)

    cos_theta, angles_degree, P_target, P_anchor = compute_principal_angles(t_emb.view(-1, 768), a_emb.view(-1, 768))
    print("Cosine of principal angles:", cos_theta.detach().cpu().numpy())
    print("first 10 Cosine values: ", cos_theta.detach().cpu().numpy()[:10])
    print("Principal angles (degrees):", angles_degree)
    print("Aligned target features (P_target):", P_target)
    print("Aligned anchor features (P_anchor):", P_anchor)


