import torch


def _to_tensor(x, dtype=torch.float32, device=None) -> torch.Tensor:
    """Convert numpy/torch input to torch.Tensor on the requested device."""
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(x, dtype=dtype, device=device)


def _soft_threshold(x: torch.Tensor, tau: float) -> torch.Tensor:
    """Element-wise soft-thresholding operator."""
    return torch.sign(x) * torch.clamp(torch.abs(x) - tau, min=0.0)


def robust_pca_target_anchor(
    target_matrix,
    anchor_matrix,
    lam: float = None,
    mu: float = None,
    max_iter: int = 1000,
    tol: float = 1e-7,
    device: str = None,
    dtype: torch.dtype = torch.float32,
):
    """
    Solve Robust PCA (Principal Component Pursuit) on M=[target; anchor].

    Objective:
        min ||L||_* + lam * ||S||_1,  s.t.  M = L + S

    Args:
        target_matrix: shape (m_t, n), target feature matrix.
        anchor_matrix: shape (m_a, n), anchor feature matrix.
        lam: sparsity weight lambda. If None, uses 1/sqrt(max(m, n)).
        mu: ADMM penalty parameter. If None, uses a common heuristic.
        max_iter: maximum number of iterations.
        tol: stopping tolerance on relative Frobenius residual.
        device: torch device string, e.g. "cuda" or "cpu". If None, auto-select.
        dtype: tensor dtype for computation.

    Returns:
        A dict with:
        - "M": concatenated matrix [target; anchor] (row-wise)
        - "L": low-rank component
        - "S": sparse component
        - "L_target", "L_anchor": split low-rank components
        - "S_target", "S_anchor": split sparse components
        - "n_iter": number of iterations used
        - "converged": whether convergence criterion is met
        - "residual": final relative residual ||M-L-S||_F / ||M||_F
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    target = _to_tensor(target_matrix, dtype=dtype, device=device)
    anchor = _to_tensor(anchor_matrix, dtype=dtype, device=device)

    if target.ndim != 2 or anchor.ndim != 2:
        raise ValueError("Both target_matrix and anchor_matrix must be 2D arrays.")
    if target.shape[1] != anchor.shape[1]:
        raise ValueError(
            "target_matrix and anchor_matrix must have the same column count for row-wise concatenation."
        )

    m_t, n = target.shape
    _, _ = anchor.shape
    M = torch.cat([target, anchor], dim=0)
    m, n = M.shape

    if lam is None:
        lam = 1.0 / (max(m, n) ** 0.5)
    if lam <= 0:
        raise ValueError("lam must be positive.")

    fro_norm_m = torch.linalg.norm(M, ord="fro")
    if fro_norm_m.item() == 0:
        L = torch.zeros_like(M)
        S = torch.zeros_like(M)
        return {
            "M": M,
            "L": L,
            "S": S,
            "L_target": L[:m_t, :],
            "L_anchor": L[m_t:, :],
            "S_target": S[:m_t, :],
            "S_anchor": S[m_t:, :],
            "n_iter": 0,
            "converged": True,
            "residual": 0.0,
        }

    # A common heuristic for PCP/ADMM initialization.
    if mu is None:
        mu = (m * n) / (4.0 * torch.sum(torch.abs(M)).item() + 1e-12)
    if mu <= 0:
        raise ValueError("mu must be positive.")

    L = torch.zeros_like(M)
    S = torch.zeros_like(M)
    Y = torch.zeros_like(M)

    converged = False
    residual = float("inf")

    for k in range(1, max_iter + 1):
        # L-update via singular value thresholding.
        U, sigma, Vh = torch.linalg.svd(M - S + (1.0 / mu) * Y, full_matrices=False)
        sigma_thresh = torch.clamp(sigma - 1.0 / mu, min=0.0)
        rank = int((sigma_thresh > 0).sum().item())
        if rank > 0:
            L = (U[:, :rank] * sigma_thresh[:rank]) @ Vh[:rank, :]
        else:
            L = torch.zeros_like(M)

        # S-update via soft-thresholding.
        S = _soft_threshold(M - L + (1.0 / mu) * Y, lam / mu)

        # Dual update.
        R = M - L - S
        Y = Y + mu * R

        residual = (torch.linalg.norm(R, ord="fro") / fro_norm_m).item()
        if residual < tol:
            converged = True
            break

    return {
        "M": M,
        "L": L,
        "S": S,
        "L_target": L[:m_t, :],
        "L_anchor": L[m_t:, :],
        "S_target": S[:m_t, :],
        "S_anchor": S[m_t:, :],
        "n_iter": k,
        "converged": converged,
        "residual": residual,
    }
