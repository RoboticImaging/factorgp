"""
Acquisition functions and heuristics for active sampling with an FGP.

All geometry-specific quantities (edges, grid_shape, h, noise_var) are passed
explicitly so these functions are reusable across experiments.
"""

import numpy as np
import torch

from fgp.interpolate import fd_interpolate_cuts
from fgp.graph_learning import build_Lambda


def node_graph_uncertainty(pi_std, edges, n_nodes):
    """Mean pi_std of edges incident to each inducing node."""
    unc   = np.zeros(n_nodes)
    count = np.zeros(n_nodes)
    for e_idx, (i, j) in enumerate(edges):
        unc[i]   += pi_std[e_idx]
        unc[j]   += pi_std[e_idx]
        count[i] += 1
        count[j] += 1
    return unc / np.maximum(count, 1)


def heuristic_scores(pts, fgp_opt, pi_map, pi_std, edges, grid_shape, h,
                     alpha=1, beta=1, norm_mask=None):
    """
    Hand-crafted heuristic:
        acq = alpha * normalize(gmrf_var) + beta * normalize(W_full @ node_unc)

    alpha: weight on field uncertainty term (set to 0 for graph-only exploration)
    beta:  weight on graph uncertainty term (set to 0 for field-only exploration)
    norm_mask: boolean array (len(pts),) — if given, normalization max is taken
               over masked points only (use to exclude wall pixels from scaling).

    Returns (acq, gmrf_norm, graph_norm) each shape (Q,).
    """
    n_nodes   = grid_shape[0] * grid_shape[1]
    act_e     = [edges[i] for i in range(len(pi_map)) if pi_map[i] > 0.5]
    node_unc  = node_graph_uncertainty(pi_std, edges, n_nodes)
    W_cut     = fd_interpolate_cuts(pts, list(grid_shape), h, edges, act_e)
    _, cov    = fgp_opt.query(pts, W_cut)
    gmrf_var  = np.diag(cov)
    W_full    = fd_interpolate_cuts(pts, list(grid_shape), h, edges, edges)
    graph_unc = np.asarray(W_full @ node_unc).ravel()
    ref = norm_mask if norm_mask is not None else np.ones(len(pts), dtype=bool)
    gmrf_norm  = gmrf_var  / (gmrf_var[ref].max()  + 1e-12)
    graph_norm = graph_unc / (graph_unc[ref].max() + 1e-12)
    return alpha * gmrf_norm + beta * graph_norm, gmrf_norm, graph_norm


def compute_particle_stats(pts, final_pi, final_kappa, cart, zs,
                           edges, grid_shape, h, noise_var, threshold=0.5):
    """Per-particle predictive mean/var at pts, using each particle's OWN cut-aware
    connectivity in BOTH the field posterior Λ(Q) and the readout W(Q).

    Returns mu (Q,P), var (Q,P) [field var, no noise], cross_ssq (Q,P) [Σ_n Cov(f_n,y_q)²].
    """
    device  = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype   = torch.float64
    n_nodes = grid_shape[0] * grid_shape[1]
    P       = len(final_pi)
    z_t     = torch.tensor(zs.ravel(), dtype=dtype, device=device)
    pi_t, kappa_t = (torch.tensor(a, dtype=dtype, device=device)
                     for a in (final_pi, final_kappa))

    Lj, eta, Wpts = [], [], []
    for k in range(P):
        A    = [edges[i] for i in range(len(edges)) if final_pi[k, i] > threshold]
        W_tr = torch.tensor(
            fd_interpolate_cuts(cart, list(grid_shape), h, edges, A).toarray(),
            dtype=dtype, device=device)
        W_pt = torch.tensor(
            fd_interpolate_cuts(pts, list(grid_shape), h, edges, A).toarray(),
            dtype=dtype, device=device)
        Lj.append(build_Lambda(edges, pi_t[k], kappa_t[k], n_nodes, h)
                  + W_tr.T @ W_tr / noise_var)
        eta.append(W_tr.T @ z_t / noise_var)
        Wpts.append(W_pt)

    Lj = torch.stack(Lj)
    m  = torch.linalg.solve(Lj, torch.stack(eta).unsqueeze(-1)).squeeze(-1)

    Q   = len(pts)
    mu  = torch.empty(Q, P, dtype=dtype, device=device)
    var = torch.empty(Q, P, dtype=dtype, device=device)
    css = torch.empty(Q, P, dtype=dtype, device=device)
    for k in range(P):
        Wk        = Wpts[k]
        Vk        = torch.linalg.solve(Lj[k], Wk.T)
        mu[:, k]  = Wk @ m[k]
        var[:, k] = (Wk * Vk.T).sum(dim=1)
        css[:, k] = (Vk ** 2).sum(dim=0)
    return mu.cpu().numpy(), var.cpu().numpy(), css.cpu().numpy()


def heuristic_demap_scores(pts, E_var, pi_std, edges, grid_shape, h, norm_mask=None):
    """
    De-MAPified heuristic: Term 1 uses the pre-computed ensemble-averaged field
    variance E[ν] (shape (Q,)) instead of the MAP particle's gmrf_var.
    Term 2 (graph uncertainty projection) is identical to heuristic_scores.

    E_var: (Q,) array — var_batch.mean(axis=1), precomputed by caller.
    Returns (acq, field_norm, graph_norm) each shape (Q,).
    """
    n_nodes  = grid_shape[0] * grid_shape[1]
    node_unc  = node_graph_uncertainty(pi_std, edges, n_nodes)
    W_full    = fd_interpolate_cuts(pts, list(grid_shape), h, edges, edges)
    graph_unc = np.asarray(W_full @ node_unc).ravel()
    ref = norm_mask if norm_mask is not None else np.ones(len(pts), dtype=bool)
    field_norm = E_var  / (E_var[ref].max()  + 1e-12)
    graph_norm = graph_unc / (graph_unc[ref].max() + 1e-12)
    return field_norm + graph_norm, field_norm, graph_norm


def acquisition_scores(name, mu_batch, var_batch, noise_var, cross_ssq=None):
    """MI-based acquisition scores from per-particle field statistics.

    var_batch = FIELD variance (no noise); noise_var is added here. 2πe dropped (cancels).
        F      = ½log σ²                         H(y|f_U,Q)  floor
        E_Hp   = mean_s ½log(ν_s+σ²)             E_Q[H(y|Q)]
        H_marg = ½log(E_Q[ν]+Var_Q[μ]+σ²)       H(y|f_obs)  pooled mixture

        field_mi  = I(y;f_U|Q) = E_Hp  - F
        graph_mi  = I(y;Q)     = H_marg - E_Hp
        joint_mi  = J          = H_marg - F      (= field_mi + graph_mi)
        field_eig = E_Q[whole-field variance reduction]   (RMSE-aligned)
    """
    E_var, Var_mu = var_batch.mean(1), mu_batch.var(1)
    F      = 0.5 * np.log(noise_var)
    E_Hp   = 0.5 * np.log(var_batch + noise_var).mean(1)
    H_marg = 0.5 * np.log(E_var + Var_mu + noise_var)

    if name == 'field_mi':  return E_Hp   - F
    if name == 'graph_mi':  return H_marg - E_Hp
    if name == 'joint_mi':  return H_marg - F
    if name == 'field_eig':
        if cross_ssq is None:
            raise ValueError("field_eig needs cross_ssq")
        return (cross_ssq / (var_batch + noise_var)).mean(1)
    raise ValueError(name)
