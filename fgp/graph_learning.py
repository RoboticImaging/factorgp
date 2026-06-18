import numpy as np
import torch
from torch.func import vmap, grad_and_value
from fgp.interpolate import fd_interpolate_cuts
from fgp.gbp import FactorGP

def build_fgp_full(inducing_xs, train_xs, train_zs, kappa_sq,
                   h, noise_var=0.01):
    """Builds a fully connected 4-grid factor graph
    
    Args:
        inducing_xs (np.array) : (n,d) array of inducing point locations (grid nodes)
        train_xs (np.array)    : (m,d) array of training data locations
        train_zs (np.array)    : (m,1) array of training data/observations
        kappa_sq (float)       : inverse 'lengthscale' of the approximated Matern kernel
        h (float)              : grid spacing
        noise_var (float)      : measurement noise variance
    
    Returns:
        fgp (FactorGP)         : Factor GP object
    """
    edges, nx, ny = build_candidate_edges(inducing_xs)
    d = 2
    a = kappa_sq * h**2 + 2 * d
    fgp = FactorGP(h=h)
    factor_id = 0

    for i, x in enumerate(inducing_xs):
        fgp.add_var_node(i, x,
                         prior_eta=np.zeros((1,1)) if i == 0 else None,
                         prior_lam=np.array([[1e-10]]) if i == 0 else None
                         )
        
    for (i, j) in edges:
        lam = np.array([[a/(2**d), -1.], [-1., a/(2**d)]]) / h**2
        fgp.add_gp_factor(factor_id,
                          np.zeros((2,1)),
                          lam,
                          [i, j]
                          )
        factor_id += 1

    for i, tx in enumerate(train_xs):
        x_obs = tx.reshape(1, -1)
        W_sp  = fd_interpolate_cuts(x_obs,
                                    np.array([nx, ny]),
                                    h,
                                    edges,
                                    edges
                                    )
        fgp.add_data_factor(z=train_zs[i],
                            x=x_obs,
                            adj_list=W_sp.indices,
                            jac=W_sp.data.reshape(-1, 1),
                            msmt_cov=noise_var
                            )
        factor_id += 1
    return fgp

def build_fgp_from_pi(inducing_xs, train_xs, train_zs, kappa_sq, grid_size,
                      h, edges, pi_mean, threshold=0.5, noise_var=0.01,
                      node_prior_lam=1.0):
    """Build FactorGP keeping only edges where probability of inclusion >= threshold.

    Args:
        inducing_xs (np.array) : (n,d) array of inducing point locations (grid nodes)
        train_xs (np.array)    : (m,d) array of training data locations
        train_zs (np.array)    : (m,1) array of training data/observations
        kappa_sq (float)       : inverse 'lengthscale' of the approximated Matern kernel
        grid_size (list)       : [nx, ny, (nz)] grid size in each direction/dim
        h (float)              : grid spacing
        edges (list)           : list of tuples describing all possible edges in graph
        pi_mean (np.array)     : array of edge probabilities ordered according to edges
        threshold (float)      : the threshold for edge inclusion
        noise_var (float)      : measurement noise variance
        node_prior_lam (float) : diagonal prior precision on each variable node in
                                 normalised z-space. Default 1.0 (prior std = 1σ),
                                 which prevents extreme predictions at under-sampled
                                 nodes while being dominated by data in observed regions.

    Returns:
        fgp (FactorGP)         : Factor GP object
    """
    nx = grid_size[0]
    ny = grid_size[1]
    d = len(grid_size)
    fgp = FactorGP(h=h)
    factor_id = 0
    for i, x in enumerate(inducing_xs):
        fgp.add_var_node(i,
                         x,
                         prior_eta=np.zeros((1,1)) if i == 0 else None,
                         prior_lam=np.array([[1e-10]]) if i == 0 else None)
        
    for e_idx, (i, j) in enumerate(edges):
        if pi_mean[e_idx] >= threshold:
            a = kappa_sq[e_idx] * h**2 + 2 * d
            lam = np.array([[a/(2**d), -1.], [-1., a/(2**d)]]) / h**2
            fgp.add_gp_factor(factor_id,
                              np.zeros((2,1)),
                              lam,
                              [i, j]
                              )
        factor_id += 1

    edges_after = [edges[i] for i in range(len(edges)) if pi_mean[i] > threshold]
    for i, tx in enumerate(train_xs):
        x_obs = tx.reshape(1, -1)
        W_sp  = fd_interpolate_cuts(x_obs,
                                    np.array([nx, ny]),
                                    h,
                                    edges,
                                    edges_after
                                    )
        fgp.add_data_factor(z=train_zs[i],
                            x=x_obs,
                            adj_list=W_sp.indices,
                            jac=W_sp.data.reshape(-1, 1),
                            msmt_cov=noise_var
                            )
        factor_id += 1
    return fgp

def build_candidate_edges(inducing_xs):
    """
    Builds all candidate edges, assuming a 4-connected grid
    
    Args:
        inducing_xs (np.array) : (n,d) array of inducing point locations (grid nodes)
    
    Returns:
        edges (list)           : list of all edges as tuples (i,j) if connecting node i to j
        nx (int)               : grid size in x
        ny (int)               : grid size in y

    """
    x_axis = np.unique(inducing_xs[:, 0])
    y_axis = np.unique(inducing_xs[:, 1])
    nx, ny = len(x_axis), len(y_axis)

    def flat(ix, iy):
        return iy * nx + ix

    edges = []
    for iy in range(ny):
        for ix in range(nx - 1):
            edges.append((flat(ix, iy), flat(ix + 1, iy)))   # horizontal
    for iy in range(ny - 1):
        for ix in range(nx):
            edges.append((flat(ix, iy), flat(ix, iy + 1)))   # vertical

    return edges, nx, ny

def build_Lambda(edges, pi, kappa_sq, n_nodes, h, d=2):
    """
    Assembles the relaxed GMRF precision matrix given edge inclusion probabilites
    (soft Lambda)

    Exactly matches matern_approx_uncorrelated at probability=1:
        Q_ii = sum_{e incident to i} pi_e * a/(2^d * h^2)
        Q_ij = -pi_e / h^2
    where a = kappa_sq * h^2 + 2*d.

    Args:
        edges (list)              : list of tuples of all candidate edges
        particles (torch array)   : array of edge inclusion probabilities, ordered according to edges
        n_nodes (int)             : number of nodes in the graph
        h (float)                 : grid spacing
        d (int)                   : grid dimension
    
    Returns:
        Lambda (np.array) : the precision matrix with entries weighted by edge probabilites 

    """
    E=len(edges)
    # a      = kappa_sq * h**2 + 2 * d
    # diag_c = (a / (2**d)) / h**2    # diagonal contribution per edge, matches prior_lam
    # off_c  = 1.0 / h**2             # off-diagonal magnitude
    dtype, device = pi.dtype, pi.device

    ei = torch.tensor([e[0] for e in edges], dtype=torch.long, device=device)
    ej = torch.tensor([e[1] for e in edges], dtype=torch.long, device=device)

    a      = kappa_sq * h**2 + 2 * d    # (E,)
    diag_c = (a / (2**d)) / h**2        # (E,)
    off_c  = 1.0 / h**2

    # Four contributions per edge: (i,i), (j,j), (i,j), (j,i).
    # scatter_add into a flat buffer avoids in-place indexed assignment,
    # which is incompatible with vmap.
    rows = torch.cat([ei, ej, ei, ej])
    cols = torch.cat([ei, ej, ej, ei])
    vals = torch.cat([pi * diag_c, pi * diag_c, -pi * off_c, -pi * off_c])

    flat   = torch.zeros(n_nodes * n_nodes, dtype=dtype, device=device)
    flat   = flat.scatter_add(0, rows * n_nodes + cols, vals)
    Lambda = flat.reshape(n_nodes, n_nodes)
    Lambda = Lambda + torch.eye(n_nodes, dtype=dtype, device=device) * 1e-6
    return Lambda

def gmrf_log_marginal_likelihood(Lambda, W, z, noise_var=0.01):
    """ Calculates the log marginal likelihood log p(D|G) given the
    GMRF/factor GP model.

    Args:
        Lambda (torch tensor) : the precision matrix of the GMRF
        W (torch tensor)      : the interpolation matrix for training data z
        z (torch tensor)      : training data/observations
        noise_var (float)     : measurement noise variance

    Returns:
        log p(D|G) (see latex note eqn 24 for exact form)
    """
    M = W.shape[0]

    # cholesky_ex returns info=0 on success instead of raising, which is
    # required for per-particle failure handling inside vmap.
    L,    info_L   = torch.linalg.cholesky_ex(Lambda)
    X       = torch.cholesky_solve(W.T.contiguous(), L)
    Sigma_y = W @ X + noise_var * torch.eye(M, dtype=Lambda.dtype,
                                             device=Lambda.device)
    L_sy, info_Lsy = torch.linalg.cholesky_ex(Sigma_y)

    log_det = 2.0 * L_sy.diagonal().log().sum()
    z_vec   = z.reshape(-1, 1)
    alpha   = torch.cholesky_solve(z_vec, L_sy)
    quad    = (z_vec * alpha).sum()
    result  = -0.5 * quad - 0.5 * log_det

    failed = (info_L != 0) | (info_Lsy != 0)
    return torch.where(failed, torch.zeros_like(result), result)

def log_posterior(particle, edges, n_nodes, h,
                  z, W_t, noise_var, log_p1, log_p0,
                  log_kappa_mean, log_kappa_std):
    """ Calculates the log posterior log p(G|D) given the Bernoulli prior
    and the GMRF.

    Args:
        particle (torch tensor)  : SVGD particle, array of logits per edge
        edges (list)             : list of tuples describing candidate edge set
        n_nodes (int)            : number of nodes in graph
        h (float)                : grid spacing
        z (torch tensor)         : training data/observations
        W_t (torch tensor)       : training data to inducing grid interpolation matrix
        noise_var (float)        : measurement noise variance
        log_p1 (torch tensor)    : Bernoulli prior log probability of inclusion
        log_p0 (torch tensor)    : Bernoulli prior log probability of exclusion (1-log_p1)

    Returns:
        log_p (torch tensor)   : log p(G|D) = log p(D|G) + log p(G)
    """
    E = len(edges)
    pi = particle[:E]
    log_kappa_sq = particle[E:]
    kappa_sq = torch.exp(log_kappa_sq)

    Lambda = build_Lambda(edges, pi, kappa_sq, n_nodes, h)
    log_ml = gmrf_log_marginal_likelihood(Lambda, W_t, z, noise_var)

    # Bernoulli prior: spike
    log_pr_spike = (pi * log_p1 + (1 - pi) * log_p0)
    
    # Log normal prior: slab
    log_pr_slab = -0.5 * ((log_kappa_sq - log_kappa_mean) ** 2 / log_kappa_std**2)
    log_pr_slab -= torch.log(log_kappa_std * torch.sqrt(torch.tensor(2 * np.pi)))

    log_prior = (log_pr_spike + log_pr_slab).sum()
    
    return log_ml + log_prior

def svgd_phi(particles, log_p_grads):
    """
    Calculates the SVGD optimal vector field $\phi$, uses exponential kernel

    Args:
        particles (torch.tensor)   : the SVGD particles/ edge inclusion logits
        log_p_grads (torch.tensor) : gradient of log p(G|D) w.r.t. particles

    Returns:
        phi (torch.tensor) : the SVGD optimal perturbation direction
    """
    K = particles.shape[0] #(K=n_particles, E=n_edges)
    with torch.no_grad():
        # broadcasting to calculate pairwise diff
        diff   = particles[:,None,:] - particles[None,:,:]        # (K, K, E)
        d2     = (diff ** 2).sum(dim=2)                           # (K, K)
        # rbf kernel
        med2   = torch.median(d2)
        h2     = torch.clamp(med2 / torch.log(torch.tensor(K + 1.0)),
                             min=0.1)
        K_mat  = torch.exp(-d2 / h2)                             # (K, K)
        grad_K = -2.0 * diff / h2 * K_mat.unsqueeze(2)           # (K, K, E)
        # first term of SVGD vector field pushes to density, k(G, \cdot)\nabla_G log_p(G|D)
        drive   = torch.einsum('li,le->ie', K_mat, log_p_grads) / K
        # second term pushes particles apart, \nabda_G k(G, \cdot) 
        repulse = grad_K.sum(dim=0) / K
    return drive + repulse, K_mat

def learn_graph_structure(inducing_xs, train_xs, train_zs, h,
                           noise_var=0.01,
                           n_particles=30,
                           n_iter=300,
                           lr=0.05,
                           logit_init=0.1,
                           logit_std = 0.1,
                           log_kappa_mean=1.0,
                           log_kappa_std=1.0,
                           tau_init=1.0,
                           tau_min=0.1,
                           tau_anneal=0.02,
                           threshold=0.5,
                           snapshot_every=20,
                           verbose=True,
                           device = 'cpu'):
    """
    Runs SVGD to learn the posterior p(G|D) over all edges.

    The model is:
        log p(G|D) = log p(D|G) + log p(G)
        log p(G)   = sum_e [ e_ij * log pi + (1-e_ij) * log(1-pi) ]
                     independent multi-var Bernoulli prior, sum over all edges
        log p(D|G) comes from the GMRF model

    Args:
        inducing_xs (np.array) : (n,d) array of inducing point locations (grid nodes)
        train_xs (np.array)    : (m,d) array of training data locations
        train_zs (np.array)    : (m,1) array of training data/observations
        kappa_sq (float)       : 'lengthscale' of the approximated Matern kernel (smaller is longer correlation)
        h (float)              : grid spacing
        noise_var (float)      : measurement noise variance
        n_particles (int)      : the number of particles to run SVGD with
        n_iter (int)           : number of iterations for SVGD
        lr (float)             : SVGD learning rate
        lambda_prior (float)   : prior on the logits (0 is uninformative)
        tau_init (float)       : initial temp for the gumbel softmax
        tau_min (float)        : minimum temp possible during annealing
        tau_anneal (float)     : annealing rate
        threshold (float)      : threshold on edge probability for hard inclusion
        snapshot_every (int)   : how many iterations to save SVGD progress with
        verbose (bool)         : True to print SVGD updates
    
    Returns:
        final_pi (np.array)    : the final probabilities of inclusions for all particles
        edges (list)           : all candidate edges represented as tuples
        snapshots (list)       : snapshots of SVGD progress, (iteration, pi_mean, pi_std)
        history (list)         : history of average log posterior values
    """
    dtype  = torch.float64
    device = torch.device(device)

    edges, nx, ny = build_candidate_edges(inducing_xs)
    n_nodes = inducing_xs.shape[0]
    E       = len(edges)

    z   = torch.tensor(train_zs, dtype=dtype, device=device).reshape(-1)
    W   = fd_interpolate_cuts(train_xs, [nx,ny], h, edges, edges)
    W_t = torch.tensor(W.toarray(), dtype=dtype, device=device)

    # prior prb of inclusion
    p1     = 1.0 / (1.0 + np.exp(-logit_init))
    # init log prior for inclusion
    log_p1 = torch.tensor(np.log(p1 + 1e-10),   dtype=dtype, device=device)
    # log prior for exclusion
    log_p0 = torch.tensor(np.log(1-p1 + 1e-10), dtype=dtype, device=device)

    # optimise the logits for unrestricted domain, then map back to probabilities with sigmoid
    particles = torch.zeros(n_particles, 2*E, dtype=dtype, device=device)
    particles[:,:E]  = logit_init + torch.randn(n_particles, E, dtype=dtype, device=device) * 0.1
    particles[:,E:]  = log_kappa_mean + torch.randn(n_particles, E, dtype=dtype, device=device) * 0.1
    if verbose:
        print(f"SVGD: {n_particles} particles, {E} edges "
              f"({nx}x{ny} grid), {n_iter} iters")
    
    snapshots = []

    def per_particle(logit_p, gumbel_k):
        pi_soft = torch.sigmoid((logit_p[:E] + gumbel_k) / tau)
        pi_hard = (pi_soft > threshold).float().detach()
        pi_st   = pi_soft + (pi_hard - pi_soft).detach()
        pc      = torch.cat([pi_st, logit_p[E:]])
        return log_posterior(pc, edges, n_nodes, h,
                             z, W_t, noise_var, log_p1, log_p0,
                             log_kappa_mean, log_kappa_std)

    batched = vmap(grad_and_value(per_particle))

    for it in range(n_iter):
        # Anneal temperature
        tau = max(tau_min, tau_init * np.exp(-tau_anneal * it))

        u = torch.rand(n_particles, E, dtype=dtype, device=device)
        gumbel_noise = -torch.log(-torch.log(u + 1e-8) + 1e-8)  # (n_particles, E)

        grads, log_p_vals = batched(particles, gumbel_noise)

        # Clip gradients — prevents large steps early when likelihood
        # gradient is steep and particles are far from the posterior
        grads = torch.clamp(grads, -5.0, 5.0)

        phi, K_mat   = svgd_phi(particles, grads)
        particles    = particles + lr * phi
            

        if (it % snapshot_every == 0 or it == n_iter - 1):
            pi_np  = torch.sigmoid(particles[:,:E]).detach().cpu().numpy()
            pi = torch.sigmoid(particles[:,:E])  # (n_particles, E)
            prior_grads = (log_p1 - log_p0) * pi * (1 - pi)
            # Likelihood gradient is just the difference
            ml_grads = grads[:,:E] - prior_grads
            n_kept = (pi_np.mean(axis=0) > threshold).sum()
            pm = pi_np.mean(axis=0)
            ps = pi_np.std(axis=0)
            snapshots.append((it, pm.copy(), ps.copy(), K_mat.cpu().numpy(),log_p_vals.detach().cpu().numpy()))
            if verbose:
                print(f"  iter {it:4d}  mean log p={log_p_vals.mean().item():10.3f}  "
                      f"edges kept={n_kept}/{E}  "
                      f"mean pi={pi_np.mean():.3f}  "
                      f"mean |ml grad|: {ml_grads.abs().mean():.6f}  "
                      f"mean |prior grad|: {prior_grads.abs().mean():.6f}"
                      )

    final_pi = torch.sigmoid(particles[:,:E]).detach().cpu().numpy()
    final_kappa = torch.exp(particles[:,E:]).detach().cpu().numpy()

    # def _eval_clean(logit_p):
    #     pi_hard = (logit_p[:E].sigmoid() > threshold).double()
    #     pc = torch.cat([pi_hard, logit_p[E:]])
    #     return log_posterior(pc, edges, n_nodes, h,
    #                          z, W_t, noise_var, log_p1, log_p0,
    #                          log_kappa_mean, log_kappa_std)

    # with torch.no_grad():
    #     final_log_p = torch.stack([_eval_clean(particles[i])
                                #    for i in range(n_particles)]).cpu().numpy()

    return final_pi, final_kappa, edges, snapshots#, final_log_p