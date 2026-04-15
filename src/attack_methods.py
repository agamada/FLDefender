import torch
import copy
import numpy as np
import math
import logging
from scipy.stats import norm, binom
from src.defend_methods import krum, median, trimmed, flame, multi_krum

logger = logging.getLogger(__name__)

def min_max_attack(updates, weights, num_attackers):
    """
    Perform a Min-Max attack on the updates in a federated learning setting.
    This function modifies the updates to simulate malicious behavior by 
    constructing a malicious update that maximizes the deviation from the 
    benign updates while remaining within the maximum L2-norm distance.
    Args:
        updates (list of torch.Tensor): A list of update tensors from clients.
        weights (list of float): A list of weights corresponding to the updates.
        num_attackers (int): The number of attackers to simulate.
    Returns:
        tuple: A tuple containing:
            - updates (list of torch.Tensor): The modified updates with malicious updates injected.
            - weights (list of float): The weights corresponding to the updates (unchanged).
    """
    # 1. get the update tensor
    all_updates = torch.stack(updates)
    
    # 2. calculating the max l2-norm of the updates 
    # distances = torch.norm(all_updates[:,None,:] - all_updates[None,:,:], dim=2) ** 2
    # max_distance = torch.max(distances)
    norms = torch.sum(all_updates ** 2, dim=1, keepdim=True)  # [N,1]
    distances = norms + norms.T - 2 * (all_updates @ all_updates.T)
    max_distance = distances.max()

    # 3. constructing the malicious update via binary searching, V(m) = V(mean) - s*V(p)
    lamda = 5
    threshold_diff = 1e-5
    step = lamda
    lamda_succ = 0
    try_count = 0

    mean_update = torch.mean(all_updates, dim=0)
    deviation = mean_update / torch.norm(mean_update) # unit vector, dir opp to good dir

    while abs(lamda_succ - lamda) > threshold_diff:  
        mal_update = mean_update - lamda * deviation
        distance = torch.norm(all_updates - mal_update, dim=1) ** 2
        max_d = torch.max(distance)

        if max_d <= max_distance:
            lamda_succ = lamda
            lamda = lamda + step / 2
        else:
            lamda = lamda - step / 2
        
        step = step / 2
        try_count += 1
    logger.info("min_max_attack try_count: {}".format(try_count))
    logger.info("min_max_attack lamda: {}, max_d: {}".format(lamda, max_d))

    # 4. calculating the malicious update
    malicious_update = mean_update - lamda * deviation

    for i in range(num_attackers):
        updates[i] = malicious_update

    return updates, weights

def vector_s_minmax_update(updates, num_attackers, vector_s, use_benign_prior=False, init_scale=5.0, threshold_diff=1e-5):
    """
    Min-max style binary search using vector_s as direction.
    Distance constraint uses attacker updates only (updates[:num_attackers]).
    Mean anchor follows prior knowledge:
    - use benign mean if use_benign_prior is True
    - otherwise use attacker mean
    """
    atk_updates = updates[:num_attackers]
    all_updates = torch.stack(atk_updates)

    norms = torch.sum(all_updates ** 2, dim=1, keepdim=True)
    distances = norms + norms.T - 2 * (all_updates @ all_updates.T)
    max_distance = distances.max()

    if use_benign_prior and len(updates) > num_attackers:
        mean_update = torch.mean(torch.stack(updates[num_attackers:]), dim=0)
    else:
        mean_update = torch.mean(all_updates, dim=0)
    direction = vector_s.to(mean_update.device).to(mean_update.dtype)
    direction = direction / (torch.norm(direction) + 1e-9)

    scale = float(init_scale) if init_scale is not None else 5.0
    if scale <= 0:
        scale = 5.0

    step = scale
    scale_succ = 0.0
    try_count = 0

    while abs(scale_succ - scale) > threshold_diff and try_count < 64:
        mal_update = mean_update - scale * direction
        distance = torch.norm(all_updates - mal_update, dim=1) ** 2
        max_d = torch.max(distance)

        if max_d <= max_distance:
            scale_succ = scale
            scale = scale + step / 2
        else:
            scale = scale - step / 2

        step = step / 2
        try_count += 1

    malicious_update = mean_update - scale_succ * direction
    logger.info("vector_s_minmax try_count: %s", try_count)
    logger.info("vector_s_minmax scale: %s", scale_succ)

    return malicious_update, scale_succ

def LIE_attack(updates, weights, num_attackers, num_clients):
    """
    LIE_attack computes a malicious update based on the uploaded updates from clients.
    Parameters:
        uploaded_updates (list of torch.Tensor): A list of tensors representing the updates 
            uploaded by clients.
        num_clients (int): The total number of clients participating in the update process.
        num_attackers (int): The number of malicious clients attempting to influence the update.
    Returns:
        torch.Tensor: A tensor representing the malicious update calculated based on the mean 
        and standard deviation of the uploaded updates, adjusted by a statistical factor.
    Notes:
        This function assumes that the uploaded updates are in the form of PyTorch tensors and 
        that the necessary libraries (torch and scipy.stats.norm) are imported.
    """
    # 1. get the update tensor
    all_updates = torch.stack(updates)

    # 2. calculate the mean and standard deviation of the updates
    mean_update = torch.mean(all_updates, dim=0)
    std_update = torch.std(all_updates, dim=0)

    # 3. calculate malicious update and model
    s = math.floor(num_clients / 2 + 1) - num_attackers
    p = (num_clients - num_attackers - s) / (num_clients - num_attackers)
    p = min(max(p, 1e-6), 1 - 1e-6)
    z = norm.ppf(p)
    # print("z is", z)

    # 4. return the malicious update
    malicious_update = mean_update + z * std_update 

    for i in range(num_attackers):
        updates[i] = malicious_update

    return updates, weights

def sign_flip_attack(updates, weights, num_attackers):
    """
    sign_flip_attack flips the sign of the updates from a specified number of attackers.
    Parameters:
        updates (list of torch.Tensor): A list of tensors representing the updates from clients.
        weights (list): A list of weights corresponding to the updates.
        num_attackers (int): The number of malicious clients whose updates will be flipped.
    Returns:
        tuple: (modified updates, weights)
    """
    for i in range(num_attackers):
        updates[i] = -updates[i] * 2
    
    return updates, weights

def global_sign_flip_attack(updates, weights, num_attackers):
    """
    Global sign flip: compute mean of attacker updates, flip sign, assign to all attackers.
    """
    mal_updates = updates[:num_attackers]
    mean_mal = torch.mean(torch.stack(mal_updates), dim=0)
    flipped = -mean_mal
    for i in range(num_attackers):
        updates[i] = flipped.clone()
    return updates, weights, None

def enhanced_sign_flip_attack(updates, weights, num_attackers):
    """
    enhenced_sign_flip_attack first computes weighted average of controlled clients' updates,
    then flips the sign of this aggregate and uses it for all malicious clients.
    Parameters:
        updates (list of torch.Tensor): A list of tensors representing the updates from clients.
        weights (list): A list of weights corresponding to the updates.
        num_attackers (int): The number of malicious clients whose updates will be flipped.
    Returns:
        tuple: (modified updates, weights)
    """
    malicious_updates = updates[:num_attackers]
    malicious_weights = weights[:num_attackers]
    weighted_avg = sum(w * u for w, u in zip(malicious_weights, malicious_updates)) / sum(malicious_weights)
    
    flipped_avg = -weighted_avg
    for i in range(num_attackers):
        updates[i] = flipped_avg
    
    return updates, weights

def random_attack(updates, weights, num_attackers):
    """
    Generates random vectors scaled relative to benign updates' magnitude.
    Parameters:
        updates (list of torch.Tensor): A list of tensors representing the updates from clients.
        weights (list): A list of weights corresponding to the updates.
        num_attackers (int): The number of malicious clients to be selected for the attack.
    Returns:
        tuple: (modified updates, weights)
    """
    if num_attackers <= 0:
        return updates, weights
    
    # Calculate average norm
    norms = torch.stack([torch.norm(u) for u in updates])
    avg_norm = torch.mean(norms)
    
    random_vector = torch.randn_like(updates[0]) 
    random_vector = random_vector / torch.norm(random_vector) 
    random_vector = random_vector * avg_norm

    for i in range(num_attackers):
        updates[i] = random_vector
        
    return updates, weights

def CAMP_attack(updates, weights, num_attacks, mode, filter, vector_s, lamda, pk='all',
                uploaded_models=None, noise_level=0.0, m=None):
    """
    Implements the CAMP (Clipping and Perturbation) attack for federated learning.
    Args:
        updates (list): A list of model updates from clients.
        weights (list): A list of weights corresponding to the updates.
        num_attacks (int): The number of malicious clients to simulate.
        mode (str): The attack mode, either 'clipping' or 'perturbation'.
        filter (str): The filter type used for calculating the ideal global update.
        vector_s (torch.Tensor): A binary vector used for masking during the attack.
        lamda (float): A scaling factor for the perturbation in 'perturbation' mode.
        pk (str, optional): The prior knowledge type. Options are:
            - 'all': full prior knowledge.
            - 'updates': have prior knowledge about all updates.
            - 'agr': have prior knowledge about the specified filter used by server.
            - 'none': poor prior knowledge.
            Default is 'all'.
    Returns:
        tuple: A tuple containing:
            - updates (list): The modified list of updates with malicious updates injected.
            - weights (list): The original list of weights (unchanged).
    """
    # 1. calculate the global update
    if pk == 'all':
        ideal_update = calculate_ideal_update(updates[num_attacks:], filter, num_attacks, weights=weights[num_attacks:], 
                                               uploaded_models=uploaded_models[num_attacks:], noise_level=noise_level, m=m)
    elif pk == 'updates':
        ideal_update = calculate_ideal_update(updates, 'avg', num_attacks, weights=weights, 
                                               uploaded_models=uploaded_models, noise_level=noise_level, m=m)
    elif pk == 'agr':
        ideal_update = calculate_ideal_update(updates[:num_attacks], filter, num_attacks, weights=weights[:num_attacks], 
                                               uploaded_models=uploaded_models, noise_level=noise_level, m=m)
    elif pk == 'none':
        ideal_update = calculate_ideal_update(updates[:num_attacks], 'avg', num_attacks, weights=weights[:num_attacks], 
                                               uploaded_models=uploaded_models, noise_level=noise_level, m=m)
    else:
        raise ValueError(f"Unknown prior knowledge type: {pk}")
    
    # 2. calculate the malicious update
    if mode == 'clipping':
        mask = (torch.sign(ideal_update) == vector_s).float()
        malicious_update = ideal_update * mask
        # malicious_update = malicious_update / torch.norm(malicious_update) * torch.norm(ideal_update) * 1.5  
    elif mode == 'clipping_v5':
        # Personalized clipping: each attacker flips its own update guided by vector_s
        # This preserves diversity (like sign_flip) while steering direction (like CAMP)
        for i in range(num_attacks):
            mask_i = (torch.sign(updates[i]) == vector_s).float()
            updates[i] = updates[i] * (2 * mask_i - 1)
        return updates, weights, vector_s
    elif mode == 'clipping_v6':
        # Mean-based selective flip: compute mean of attacker updates, then flip dims aligned with vector_s
        # Analogous to enhanced_sign_flip vs sign_flip, but with directional guidance
        mean_update = torch.mean(torch.stack(updates[:num_attacks]), dim=0)
        mask = (torch.sign(mean_update) == vector_s).float()
        malicious_update = mean_update * (2 * mask - 1)
        for i in range(num_attacks):
            updates[i] = malicious_update.clone()
        return updates, weights, vector_s
    elif mode == 'clipping_v7':
        # Prior-aware selective flip: use ideal_update (from pk branch) as base signal.
        # Keep dims aligned with vector_s, flip the rest.
        mask = (torch.sign(ideal_update) == vector_s).float()
        malicious_update = ideal_update * (2 * mask - 1)
    elif mode == 'clipping_v8':
        # PK-aware aggregated flip: select mean anchor based on prior knowledge,
        # then selectively flip dimensions guided by vector_s.
        if pk in ['all', 'updates'] and len(updates) > num_attacks:
            mean_update = torch.mean(torch.stack(updates[num_attacks:]), dim=0)
        else:
            mean_update = torch.mean(torch.stack(updates[:num_attacks]), dim=0)
        mask = (torch.sign(mean_update) == vector_s).float()
        malicious_update = mean_update * (2 * mask - 1)
        for i in range(num_attacks):
            updates[i] = malicious_update.clone()
        return updates, weights, vector_s
    elif mode == 'perturbation':
        vector_z = torch.randn_like(ideal_update)
        mask = (torch.sign(vector_z) == vector_s).float()
        delta_z = vector_z * mask
        malicious_update = ideal_update + delta_z * lamda
        # malicious_update = malicious_update / torch.norm(malicious_update) * torch.norm(ideal_update) * 1.5
    elif mode == 'perturbation_v5':
        # min-max base + deterministic fixed-direction CAMP bias (NO jitter)
        # Step 1: get min-max base via calculate_ideal_update
        base_update = calculate_ideal_update(updates[:num_attacks], 'min-max', num_attacks, weights=weights)
        
        # Step 2: add fixed-direction perturbation: base + vector_s * scale
        # vector_s is ±1 per dim, fixed across rounds -> cumulative bias
        # Scale relative to base norm, controlled by lamda
        perturbation = vector_s * torch.norm(base_update) * lamda * 0.1
        malicious_update = base_update + perturbation
        
        # Renormalize to base norm to preserve distance property
        malicious_update = malicious_update / (torch.norm(malicious_update) + 1e-9) * torch.norm(base_update)
        
        # All attackers get IDENTICAL update (no jitter!)
        for i in range(num_attacks):
            updates[i] = malicious_update.clone()
        return updates, weights, vector_s
    elif mode == 'perturbation_v6':
        # Your idea: use attacker-mean anchor and vector_s as search direction.
        init_scale = lamda if lamda is not None and lamda > 0 else 5.0
        use_benign_prior = pk in ['all', 'updates']
        malicious_update, _ = vector_s_minmax_update(
            updates=updates,
            num_attackers=num_attacks,
            vector_s=vector_s,
            use_benign_prior=use_benign_prior,
            init_scale=init_scale,
        )

        for i in range(num_attacks):
            updates[i] = malicious_update.clone()
        return updates, weights, vector_s
    
    else:
        raise ValueError(f"Unknown CAMP mode: {mode}")
    
    # 3. adaptive clipping by median norm
    if pk in ['all', 'updates']:
        ref_updates = updates[num_attacks:]
    else:
        ref_updates = updates[:num_attacks]
    norms = [torch.norm(u, p=2).item() for u in ref_updates]
    C = float(np.max(np.array(norms)))  if len(norms) > 0 else 0.0

    if C > 0 :
        malicious_update = malicious_update / torch.norm(malicious_update, p=2) * C
    
    # # 4. update the malicious clients' updates
    # for i in range(num_attacks):
    #     jitter = torch.randn_like(malicious_update) * 0.01 * torch.norm(malicious_update)
    #     updates[i] = (malicious_update + jitter).clone()
    
    return updates, weights, vector_s


def ideal_update_flame(uploaded_models, uploaded_updates, uploaded_weights=None, m=0, noise_level=0.0):
    # 1) copy updates to avoid in-place side effects
    tmp_updates = [u.clone() for u in uploaded_updates]
    weights = None if uploaded_weights is None else list(uploaded_weights)

    # 2) run your flame (clusters on model params + clips benign updates in-place)
    benign_idx, C_t = flame(uploaded_models, tmp_updates, m)

    # 3) FedAvg over benign
    sel_updates = [tmp_updates[i] for i in benign_idx]
    if weights is None:
        ideal = torch.zeros_like(sel_updates[0])
        for u in sel_updates:
            ideal += u / len(sel_updates)
    else:
        sel_weights = [weights[i] for i in benign_idx]
        total_w = float(sum(sel_weights))
        ideal = torch.zeros_like(sel_updates[0])
        for w, u in zip(sel_weights, sel_updates):
            ideal += (float(w) / total_w) * u

    # 4) optional noise (mimic roles.py: std = C_t * noise_level)
    if noise_level is not None and noise_level > 0 and C_t > 0:
        ideal = ideal + torch.normal(mean=0.0, std=float(C_t) * float(noise_level), size=ideal.size()).to(ideal.device)

    return ideal, benign_idx, float(C_t)



def calculate_ideal_update(updates, agregation_method, num_attackers,
                           weights=None, uploaded_models=None, noise_level=0.0, m=None):
    """
    Calculate the ideal update based on the aggregation method.
    
    Parameters:
        updates (list of torch.Tensor): A list of tensors representing the updates from clients.
        agregation_method (str): The aggregation method to use.
        
    Returns:
        torch.Tensor: The aggregated ideal update.
    """
    if agregation_method == 'multi-krum':
        selected_indices = multi_krum(updates, num_attackers)
        selected_updates = [updates[i] for i in selected_indices]
        ideal_update = torch.zeros_like(selected_updates[0])
        for update in selected_updates:
            ideal_update += update / len(selected_updates)
        return ideal_update
        # ideal_update = torch.zeros_like(updates[0])
        # for update in updates:
        #     ideal_update += update / len(updates)
        # return ideal_update
    elif agregation_method == 'median':
        return median(updates)
    elif agregation_method == 'trmean':
        return trimmed(updates, ratio=0.2)
    elif agregation_method == 'avg':
        ideal_update = torch.zeros_like(updates[0])
        for update in updates:
            ideal_update += update / len(updates)
        return ideal_update
    elif agregation_method == 'min-max':
        # Run min_max_attack to get the malicious direction
        tmp_updates = [u.clone() for u in updates]
        tmp_weights = [1.0] * len(updates)
        tmp_updates, _ = min_max_attack(tmp_updates, tmp_weights, len(updates))
        return tmp_updates[0]
    elif agregation_method in ('maud-norm', 'maud-cosine'):
        ideal_update = torch.zeros_like(updates[0])
        for update in updates:
            ideal_update += update / len(updates)
        return ideal_update
    elif agregation_method == 'flame':
        ideal_update, _, _ = ideal_update_flame(
            uploaded_models=uploaded_models,
            uploaded_updates=updates,
            uploaded_weights=weights,
            m=m,
            noise_level=noise_level
        )
        return ideal_update
    else:
        raise ValueError(f"Unknown aggregation method: {agregation_method}")


def scale_attack(updates, num_attackers, scale):
    for i in range(num_attackers):
        updates[i] = updates[i] * scale
    return updates


def init_MPAF_model(global_model):
    MPAF_model = copy.deepcopy(global_model)
    for param in MPAF_model.parameters():
        torch.nn.init.normal_(param, mean=0.0, std=0.01)
    return MPAF_model


def _binom_k(d: int, q: float = 0.99, p: float = 0.5) -> int:
    """
    Binomial(d, p) quantile threshold used to judge sign-alignment feedback.
    The official repo hard-codes k_99 for several d; here we generalize via binom.ppf. :contentReference[oaicite:5]{index=5}
    """
    d = int(d)
    if d <= 0:
        return 0
    return int(binom.ppf(q, d, p))

def _ensure_pm_one(vec: torch.Tensor) -> torch.Tensor:
    """
    Convert a tensor to +/-1 (no zeros). Zero entries are mapped to +1.
    """
    s = torch.sign(vec)
    s[s == 0] = 1
    return s


def poisonedfl_attack(
    updates,
    weights,
    num_attackers,
    state: dict,
    round_idx: int,
    scaling_factor: float = 1e5,
    adjust_period: int = 50,
    # optional "feedback" signals for dynamic magnitude adjustment:
    global_model_vec: torch.Tensor = None,
    global_model_vec_prev_period: torch.Tensor = None,
    # optional: server aggregated update/grad from last round
    last_global_grad: torch.Tensor = None,
    jitter_ratio: float = 0.0,
    eps: float = 1e-9,
):
    """
    PoisonedFL attack (PyTorch version) with:
      1) multi-round consistency via a persistent direction (fixed_rand) and history
      2) dynamic attack magnitude adjustment via sign-alignment feedback (optional)

    Args:
        updates: list[torch.Tensor], each is a flattened update vector (same shape)
        weights: list[float]
        num_attackers: int, number of malicious clients (assumed to be the first num_attackers)
        state: dict, must be persisted across rounds. This function will read/write:
            - state["fixed_rand"]: torch.Tensor in {+1,-1}^d
            - state["history"]: torch.Tensor (malicious update history) or None
            - state["sf"]: float, current scaling factor
        round_idx: int, current FL round index (0-based or 1-based ok; only affects modulo)
        scaling_factor: float, base magnitude (PoisonedFL uses large sf and adapts it) :contentReference[oaicite:6]{index=6}
        adjust_period: int, how often to apply feedback adjustment (official code uses 50) :contentReference[oaicite:7]{index=7}
        global_model_vec / global_model_vec_prev_period:
            flattened model params at current round and (round_idx-adjust_period) round.
            If provided, enable dynamic magnitude adjustment.
        last_global_grad:
            optional last-round aggregated update direction for scale shaping (like official "last_grad"). :contentReference[oaicite:8]{index=8}
        jitter_ratio:
            add tiny noise: jitter_ratio * ||mal_update|| * N(0,1). Default 0 for exact consistency.

    Returns:
        (updates, weights, state)
    """
    if num_attackers <= 0:
        return updates, weights, state

    device = updates[0].device
    d = int(updates[0].numel())

    # ---- init state ----
    if state is None:
        state = {}

    if "fixed_rand" not in state:
        # persistent sign pattern (multi-round consistency anchor)
        fixed_rand = _ensure_pm_one(torch.randn(d, device=device))
        state["fixed_rand"] = fixed_rand
    else:
        state["fixed_rand"] = state["fixed_rand"].to(device)

    if "sf" not in state:
        state["sf"] = float(scaling_factor)

    fixed_rand = state["fixed_rand"]
    sf = float(state["sf"])

    # ---- dynamic magnitude adjustment (optional feedback) ----
    # Official code checks every 50 rounds and may shrink sf by 0.7 under certain alignment conditions. :contentReference[oaicite:9]{index=9}
    if (
        adjust_period is not None
        and adjust_period > 0
        and (round_idx % adjust_period == 0)
        and (global_model_vec is not None)
        and (global_model_vec_prev_period is not None)
    ):
        total_update = global_model_vec.to(device) - global_model_vec_prev_period.to(device)
        total_update = total_update.view(-1)
        # avoid all-zero diff corner case
        if torch.all(total_update == 0):
            total_update = global_model_vec.to(device).view(-1)

        current_sign = _ensure_pm_one(torch.sign(total_update))
        aligned_dim_cnt = int((current_sign == fixed_rand).sum().item())

        k_99 = _binom_k(d, q=0.99, p=0.5)
        if aligned_dim_cnt < k_99 and sf * 0.7 >= 0.5:
            sf = sf * 0.7

        logger.info(f"[PoisonedFL] round={round_idx} aligned={aligned_dim_cnt}/{d}, k_99={k_99}, sf={sf:.6g}")

    # ---- multi-round consistency core ----
    history = state.get("history", None)

    if history is None:
        # "Start from round 2" in official code; here we bootstrap history from current malicious mean. :contentReference[oaicite:10]{index=10}
        mal_seed = torch.mean(torch.stack(updates[:num_attackers]), dim=0).detach()
        if torch.norm(mal_seed) < eps:
            mal_seed = torch.mean(torch.stack(updates), dim=0).detach()
        history = mal_seed.view(-1).clone()

    history = history.view(-1).detach()
    history_norm = torch.norm(history) + eps

    # official code shapes per-dim scale using last_grad; if missing, fall back to |history| shaping. :contentReference[oaicite:11]{index=11}
    if last_global_grad is not None:
        last_g = last_global_grad.view(-1).to(device).detach()
        last_g_norm = torch.norm(last_g) + eps
        # scale_i = || history_i - last_g_i * ||history||/||last_g|| || (since it's 1-dim per entry)
        scale = torch.abs(history - last_g * (history_norm / last_g_norm))
    else:
        scale = torch.abs(history)

    # deviation = scale * fixed_rand / ||scale||
    scale_norm = torch.norm(scale) + eps
    deviation = (scale * fixed_rand) / scale_norm

    lam = sf * float(history_norm.item())
    malicious_update = (lam * deviation).view_as(updates[0])

    if jitter_ratio is not None and jitter_ratio > 0:
        malicious_update = malicious_update + torch.randn_like(malicious_update) * (jitter_ratio * torch.norm(malicious_update))

    # inject
    for i in range(num_attackers):
        updates[i] = malicious_update.clone()

    # update history for next round (keep it aligned across rounds)
    state["history"] = malicious_update.view(-1).detach().clone()
    state["sf"] = float(sf)

    return updates, weights, state