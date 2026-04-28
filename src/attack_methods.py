import torch
import copy
import numpy as np
import math
import logging
from scipy.stats import norm, binom
from src.defend_methods import krum, median, trimmed, flame, multi_krum, agnostic

logger = logging.getLogger(__name__)


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
    Sign flip with adaptive scaling: ratio = (n - m) / m.
    Ensures attackers contribution cancels benign clients.
    """
    n = len(updates)
    ratio = (n - num_attackers) / num_attackers
    for i in range(num_attackers):
        updates[i] = -updates[i] * ratio
    
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
        act_updates = updates[num_attacks:]
        ideal_update = calculate_ideal_update(act_updates, filter, num_attacks, weights=weights[num_attacks:], 
                                               uploaded_models=uploaded_models[num_attacks:], noise_level=noise_level, m=m)
    elif pk == 'updates':
        act_updates = updates[num_attacks:]
        ideal_update = calculate_ideal_update(act_updates, 'avg', num_attacks, weights=weights, 
                                               uploaded_models=uploaded_models, noise_level=noise_level, m=m)
    elif pk == 'agr':
        act_updates = updates[:num_attacks]
        ideal_update = calculate_ideal_update(act_updates, filter, num_attacks, weights=weights[:num_attacks], 
                                               uploaded_models=uploaded_models, noise_level=noise_level, m=m)
    elif pk == 'none':
        act_updates = updates[:num_attacks]
        ideal_update = calculate_ideal_update(act_updates, 'avg', num_attacks, weights=weights[:num_attacks], 
                                               uploaded_models=uploaded_models, noise_level=noise_level, m=m)
    else:
        raise ValueError(f"Unknown prior knowledge type: {pk}")
    
    # 2. calculate the malicious update
    if mode == 'clipping':
        ideal_update = calculate_ideal_update(updates[num_attacks:], filter, num_attacks, weights=weights[num_attacks:], 
                                               uploaded_models=uploaded_models[num_attacks:], noise_level=noise_level, m=m)
        mask = (torch.sign(ideal_update) == vector_s).float()
        malicious_update = ideal_update * mask
        # malicious_update = malicious_update / torch.norm(malicious_update) * torch.norm(ideal_update)  
    elif mode == 'clipping_v5':
        # Personalized clipping
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
    elif mode == 'clipping_v8':
        # PK-aware aggregated flip: use ideal_update (from pk branch) as base signal.
        mask = (torch.sign(ideal_update) == vector_s).float()
        malicious_update = ideal_update * (2 * mask - 1)
        for i in range(num_attacks):
            updates[i] = malicious_update.clone()
        return updates, weights, vector_s
    elif mode == 'perturbation':
        perturbation = vector_s * torch.norm(ideal_update) * lamda * 0.1
        malicious_update = ideal_update + perturbation
        # malicious_update = malicious_update / torch.norm(malicious_update) * torch.norm(ideal_update) * 1.5
    elif mode == 'perturbation_v5':
        ideal_update = calculate_ideal_update(act_updates)
        vector_z = torch.randn_like(ideal_update)
        mask = (torch.sign(vector_z) == vector_s).float()
        delta_z = vector_z * mask
        perturbation = delta_z * torch.norm(ideal_update) * lamda * 0.1
        malicious_update = ideal_update + perturbation
        malicious_update = malicious_update / (torch.norm(malicious_update) + 1e-9) * torch.norm(ideal_update)
        
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

    # 4) optional noise (mimic roles.py: std = C_t * noise_level)    # maybe adding noise to ideal update is not necessary
    # if noise_level is not None and noise_level > 0 and C_t > 0:
    #     ideal = ideal + torch.normal(mean=0.0, std=float(C_t) * float(noise_level), size=ideal.size()).to(ideal.device)

    return ideal, benign_idx, float(C_t)



def calculate_ideal_update(updates, agregation_method='ag', num_attackers=0,
                           weights=None, uploaded_models=None, noise_level=0.0, m=None):
    """
    Calculate the ideal update based on the aggregation method.
    
    Parameters:
        updates (list of torch.Tensor): A list of tensors representing the updates from clients.
        agregation_method (str): The aggregation method to use.
        
    Returns:
        torch.Tensor: The aggregated ideal update.
    """
    if agregation_method == 'Multi_krum':
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
    elif agregation_method == 'Median':
        return median(updates)
    elif agregation_method == 'Trmean':
        return trimmed(updates, ratio=0.2)
    elif agregation_method == 'avg':
        ideal_update = torch.zeros_like(updates[0])
        for update in updates:
            ideal_update += update / len(updates)
        return ideal_update
    elif agregation_method == 'Flame':
        ideal_update, _, _ = ideal_update_flame(
            uploaded_models=uploaded_models,
            uploaded_updates=updates,
            uploaded_weights=weights,
            m=m,
            noise_level=noise_level
        )
        return ideal_update
    elif agregation_method == 'ag':
        return agnostic(updates)
    elif agregation_method in ('maud-norm', 'maud-cosine'):
        ideal_update = torch.zeros_like(updates[0])
        for update in updates:
            ideal_update += update / len(updates)
        return ideal_update
    else:
        ideal_update = torch.zeros_like(updates[0])
        for update in updates:
            ideal_update += update / len(updates)
        return ideal_update


def scale_attack(updates, num_attackers, scale):
    for i in range(num_attackers):
        updates[i] = updates[i] * scale
    return updates


def init_MPAF_model(global_model):
    MPAF_model = copy.deepcopy(global_model)
    for param in MPAF_model.parameters():
        torch.nn.init.normal_(param, mean=0.0, std=0.01)
    return MPAF_model


def min_max_attack(updates, weights, num_attackers):
    """
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
    # logger.info("min_max_attack try_count: {}".format(try_count))
    # logger.info("min_max_attack lamda: {}, max_d: {}".format(lamda, max_d))

    # 4. calculating the malicious update
    malicious_update = mean_update - lamda * deviation

    for i in range(num_attackers):
        updates[i] = malicious_update

    return updates, weights

