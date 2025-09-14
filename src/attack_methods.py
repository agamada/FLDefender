import torch
import copy
import math
import logging
from scipy.stats import norm
from src.defend_methods import krum, median, trimmed

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
    z = norm.ppf((num_clients - num_attackers - s) / (num_clients - num_attackers))
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
        updates[i] = -updates[i]
    
    return updates, weights

def enhenced_sign_flip_attack(updates, weights, num_attackers):
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

def CAMP_attack(updates, weights, num_attacks, mode, filter, vector_s, lamda, pk='all'):
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
        ideal_update = calculate_ideal_update(updates, filter, num_attacks)
    elif pk == 'updates':
        ideal_update = calculate_ideal_update(updates, 'avg', num_attacks)
    elif pk == 'agr':
        ideal_update = calculate_ideal_update(updates[:num_attacks], filter, num_attacks)
    elif pk == 'none':
        ideal_update = calculate_ideal_update(updates[:num_attacks], 'avg', num_attacks)
    else:
        raise ValueError(f"Unknown prior knowledge type: {pk}")
    
    # 2. calculate the malicious update
    if mode == 'clipping':
        mask = (torch.sign(ideal_update) == vector_s).float()
        malicious_update = ideal_update * mask
        # malicious_update = malicious_update / torch.norm(malicious_update) * torch.norm(ideal_update) * 1.5  
    elif mode == 'perturbation':
        vector_z = torch.randn_like(ideal_update)
        mask = (torch.sign(vector_z) == vector_s).float()
        delta_z = vector_z * mask
        malicious_update = ideal_update + delta_z * lamda
        # malicious_update = malicious_update / torch.norm(malicious_update) * torch.norm(ideal_update) * 1.5
    
    else:
        raise ValueError(f"Unknown CAMP mode: {mode}")
    
    # 3. update the malicious clients' updates
    for i in range(num_attacks):
        updates[i] = malicious_update
    
    return updates, weights





def calculate_ideal_update(updates, agregation_method, num_attackers):
    """
    Calculate the ideal update based on the aggregation method.
    
    Parameters:
        updates (list of torch.Tensor): A list of tensors representing the updates from clients.
        agregation_method (str): The aggregation method to use.
        
    Returns:
        torch.Tensor: The aggregated ideal update.
    """
    if agregation_method == 'krum':
        selected_id = krum(updates, num_attackers)
        return updates[selected_id]
    elif agregation_method == 'median':
        return median(updates)
    elif agregation_method == 'trimmed':
        return trimmed(updates, ratio=0.2)
    elif agregation_method == 'avg':
        ideal_update = torch.zeros_like(updates[0])
        for update in updates:
            ideal_update += update / len(updates)
        return ideal_update
    else:
        raise ValueError(f"Unknown aggregation method: {agregation_method}")


def scale_attack(updates, num_attackers, scale):
    for i in range(num_attackers):
        updates[i] = updates[i] * scale
    return updates