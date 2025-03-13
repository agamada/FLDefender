import torch
import copy
import math
import logging
from scipy.stats import norm

logger = logging.getLogger(__name__)

def min_max_attack(updates, weights, num_attackers):
    # 1. get the update tensor
    all_updates = torch.stack(updates)
    
    # 2. calculating the max l2-norm of the updates 
    distances = torch.norm(all_updates[:,None,:] - all_updates[None,:,:], dim=2) ** 2
    max_distance = torch.max(distances)

    # 3. constructing the malicious update via binary searching, V(m) = V(mean) - s*V(p)
    lamda = torch.Tensor([10.0]).float().cuda()
    threshold_diff = 1e-5
    step = lamda
    lamda_succ = 0
    try_count = 0

    mean_update = torch.mean(all_updates, dim=0)
    deviation = mean_update / torch.norm(mean_update) # unit vector, dir opp to good dir

    while torch.abs(lamda_succ - lamda) > threshold_diff:  
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