import torch
import copy
import math
import time
import logging
from scipy.stats import norm
from sklearn.cluster import KMeans
import numpy as np

logger = logging.getLogger(__name__)

def krum(uploaded_updates, num_attackers):
    num_clients = len(uploaded_updates)
    assert num_clients > 2 * num_attackers + 2

    distances = torch.zeros((num_clients, num_clients))

    for i in range(num_clients):
        for j in range(i + 1, num_clients):
            distances[i, j] = torch.norm(uploaded_updates[i] - uploaded_updates[j])
            distances[j, i] = distances[i, j]
    
    scores = []
    for i in range(num_clients):
        sorted_distances, _ = torch.sort(distances[i])
        score = torch.sum(sorted_distances[1: num_clients - num_attackers - 1])
        scores.append(score)
    
    krum_index = torch.argmin(torch.tensor(scores)).item()
    return krum_index


def median(uploaded_updates):
    return torch.median(torch.stack(uploaded_updates), dim=0).values


def trimmed(uploaded_updates, ratio):
    assert 0 < ratio < 0.5

    n = len(uploaded_updates)

    sorted_updates, _ = torch.sort(torch.stack(uploaded_updates), dim=0)
    trim_count = int(n * ratio)
    trim_mean = torch.mean(sorted_updates[trim_count:-trim_count], dim=0)

    return trim_mean


def multi_krum(uploaded_updates, num_attackers):
    num_clients = len(uploaded_updates)
    assert num_clients > 2 * num_attackers + 2

    distances = torch.zeros((num_clients, num_clients))

    for i in range(num_clients):
        for j in range(i + 1, num_clients):
            distances[i, j] = torch.norm(uploaded_updates[i] - uploaded_updates[j])
            distances[j, i] = distances[i, j]
    
    scores = []
    for i in range(num_clients):
        sorted_distances, _ = torch.sort(distances[i])
        score = torch.sum(sorted_distances[1: num_clients - num_attackers - 1])  # 不考虑自己与自己
        scores.append(score)

    num_selected = num_clients - num_attackers - 1
    sorted_scores, sorted_indices = torch.sort(torch.tensor(scores))
    sorted_indices = sorted_indices[:num_selected]

    return sorted_indices.tolist()


def selective_mean(uploaded_updates, args):
    # 1. norm clipping
    n = len(uploaded_updates)
    norm_list = [torch.norm(update).item() for update in uploaded_updates]
    sorted_norm_list = sorted(norm_list)

    if n % 2:
        C_t = sorted_norm_list[n // 2]
    else:
        C_t = (sorted_norm_list[n // 2 - 1] + sorted_norm_list[n // 2]) / 2
    
    for i in range(n):
        if norm_list[i] > C_t:
            uploaded_updates[i] = uploaded_updates[i] * (C_t / norm_list[i])

    # 2. hamming distance & flag-value filling
    d = len(uploaded_updates[0])

    sign_w = 0
    for update in uploaded_updates:
        sign_w += torch.sign(update)
    sign_w = torch.sign(sign_w)

    # flag-value filling
    k_list = []
    for i, update in enumerate(uploaded_updates):
        mask = (torch.sign(update) != sign_w).float()

        k = torch.sum(mask).item() / d
        k_list.append(k)
        k = min(k, 0.2)  
        bool_mask = torch.rand_like(update, dtype=torch.float) < k

        uploaded_updates[i] = torch.where(bool_mask, torch.tensor(float('nan')), update)
    
    n = len(k_list)
    k_array = np.array(k_list).reshape(-1, 1)
    kmeans = KMeans(n_clusters=2, n_init=10, random_state=0).fit(k_array)
    labels = kmeans.labels_
    counts = np.bincount(labels)
    cluster_means = [k_array[labels == i].mean() for i in range(2)]
    # 判断哪个簇是 benign，两簇相差不远选k小的簇，否则选元素多的簇
    if abs(cluster_means[0] - cluster_means[1]) < 0.1:
        benign_label = np.argmin(cluster_means)  # k 小的簇是 benign
    else:
        if counts[0] > counts[1]:
            benign_label = 0
        elif counts[1] > counts[0]:
            benign_label = 1
        else:
            benign_label = np.argmin(cluster_means)  # k 小的簇是 benign

    attacker_idx = [i for i in range(n) if labels[i] != benign_label]
    logger.info(f"suspected attackers: {attacker_idx}")
    logger.info(f"clients' k values: {k_list}")
    logger.info(f"cluster means: {cluster_means}")
    logger.info(f"benign cluster label: {benign_label}")

    # 3. selected parameters mean
    # Filter out suspected attackers
    filtered_updates = [u for i, u in enumerate(uploaded_updates) if i not in attacker_idx]
    stacked_updates = torch.stack(filtered_updates)
    # stacked_updates = torch.stack(uploaded_updates)
    
    # print(d)
    # for i in range(d):
    #     # print("test:", i, "/", d)
    #     column = stacked_updates.index_select(1, torch.tensor(i).to(args.device)).flatten()
    #     mask = ~ torch.isnan(column)    # get valid values
    #     if torch.sum(mask) > 0:
    #         column_mean = torch.mean(column[mask])
    #         aggregated_update[i] = column_mean
    column_means = torch.nanmean(stacked_updates, dim=0)
    column_means[torch.isnan(column_means)] = 0
    
    return column_means
        

def dpd(uploaded_updates, clip_strategy, noise_level):

    # 1. calculate norm
    n = len(uploaded_updates)
    norm_list = [torch.norm(update).item() for update in uploaded_updates]

    # 2. get clipping bound (C_t), which is decided by clip_strategy
    C_t = None
    if clip_strategy == 'none':
        C_t = max(norm_list)
    elif clip_strategy == 'low':
        C_t = 0.4
    elif clip_strategy == 'high':
        C_t = 4
    elif clip_strategy == 'auto':
        sorted_norm_list = sorted(norm_list)
        if n % 2:
            C_t = sorted_norm_list[n // 2]
        else:
            C_t = (sorted_norm_list[n // 2 - 1] + sorted_norm_list[n // 2]) / 2
    else:
        raise ValueError("Invalid clip_strategy")
    logger.info(f"DPD clipping bound C_t: {C_t}")
    # logger.info(f"dimension of updates: {len(uploaded_updates[0])}")
    
    # 3. norm clipping
    for i in range(n):
        if norm_list[i] > C_t:
            uploaded_updates[i] = uploaded_updates[i] * (C_t / norm_list[i])

    # 4. noise adding
    for update in uploaded_updates:
        noise = torch.randn_like(update) * noise_level * C_t
        update += noise

    logger.info(f"DPD noise norm:{torch.norm(noise).item()}")
    # print("noise norm" , torch.norm(noise))
    # print("total noise norm" , torch.norm(total_noise))

    return
            
