import torch
import copy
import math
import logging
from scipy.stats import norm

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


def multi_Krum(uploaded_updates, num_attackers):
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