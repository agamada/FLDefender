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


# L-BFGS for FLDetector
def lbfgs_torch(S_k_list, Y_k_list, v):
    curr_S_k = torch.stack(S_k_list)
    curr_S_k = curr_S_k.transpose(0, 1).cpu() #(10,xxxxxx)
    # print('------------------------')
    # print('curr_S_k.shape', curr_S_k.shape)
    curr_Y_k = torch.stack(Y_k_list)
    curr_Y_k = curr_Y_k.transpose(0, 1).cpu() #(10,xxxxxx)
    S_k_time_Y_k = curr_S_k.transpose(0, 1) @ curr_Y_k
    S_k_time_Y_k = S_k_time_Y_k.cpu()


    S_k_time_S_k = curr_S_k.transpose(0, 1) @ curr_S_k
    S_k_time_S_k = S_k_time_S_k.cpu()
    # print('S_k_time_S_k.shape', S_k_time_S_k.shape)
    R_k = np.triu(S_k_time_Y_k.numpy())
    L_k = S_k_time_Y_k - torch.from_numpy(R_k).cpu()
    sigma_k = Y_k_list[-1].view(-1,1).transpose(0, 1) @ S_k_list[-1].view(-1,1) / (S_k_list[-1].view(-1,1).transpose(0, 1) @ S_k_list[-1].view(-1,1))
    sigma_k=sigma_k.cpu()
    
    D_k_diag = S_k_time_Y_k.diagonal()
    upper_mat = torch.cat([sigma_k * S_k_time_S_k, L_k], dim=1)
    lower_mat = torch.cat([L_k.transpose(0, 1), -D_k_diag.diag()], dim=1)
    mat = torch.cat([upper_mat, lower_mat], dim=0)
    mat_inv = mat.inverse()
    # print('mat_inv.shape',mat_inv.shape)
    v = v.view(-1,1).cpu()

    approx_prod = sigma_k * v
    # print('approx_prod.shape',approx_prod.shape)
    # print('v.shape',v.shape)
    # print('sigma_k.shape',sigma_k.shape)
    # print('sigma_k',sigma_k)
    p_mat = torch.cat([curr_S_k.transpose(0, 1) @ (sigma_k * v), curr_Y_k.transpose(0, 1) @ v], dim=0)
    
    approx_prod -= torch.cat([sigma_k * curr_S_k, curr_Y_k], dim=1) @ mat_inv @ p_mat
    # print('approx_prod.shape',approx_prod.shape)
    # print('approx_prod.shape',approx_prod.shape)
    # print('approx_prod.shape.T',approx_prod.T.shape)

    return approx_prod.T


def fld_distance(old_update_list, local_update_list, hvp):
    pred_update = []
    distance = []
    for i in range(len(old_update_list)):
        pred_update.append((old_update_list[i] + hvp).view(-1))
        
    
    pred_update = torch.stack(pred_update)
    local_update_list = torch.stack(local_update_list)
    old_update_list = torch.stack(old_update_list)
    
    # distance = torch.norm((old_update_list - local_update_list), dim=1)
    # print('defense line219 distance(old_update_list - local_update_list):',distance)
    # auc1 = roc_auc_score(pred_update.numpy(), distance)
    # distance = torch.norm((pred_update - local_update_list), dim=1).numpy()
    # auc2 = roc_auc_score(pred_update.numpy(), distance)
    # print("Detection AUC: %0.4f; Detection AUC: %0.4f" % (auc1, auc2))
    
    # print('defence line 211 pred_update.shape:', pred_update.shape)
    distance = torch.norm((pred_update - local_update_list), dim=1)
    # print('defence line 211 distance.shape:', distance.shape)
    # distance = nn.functional.norm((pred_update - local_update_list), dim=0).numpy()
    distance = distance / torch.sum(distance)
    return distance


def detection(score, nobyz, k):
    estimator = KMeans(n_clusters=2)
    estimator.fit(score.reshape(-1, 1))
    label_pred = estimator.labels_
    
    if np.mean(score[label_pred==0])<np.mean(score[label_pred==1]):
        #0 is the label of malicious clients
        label_pred = 1 - label_pred
    real_label=np.ones(k)
    real_label[:nobyz]=0
    acc=len(label_pred[label_pred==real_label])/k
    recall=1-np.sum(label_pred[:nobyz])/nobyz
    fpr=1-np.sum(label_pred[nobyz:])/(k-nobyz)
    fnr=np.sum(label_pred[:nobyz])/nobyz
    # print("acc %0.4f; recall %0.4f; fpr %0.4f; fnr %0.4f;" % (acc, recall, fpr, fnr))
    # print(silhouette_score(score.reshape(-1, 1), label_pred))
    # print('defence.py line233 label_pred (0 = malicious pred)', label_pred)
    return label_pred


def detection1(score):
    nrefs = 10
    ks = range(1, 8)
    gaps = np.zeros(len(ks))
    gapDiff = np.zeros(len(ks) - 1)
    sdk = np.zeros(len(ks))
    min = np.min(score)
    max = np.max(score)
    score = (score - min)/(max-min)
    for i, k in enumerate(ks):
        estimator = KMeans(n_clusters=k)
        estimator.fit(score.reshape(-1, 1))
        label_pred = estimator.labels_
        center = estimator.cluster_centers_
        Wk = np.sum([np.square(score[m]-center[label_pred[m]]) for m in range(len(score))])
        WkRef = np.zeros(nrefs)
        for j in range(nrefs):
            rand = np.random.uniform(0, 1, len(score))
            estimator = KMeans(n_clusters=k, n_init=10)
            estimator.fit(rand.reshape(-1, 1))
            label_pred = estimator.labels_
            center = estimator.cluster_centers_
            WkRef[j] = np.sum([np.square(rand[m]-center[label_pred[m]]) for m in range(len(rand))])
        gaps[i] = np.log(np.mean(WkRef)) - np.log(Wk)
        sdk[i] = np.sqrt((1.0 + nrefs) / nrefs) * np.std(np.log(WkRef))

        if i > 0:
            gapDiff[i - 1] = gaps[i - 1] - gaps[i] + sdk[i]
    # print('defense line278 gapDiff:', gapDiff)
    select_k = 2  # default detect attacks
    for i in range(len(gapDiff)):
        if gapDiff[i] >= 0:
            select_k = i+1
            break
    if select_k == 1:
        print('No attack detected!')
        return 0
    else:
        print('Attack Detected!')
        return 1

def FLDetector(uploaded_updates, model_record, update_record, global_model, N=5):
    global_params = torch.cat([param.data.view(-1) for param in global_model.parameters()])
    pass