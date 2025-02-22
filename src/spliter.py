import torch
import random
import numpy as np
from torchvision import transforms, datasets
from torch.utils import data


data_path = "./data/"  # where data sets store, the path is from the perspective of root


def get_dataset(dataset_name):
    """Return the train_set, the test_set of the given dataset name
    """
    target_set, trans = None, None

    if dataset_name == "FashionMNIST":
        target_set = datasets.FashionMNIST
        trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    elif dataset_name == "Cifar10":
        target_set = datasets.CIFAR10
        trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # The first time to run the code, please set download=True to obtain data online
    # Here we set download=False to ignore the unpleasant prompt from CIFAR-10:
    # "Files already downloaded and verified"
    train_set = target_set(root=data_path, download=False, train=True, transform=trans)
    test_set = target_set(root=data_path, download=False, train=False, transform=trans)

    # def _preload_dataset(dataset):
    #     data_tensor = torch.stack([x for x, _ in dataset])
    #     target_tensor = torch.tensor([y for _, y in dataset])
    #     return data.TensorDataset(data_tensor, target_tensor)

    # train_set = _preload_dataset(train_set)
    # test_set = _preload_dataset(test_set)
    return train_set, test_set


def split_iid(dataset_name, num_clients):
    """Split the dataset evenly according to the number of users
    Args:
        dataset_name: the name of the dataset to be used
        num_clients: the number of clients participated in FL

    Returns: (dict)
        The dict of dataset that is split
        {
            'train': (list) the split training subsets
            'test': (Dataset) testing dataset
        }
    """
    train_set, test_set = get_dataset(dataset_name)

    num_all_samples = len(train_set)
    idx_list = list(range(num_all_samples))
    num_sps = num_all_samples // num_clients  # number of samples for each client
    random.shuffle(idx_list)    # disorder the indexes

    train_subsets = [data.Subset(train_set, idx_list[i: i+num_sps]) for i in range(0, num_all_samples, num_sps)]

    return {'train': train_subsets, 'test': test_set}


def split_non_iid_exdir(dataset_name, num_clients, num_classes, p):
    """Split the dataset in a non-IID form
    1. Evenly divide the clients into L groups, where L denotes the classes in the dataset
    2. Assign the training sample labeled by i to i-th group with probability p, to other groups with (1-p)/(L-1)
    3. The sample is finally obtained by one of the clients in the assigned group randomly

    the degree of non-IID grows as the value of p increases

    Two extreme cases:
    When p = 1/L, each training sample is assigned to the groups with equal probability(similar to IID setting);
    when p = 1, each group' samples are with only one kind of label(the highest degree of non-IID)

    Args:
        dataset_name: (str), the dataset needs to be split
        num_clients: (int), the number of clients participated in FL
        p: (float), the degree of non-IID

    Returns: (dict)
        The dict of dataset that is split
        {
            'train': (list) the split training subsets
            'test': (Dataset) testing dataset
        }
    """
    train_set, test_set = get_dataset(dataset_name)

    L = num_classes  # groups / classes of the dataset
    g = num_clients // L  # clients in each group
    subset_idx_list = [[] for a in range(num_clients)]  # the index list of each subset

    for idx, (X, y) in enumerate(train_set):
        if random.random() <= p:
            group_idx = y
        else:
            other_idx = list(range(L))
            other_idx.remove(y)
            group_idx = random.choice(other_idx)
        selected_client = random.choice(range(group_idx * g, group_idx * g + g))

        subset_idx_list[selected_client].append(idx)  # the index of the current sample

    train_subsets = [data.Subset(train_set, idx_list) for idx_list in subset_idx_list]

    return {'train': train_subsets, 'test': test_set}


def dirichlet_distribution(alpha, num_clients, num_classes):
    """Generate Dirichlet distribution for each class.
    Args:
        alpha: (float) Dirichlet concentration parameter, controls the degree of imbalance.
        num_clients: (int) number of clients.
        num_classes: (int) number of classes in the dataset.
    Returns:
        A list of arrays where each array represents the distribution of one class
        across all clients.
    """
    # Generate Dirichlet distribution for each class
    proportions = np.zeros((num_classes, num_clients))
    for i in range(num_classes):
        proportions[i, :] = np.random.dirichlet([alpha] * num_clients)
    
    return proportions


def split_non_iid_dir(dataset_name, num_clients, num_classes, alpha=0.5):
    """Split the dataset in a non-IID form using Dirichlet distribution.
    Args:
        dataset_name: (str) the dataset name, e.g., "CIFAR10".
        num_clients: (int) the number of clients.
        alpha: (float) Dirichlet concentration parameter, controls the degree of non-IID.
    
    Returns:
        dict: {'train': list of Subset objects, 'test': Dataset}
    """
    train_set, test_set = get_dataset(dataset_name)
    subset_idx_list = [[] for _ in range(num_clients)]  # To store indices for each client

    proportions = dirichlet_distribution(alpha, num_clients, num_classes)

    for idx, (X, y) in enumerate(train_set):
        class_idx = y  
        client_distribution = proportions[class_idx]

        selected_client = np.random.choice(num_clients, p=client_distribution)

        subset_idx_list[selected_client].append(idx)

    # Create Subset objects for each client
    train_subsets = [data.Subset(train_set, idx_list) for idx_list in subset_idx_list]

    return {'train': train_subsets, 'test': test_set}


if __name__ == '__main__':
    train_set, test_set = get_dataset("Cifar10")