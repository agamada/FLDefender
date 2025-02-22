import copy
import torch
import argparse
import os
import time
import warnings
import numpy as np
import torchvision
import logging

from src.model import *
from src import spliter
from src import parser
from src import roles


torch.manual_seed(0)

if __name__ == '__main__':
    args = parser.args_parser()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"

    parser.parameters_info(args)

    # split dataset
    if args.iid:
        data_dict = spliter.split_iid(args.dataset, args.k)
    else:
        if args.partition == "dir":
            data_dict = spliter.split_non_iid_dir(args.dataset, args.k, args.nc, args.p)
        elif args.partition == "exdir":
            data_dict = spliter.split_non_iid_exdir(args.dataset, args.k, args.nc, args.p)
        else:
            raise NotImplementedError
    
    # prepare model
    if args.model == "cnn":
        if args.dataset == "FashionMNIST":
            model = CNN(in_features=1, num_classes=args.nc, dim=1024).to(args.device)
        elif args.dataset == "Cifar10":
            model = CNN(in_features=3, num_classes=args.nc, dim=1600).to(args.device)
    elif args.model == "cnn2":
        if args.dataset == "FashionMNIST":
            model = CNN2(in_features=1, num_classes=args.nc, dim=1024).to(args.device)
        elif args.dataset == "Cifar10":
            model = CNN2(in_features=3, num_classes=args.nc, dim=2048).to(args.device)
    else:
        raise NotImplementedError
    
    for n in range(args.n):
        print(f"\n============= Running time: {n}th =============")
        start = time.time()
        server = roles.Server(model, args)
        server.load_data(data_dict)
        server.train()
    
