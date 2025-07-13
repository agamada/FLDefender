import copy
import torch
import random
import os
import time
import warnings
import numpy as np
import logging

from src.model import *
from src import spliter
from src import parser
from src import roles

seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

if __name__ == '__main__':
    args = parser.args_parser()

    # set result file name and save path
    # exp1:find good lr
    if args.exp == 1:
        args.sp = f'./exp/exp1/{args.dataset}'
        args.sn = f'{args.mp}_{args.filter}_{args.lr}'
    else:
        args.sp = f'./exp/exp0/{args.dataset}'
        args.sn = f'{args.mp}_{args.filter}_{args.m}'
    
    if not os.path.exists(args.sp):
        os.makedirs(args.sp)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(args.sp, args.sn) + '.log', mode="w")
        ]
    )
    logger = logging.getLogger(__name__)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    parser.parameters_info(args)

    # # split dataset
    # if args.iid:
    #     data_dict = spliter.split_iid(args.dataset, args.k)
    # else:
    #     if args.partition == "dir":
    #         data_dict = spliter.split_non_iid_dir(args.dataset, args.k, args.nc, args.p)
    #     elif args.partition == "exdir":
    #         data_dict = spliter.split_non_iid_exdir(args.dataset, args.k, args.nc, args.p)
    #     else:
    #         raise NotImplementedError
    
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
        logger.info(f"Running time: {n}th ==========================")
        start = time.time()
        server = roles.Server(model, n, args)
        server.load_data()
        server.train()
    
