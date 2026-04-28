import argparse
import logging

logger = logging.getLogger(__name__)


def args_parser():
    """Parse the given arguments in command-line environment

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser()

    # experiments related arguments
    parser.add_argument('--exp', type=int, default=0, help='tell which experiment is ongoing')
    parser.add_argument('--n', type=int, default=1, help='repeat the experiment for n times')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--sp', type=str, default='./', help='save_path')
    parser.add_argument('--sn', type=str, default='run', help='save_name')
    parser.add_argument('--log_file', type=str, default='training_log.json', help='log file name')

    # federated learning related arguments
    parser.add_argument('--r', type=int, default=100, help='communication rounds of training')
    parser.add_argument('--k', type=int, default=40, help='number of participants')
    parser.add_argument('--e', type=int, default=2, help='number of local epochs')
    parser.add_argument('--b', type=int, default=64, help='local batch size')
    parser.add_argument('--lr', type=float, default=0.02, help='learning rate')
    parser.add_argument('--ld', type=bool, default=False, help='use learning rate decay or not')
    parser.add_argument('--ldg', type=float, default=0.98, help='learning rate decay gamma')
    parser.add_argument('--nc', type=int, default=10, help='number of classes')
    parser.add_argument('--jr', type=float, default=1.0, help='joint ratio, the ratio of clients that join each round')
    parser.add_argument('--mu', type=float, default=0.5, help='hyper paramater, default 0.5')

    # metrics related arguments
    parser.add_argument('--metric', 
                        choices=["none", "cos_check"],
                        default='none', help='some metrics need to be obtained during learning')
    parser.add_argument('--agr', 
                        choices=['FedAvg', 'FedProx'],
                        default='FedAvg', help='FL algorithm used by server'
                        )

    # defense related arguments
    parser.add_argument('--filter',
                        choices=['avg', 'krum', 'median', 'trmean', 'multi-krum', 'sad', 'dpd', 'FLDetector', 'flame', 'maud-norm', 'maud-cosine'],
                        default='avg', help='filter rule used by server'
                        )

    # model & dataset related arguments
    parser.add_argument('--model', choices=['cnn', 'cnn2'], default='cnn2', help='model type')
    parser.add_argument('--dataset', choices=['FashionMNIST', 'Cifar10', 'MNIST'],
                        default='Cifar10', help='name of the dataset')
    # parser.add_argument('--iid', choices=[0, 1], default=1, type=int, help='IID dataset split(1 for IID, 0 for non-IID)')
    # parser.add_argument('--partition', choices=['dir', 'exdir'], default='exdir', type=str, help='non-IID partition method')
    # parser.add_argument('--p', type=float, default=0.1, help='degree of non-IID')

    # poisoning attacks related arguments
    parser.add_argument('--m', type=int, default=0, help='number of malicious clients/ attackers')
    parser.add_argument('--dp', choices=['none', 'lf', 'rlf'], default='none', help='data poisoning method')
    parser.add_argument('--ls', type=int, default=5, help='source label to flip')
    parser.add_argument('--lt', type=int, default=3, help='targeted label after flipping')
    parser.add_argument('--mp', choices=['none', 'min-max', 'LIE', 'random', 'sign_flip', 'CAMP', 'scale', 'MPAF'], default='none',
                        help='model poisoning method')
    parser.add_argument('--dpd_mode', choices=['none', 'low', 'high', 'auto'], default='auto',
                        help='clipping strategy')
    parser.add_argument('--noise_level', type=float, default=0, help='noise level for dpd')
    parser.add_argument('--s', type=float, default=1, help='scaling factor')
    parser.add_argument('--lamda', type=float, default=2, help='hyper parameter')
    parser.add_argument('--trmean_ratio', type=float, default=0.2, help='ratio for trimmed mean')
    parser.add_argument('--maud_window', type=int, default=10, help='MAUD accumulation window size N')
    parser.add_argument('--pk', choices=['none', 'agr', 'updates', 'all'], default='all', help='prior knowledge')
    parser.add_argument('--CAMP_mode', choices=['clipping', 'clipping_v5', 'clipping_v5_1', 'clipping_v6', 'clipping_v7', 'clipping_v8', 'perturbation', 'perturbation_v5', 'perturbation_v6'], default='clipping_v8', help='CAMP mode')

    # other arguments
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cpu", "cuda"])
    parser.add_argument("--device_id", type=str, default="0")

    args = parser.parse_args()

    return args
 # type: ignore


def parameters_info(args):
    """display parameter pertaining to exp settings
    """
    logger.info("*****************EXP-{0}*****************".format(args.exp))
    logger.info("Exp repeats: {0}".format(args.n))
    logger.info("Communication Rounds: {0}".format(args.r))
    logger.info("Clients: {0}\n".format(args.k))
    logger.info("Attack method: {0}".format(args.mp))
    logger.info("Filtering rule: {0}".format(args.filter))

    logger.info("Dataset: {0}".format(args.dataset))
    logger.info("Model: {0}".format(args.model))
    logger.info("Learning Rate: {0}".format(args.lr))
    logger.info("Batch Size: {0}".format(args.b))
    logger.info("Device: {0}".format(args.device))
    if args.device == "cuda":
        logger.info("Device id: {0}".format(args.device_id))
    # if args.iid == 1:
    #     logger.info("IID: True")
    # elif args.iid == 0:
    #     logger.info("Non-IID, Partition: {0}".format(args.partition))
    #     logger.info("Degree of non-IID: {0}\n".format(args.p))
    if args.filter == 'dpd':
        logger.info("DPD clipping strategy: {0}".format(args.dpd_mode))
        logger.info("DPD noise level: {0}".format(args.noise_level))

    logger.info("Adversaries: {0}".format(args.m))
    if args.dp == 'lf':
        logger.info("Targeted Label Flipping: {0}-->{1}".format(args.ls, args.lt))
    elif args.dp == 'rlf':
        logger.info("Untargeted label Flipping")
    if args.mp == "min-max":
        logger.info("Model Poisoning: Min-max attack")
    elif args.mp == "LIE":
        logger.info("Model Poisoning: A Little Is Enough attack")
    elif args.mp == "random":
        logger.info("Model Poisoning: Random Vector attack")
    elif args.mp == "sign_flip":
        logger.info("Model Poisoning: Sign Flip attack")
    elif args.mp == "CAMP":
        logger.info("Model Poisoning: CAMP attack, mode {0}".format(args.CAMP_mode))

    logger.info("*****************EXP-{0}*****************".format(args.exp))