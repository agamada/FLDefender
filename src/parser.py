import argparse


def args_parser():
    """Parse the given arguments in command-line environment

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser()

    # experiments related arguments
    parser.add_argument('--exp', type=int, default=0, help='tell which experiment is ongoing')
    parser.add_argument('--n', type=int, default=1, help='repeat the experiment for n times')

    # federated learning related arguments
    parser.add_argument('--r', type=int, default=5, help='communication rounds of training')
    parser.add_argument('--k', type=int, default=20, help='number of participants')
    parser.add_argument('--e', type=int, default=5, help='number of local epochs')
    parser.add_argument('--b', type=int, default=64, help='local batch size')
    parser.add_argument('--lr', type=float, default=0.04, help='learning rate')
    parser.add_argument('--ld', type=bool, default=False, help='use learning rate decay or not')
    parser.add_argument('--ldg', type=float, default=0.995, help='learning rate decay gamma')
    parser.add_argument('--nc', type=int, default=10, help='number of classes')

    # metrics related arguments
    parser.add_argument('--metric', choices=["none", "cos_check"],
                        default='none', help='some metrics need to be obtained during learning'),

    # defense related arguments
    parser.add_argument('--filter',
                        choices=['avg', 'krum', 'median', 'trmean', 'test_cos', 'baseline', 'dspo', 'dspo_plus', 'dspo_pro'],
                        default='avg', help='filter rule used by server'
                        )

    # model & dataset related arguments
    parser.add_argument('--model', choices=['cnn', 'cnn2'], default='cnn2', help='model type')
    parser.add_argument('--dataset', choices=['FashionMNIST', 'Cifar10'],
                        default='Cifar10', help='name of the dataset')
    parser.add_argument('--iid', choices=[0, 1], default=1, type=int, help='IID dataset split(1 for IID, 0 for non-IID)')
    parser.add_argument('--partition', choices=['dir', 'exdir'], default='exdir', type=str, help='non-IID partition method')
    parser.add_argument('--p', type=float, default=0.1, help='degree of non-IID')

    # poisoning attacks related arguments
    parser.add_argument('--m', type=int, default=0, help='number of malicious clients/ attackers')
    parser.add_argument('--dp', choices=['none', 'lf', 'rlf'], default='none', help='data poisoning method')
    parser.add_argument('--ls', type=int, default=5, help='source label to flip')
    parser.add_argument('--lt', type=int, default=3, help='targeted label after flipping')
    parser.add_argument('--mp', choices=['none', 'scale', 'neg', 'rand', 'min_max'], default='none',
                        help='model poisoning method')
    parser.add_argument('--s', type=float, default=1, help='scaling factor')
    parser.add_argument('--lamda', type=float, default=0.2, help='hyper parameter')

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
    print("\n*****************EXP-{0}*****************".format(args.exp))
    print("Exp repeats: {0}".format(args.n))
    print("Communication Rounds: {0}".format(args.r))
    print("Clients: {0}\n".format(args.k))

    print("Filtering rule: {0}".format(args.filter))

    print("Dataset: {0}".format(args.dataset))
    print("Model: {0}".format(args.model))
    print("Learning Rate: {0}".format(args.lr))
    print("Batch Size: {0}".format(args.b))
    print("Device: {0}".format(args.device))
    if args.device == "cuda":
        print("Device id: {0}".format(args.device_id))
    if args.iid == 1:
        print("IID: True\n")
    elif args.iid == 0:
        print("Non-IID, Partition: {0}".format(args.partition))
        print("Degree of non-IID: {0}\n".format(args.p))

    print("Adversaries: {0}".format(args.m))
    if args.dp == 'lf':
        print("Targeted Label Flipping: {0}-->{1}".format(args.ls, args.lt))
    elif args.dp == 'rlf':
        print("Untargeted label Flipping")
    if args.mp == "scale":
        print("Model Poisoning: Scaling up")
        print("Scaling factor: {0}".format(args.s))
    elif args.mp == "neg":
        print("Model Poisoning: Negative contribution")
    elif args.mp == "rand":
        print("Model Poisoning: Random contribution")
    elif args.mp == "min_max":
        print("Model Poisoning: Min-max attack")

    print("*****************EXP-{0}*****************\n".format(args.exp))