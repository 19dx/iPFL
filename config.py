import argparse
import os

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default="1")
    parser.add_argument('--model', type=str, default='simplecnn', help='neural network used in training')
    parser.add_argument('--init_seed', type=int, default=0, help="Random seed")
    parser.add_argument('--load_path', type=str, default=None)
    # ====== FL setting ======
    parser.add_argument('--alg', type=str, default='fedavg', help='name of federated learning algorithm')
    parser.add_argument('--num_local_iterations', type=int, default=400, help='number of local iterations')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.1)')
    parser.add_argument('--epochs', type=int, default=10, help='number of local epochs')
    parser.add_argument('--n_parties', type=int, default=6, help='number of workers in a distributed cluster')
    parser.add_argument('--comm_round', type=int, default=50, help='number of maximum communication roun')
    parser.add_argument('--reg', type=float, default=1e-5, help="L2 regularization strength")
    parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')
    parser.add_argument('--optimizer', type=str, default='sgd', help='the optimizer')
    # parser.add_argument('--sample_fraction', type=float, default=1.0, help='how many clients are sampled in each round')
    parser.add_argument('--difference_metric', type=str, default='cos', help='How to measure the model difference')

    # ====== incentive ======
    parser.add_argument('--K', type=float, default=5e5, help="Hyper-parameter in the utility function")
    parser.add_argument('--C', type=float, default=0.0, help="cost")
    # attack
    parser.add_argument('--attack_type', type=str, default="inv_grad")
    parser.add_argument('--attack_ratio', type=float, default=0.0)
    # freeriding
    parser.add_argument('--freeride_type', type=str, default=None, help="the freeride type")
    parser.add_argument('--freerider_ratio', type=float, default=0.0, help='ratio to sample for freeriders')
    # liar
    parser.add_argument('--liar_ratio', type=float, default=0.0)
    parser.add_argument('--liar_type', type=str, default="size")
    parser.add_argument('--liar_exaggerate_ratio', type=float, default=2.0)
    # dropout
    parser.add_argument('--dropout', action='store_true', help="whether to choose dropout if negative utility occurs")
    parser.add_argument('--dropout_type', type=str, default='IR-strict', help="the reason for dropping out")
    parser.add_argument('--dropout_p', type=float, required=False, default=0.5, help="Dropout probability. Default=0.0")
    # Diverse agent type
    parser.add_argument('--diverse', action='store_true', help="whether to use diverse agent type")

    # ====== Dataset ======
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset used for training')
    parser.add_argument('--datadir', type=str, required=False, default="./data/", help="Data directory")
    parser.add_argument('--partition', type=str, default='noniid', help='the data partitioning strategy')
    parser.add_argument('--beta', type=float, default=0.5, help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--skew_class', type=int, default = 5, help='The parameter for the noniid-skew for data partitioning')
    # For LEAF dataset
    parser.add_argument('--leaf_sample_top', type=int, default=1, help='whether to sample top clients from femnist')
    parser.add_argument('--leaf_train_num', type=int, default=20, help='how many clients from femnist are sampled')
    parser.add_argument('--leaf_test_num', type=int, default=10, help='number of testing clients from femnist')
    
    # ====== iPFL Hyper-parameters ======
    parser.add_argument('--ipfl_lam', type=float, default=1, help="Hyper-parameter in the updating")
    parser.add_argument('--ipfl_eta', type=float, default=5, help="learning rate for proxy center")
    # ====== Other Baselines Hyper-parameters ======
    # FedProx / MOON
    parser.add_argument('--mu', type=float, default=0.01)
    # Ditto
    parser.add_argument('--ditto_lamda', type=float, default=1, help="lambda in the objective")
    # FedAMP
    parser.add_argument('--fedamp_lam1', type=float, default=0.01, help='hyper-param used in local training for fedamp')
    # pFedGraph
    parser.add_argument('--pfedgraph_alpha', type=float, default=0.8, help='hyper-param used in local training for pfedgraph')
    parser.add_argument('--pfedgraph_lam', type=float, default=0.01, help="Hyper-parameter in the objective")
    # CFL
    parser.add_argument('--cfl_eps1', type=float, default=2.0, help='The hyper-parameter to control clustering')
    parser.add_argument('--cfl_eps2', type=float, default=2.5, help='The hyper-parameter to control clustering')
    # FedFomo
    parser.add_argument('--fedfomo_M', type=int, default=6, help=" the  maximum number of model server send to clients")

    args = parser.parse_args()
    
    dataset_info = {
        'fashionmnist': {'n_class': 10, 'n_channel': 1, 'img_size': '28'},
        'femnist': {'n_class': 62, 'n_channel': 1, 'img_size': '28'},
        'cifar10': {'n_class': 10, 'n_channel': 3, 'img_size': '32'},
        'shakespeare': {'n_class': 80, 'n_channel': 1, 'img_size': '1'},
        'sent140': {'n_class': 2, 'n_channel': 1, 'img_size': '1'},
        'pacs':{'n_class':7,'n_channel':3,'img_size':'227'},
    }

    args.n_class = dataset_info[args.dataset]['n_class']
    args.n_channel = dataset_info[args.dataset]['n_channel']
    args.img_size = dataset_info[args.dataset]['img_size']
    
    if args.dataset == 'femnist':
        args.n_parties = args.leaf_train_num
        args.datadir = args.datadir + 'femnist/data/all_data'
    elif args.dataset == 'shakespeare':
        args.n_parties = args.leaf_train_num
        args.datadir = args.datadir + 'shakespeare/data/all_data'
    elif args.dataset in ('cifar10', 'cifar100'):
        args.datadir = args.datadir + 'cifar'


    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    print_parsed_args(args)

    return args

def print_parsed_args(args):
    print(f"{'='*20} Parsed Arguments {'='*20}")  
    max_key_length = max(len(key) for key in vars(args).keys())  
    for key, value in vars(args).items():  
        print(f"{key.ljust(max_key_length)}: {value}")  
  
    print('=' * 40)