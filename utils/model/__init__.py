from .cv_model import *
from .nlp_model import *

Name2Function = {
    'resnet20_cifar': resnet20_cifar,
    'resnet32_cifar': resnet32_cifar,
    'resnet44_cifar': resnet44_cifar,
    'resnet56_cifar': resnet56_cifar,
    'resnet110_cifar': resnet110_cifar,
    'resnet1202_cifar': resnet1202_cifar,
}

def get_model(args):
    if args.model in Name2Function:
        return Name2Function[args.model](args.n_class)
    elif args.model == 'simplecnn':
        return SimpleCNN(channel=3, input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=args.n_class)
    elif args.model == 'simplecnn-mnist':
        return SimpleCNN(channel=1, input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=args.n_class)
    elif args.model == 'lstm_shakes':
        return RNN_Shakespeare()
    elif args.model == 'cnn_femnist':
        return CNN_FEMNIST()