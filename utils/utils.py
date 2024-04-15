import torch
import numpy as np
import random

def set_seed(seed):  
    torch.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)  
    random.seed(seed)  
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False 

def set_model_zero(model):
    model_state = model.state_dict()
    for key in model_state.keys():
        model_state[key] = torch.zeros_like(model_state[key])
    model.load_state_dict(model_state)
    return model

# ========= For Instruction-tuning tasks =========
import math

def cosine_learning_rate(current_round, total_rounds, initial_lr=0.001, min_lr=0):
    """
    Compute the learning rate based on a cosine schedule.

    :param current_round: The current training round (0-indexed).
    :param total_rounds: The total number of training rounds.
    :param initial_lr: The initial learning rate.
    :param min_lr: The minimum learning rate.
    :return: The computed learning rate for the current round.
    """
    # Compute the cosine learning rate
    cosine_lr = min_lr + 0.5 * (initial_lr - min_lr) * (1 + math.cos(math.pi * current_round / total_rounds))
    return cosine_lr

def split_dataset(fed_args, script_args, dataset, prefix='Training'):
    local_datasets = []
    if fed_args.split_strategy == "iid":
        dataset = dataset.shuffle(seed=script_args.seed)        # Shuffle the dataset
        for i in range(fed_args.n_parties):
            local_datasets.append(dataset.shard(fed_args.n_parties, i))
    elif fed_args.split_strategy=="cluster":
        num_cluster = len(dataset)
        avg_num_client_per_cluster = fed_args.n_parties // num_cluster
        num_client_per_cluster = [avg_num_client_per_cluster] * num_cluster
        if sum(num_client_per_cluster) < fed_args.n_parties:
            num_client_per_cluster[-1] += fed_args.n_parties - sum(num_client_per_cluster)
        if script_args.dataset_name == 'CoFinance' and fed_args.n_parties == 8:
            num_client_per_cluster = [5,3]
        for i in range(num_cluster):
            for j in range(num_client_per_cluster[i]):
                local_datasets.append(dataset[i].shard(num_client_per_cluster[i], j))
        print(f">> Split the {prefix} dataset into {num_client_per_cluster}.")
    return local_datasets

def get_dataset_this_round(dataset, round, fed_args, script_args):
    num2sample = script_args.batch_size * script_args.gradient_accumulation_steps * script_args.max_steps
    num2sample = min(num2sample, len(dataset))
    random.seed(round)
    random_idx = random.sample(range(0, len(dataset)), num2sample)
    dataset_this_round = dataset.select(random_idx)
    return dataset_this_round