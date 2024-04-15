import torch
import numpy as np

def manipulate_gradient(args, nets_this_round, malicious_clients):
    index_clientid = list(nets_this_round.keys())
    (freeriders, attackers) = malicious_clients
    attackers_this_round = [client_id for client_id in attackers if client_id in index_clientid]
    freeriders_this_round = [client_id for client_id in freeriders if client_id in index_clientid]
    if len(attackers_this_round) > 0:
        print(f'Manipulating Client (Attacker) {attackers_this_round} with {args.attack_type}')
        for attacker in attackers_this_round:
            manipulate_attacker_model(args.attack_type, nets_this_round[attacker])
    if len(freeriders_this_round) > 0:
        print(f'Manipulating Client (Freerider) {freeriders_this_round} with {args.freeride_type}')
        for freerider in freeriders_this_round:
            manipulate_freerider_model(args.freeride_type, nets_this_round[freerider])

def manipulate_attacker_model(attack_type, net):
    if attack_type == 'shuffle':     # shuffle model parameters
        flat_params = get_flat_params_from(net)
        shuffled_flat_params = flat_params[torch.randperm(len(flat_params))]
        set_flat_params_to(net, shuffled_flat_params)
    elif attack_type == 'sign_flip':
        flat_params = get_flat_params_from(net)
        flat_params = -flat_params
        set_flat_params_to(net, flat_params)

def manipulate_freerider_model(freeride_type, net):
    if freeride_type == 'same_value': # free-riding
        flat_params = get_flat_params_from(net)
        flat_params = torch.ones_like(flat_params)
        set_flat_params_to(net, flat_params)
    elif freeride_type == 'gauss':
        flat_params = get_flat_params_from(net)
        flat_params = torch.normal(0, 1, size=flat_params.shape)
        set_flat_params_to(net, flat_params)

def get_flat_params_from(model):
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))

    flat_params = torch.cat(params)
    return flat_params

def set_flat_params_to(model, flat_params):
    prev_ind = 0
    for param in model.parameters():
        flat_size = int(np.prod(list(param.size())))
        param.data.copy_(
            flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size