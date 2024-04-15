import torch
import numpy as np
import math
import copy
import cvxpy as cp
from scipy.optimize import fsolve


def update_graph_ipfl(args, graph_matrix, nets_this_round, gain_list, data_num_list, cost_list, model_difference_matrix):
    index_clientid = list(nets_this_round.keys())
    for id in index_clientid:
        graph_matrix[id] = torch.zeros(args.n_parties)
        if gain_list[id] == 0:
            continue
        chosen_clients = choose_clients_direct(gain_list[id], id, nets_this_round, data_num_list, cost_list, model_difference_matrix[id]*args.ipfl_lam)
        for j in chosen_clients:
            graph_matrix[id,j] = data_num_list[j]/data_num_list[id]
    return graph_matrix

def choose_clients_direct(K, self_id, nets_this_round, data_num_list, cost_list, model_div_list):
    index_clientid = list(nets_this_round.keys())
    M = len(index_clientid)
    NTH = []
    for j in range(M):
        def f(x):
            if x < data_num_list[j]:
                return 0
            else:
                return math.sqrt(K/(x-data_num_list[j]))-math.sqrt(K/x)-cost_list[j]-model_div_list[j]*data_num_list[j]/data_num_list[self_id]
        x =  fsolve(f,3*data_num_list[j])
        NTH.append(x)
    print('Thereshold for client',self_id,':',NTH)
    n = data_num_list[self_id]
    sorted_clients = sorted(index_clientid, key=lambda i:NTH[i], reverse=True)
    chosen_clients = []
    for j in range(M):
        if sorted_clients[j] == self_id:
            continue
        if n+data_num_list[sorted_clients[j]] < NTH[sorted_clients[j]]:
            chosen_clients.append(sorted_clients[j])
            n += data_num_list[sorted_clients[j]]
        else:
            break
    print('Chosen for client',self_id,':',chosen_clients)
    print('Data Access:',n)
    return chosen_clients
 
def weight_flatten_all(model):
    params = []
    for param in model:
        params.append(param.data.reshape(-1))
    params = torch.cat(params)
    return params

def weight_flatten_all_llm(model):
    params = []
    for k in model:
        params.append(model[k].reshape(-1))
    params = torch.cat(params)
    return params

def aggregation_ipfl_cos(args, graph_matrix, nets_this_round, global_model):
    global_vec = weight_flatten_all(global_model.parameters())
    prox_vectors = {}
    for i in nets_this_round.keys():
        aggregation_weight_vector = graph_matrix[i]
        prox_vectors[i] = weight_flatten_all(nets_this_round[i].parameters())
        prox_vectors[i].requires_grad_(True)
        loss = 0
        for j in nets_this_round.keys():
            model_j = weight_flatten_all(nets_this_round[j].parameters())-global_vec
            loss -= aggregation_weight_vector[j]*torch.dot(prox_vectors[i]-global_vec,model_j)/torch.norm(prox_vectors[i]-global_vec)/torch.norm(model_j)
        loss.backward()
        with torch.no_grad():
            prox_vectors[i] -= args.ipfl_eta*prox_vectors[i].grad
        prox_vectors[i].requires_grad_(False)
    graph_matrix += np.identity(args.n_parties)
    return prox_vectors

def aggregation_ipfl_cos_llm(args, graph_matrix, nets_this_round, global_dict):
    global_vec = weight_flatten_all_llm(global_dict)
    prox_vectors = {}
    for i in nets_this_round.keys():
        aggregation_weight_vector = graph_matrix[i]
        prox_vectors[i] = weight_flatten_all_llm(nets_this_round[i])
        prox_vectors[i].requires_grad_(True)
        loss = 0
        for j in nets_this_round.keys():
            model_j = weight_flatten_all_llm(nets_this_round[j])-global_vec
            loss -= aggregation_weight_vector[j]*torch.dot(prox_vectors[i]-global_vec,model_j)/torch.norm(prox_vectors[i]-global_vec)/torch.norm(model_j)
        loss.backward()
        with torch.no_grad():
            prox_vectors[i] -= args.ipfl_eta*prox_vectors[i].grad
        prox_vectors[i].requires_grad_(False)
    graph_matrix += np.identity(args.n_parties)
    return prox_vectors

