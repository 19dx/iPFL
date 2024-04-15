import torch
import numpy as np
import copy
import cvxpy as cp

def update_graph_fedamp(graph_matrix, nets_this_round, global_parameters):
    index_clientid = list(nets_this_round.keys())
    model_similarity_matrix = cal_model_difference_fedamp(graph_matrix.shape[0], nets_this_round, global_parameters)
    graph_matrix = calculate_graph_matrix(graph_matrix, index_clientid, model_similarity_matrix)

    return graph_matrix

def update_graph_fedamp_llm(graph_matrix, nets_this_round, global_parameters):
    index_clientid = list(nets_this_round.keys())
    model_similarity_matrix = cal_model_difference_fedamp_llm(graph_matrix.shape[0], nets_this_round, global_parameters)
    graph_matrix = calculate_graph_matrix(graph_matrix, index_clientid, model_similarity_matrix)

    return graph_matrix

def calculate_graph_matrix(graph_matrix, index_clientid, model_similarity_matrix):
    self_weight = 0.3
    if len(index_clientid) == 1:
        graph_matrix[index_clientid[0]] = torch.zeros(graph_matrix.shape[0])
        graph_matrix[index_clientid[0]][index_clientid[0]] = 1
        return graph_matrix
    for id in index_clientid:
        weight = torch.zeros(graph_matrix.shape[0])
        weight[index_clientid] = torch.exp(10 * model_similarity_matrix[id, index_clientid])
        # check if nan in weight
        if torch.isnan(weight).any():
            continue
        weight[weight < 1/(graph_matrix.shape[0]*1e4)] = 0 # set the weight to 0 if it is too small
        weight[id] = 0
        weight = (1 - self_weight) * weight / weight.sum()
        weight[id] = self_weight
        graph_matrix[id] = weight
        
    return graph_matrix


def aggregation_fedamp(graph_matrix, nets_this_round, cluster_models):
    tmp_client_state_dict = {}
    # 
    state_dict_template = nets_this_round[list(nets_this_round.keys())[0]].state_dict()
    for client_id in nets_this_round.keys():
        tmp_client_state_dict[client_id] = copy.deepcopy(state_dict_template)
        for key in tmp_client_state_dict[client_id]:
            tmp_client_state_dict[client_id][key] = torch.zeros_like(tmp_client_state_dict[client_id][key]).float()

    for client_id in nets_this_round.keys():
        tmp_client_state = tmp_client_state_dict[client_id]
        aggregation_weight_vector = graph_matrix[client_id]

        for neighbor_id in nets_this_round.keys():
            net_para = nets_this_round[neighbor_id].state_dict()
            for key in tmp_client_state:
                # print(type(tmp_client_state[key]), type(net_para[key]), type(aggregation_weight_vector[neighbor_id]))
                tmp_client_state[key] += net_para[key] * aggregation_weight_vector[neighbor_id]
    for client_id in nets_this_round.keys():
        cluster_models[client_id].load_state_dict(tmp_client_state_dict[client_id])

def aggregation_fedamp_llm(graph_matrix, nets_this_round, cluster_dict_list):
    tmp_client_state_dict = {}
    # 
    state_dict_template = nets_this_round[list(nets_this_round.keys())[0]]
    for client_id in nets_this_round.keys():
        tmp_client_state_dict[client_id] = copy.deepcopy(state_dict_template)
        for key in tmp_client_state_dict[client_id]:
            tmp_client_state_dict[client_id][key] = torch.zeros_like(tmp_client_state_dict[client_id][key]).float()

    for client_id in nets_this_round.keys():
        tmp_client_state = tmp_client_state_dict[client_id]
        aggregation_weight_vector = graph_matrix[client_id]

        for neighbor_id in nets_this_round.keys():
            net_para = nets_this_round[neighbor_id]
            for key in tmp_client_state:
                # print(type(tmp_client_state[key]), type(net_para[key]), type(aggregation_weight_vector[neighbor_id]))
                tmp_client_state[key] += net_para[key] * aggregation_weight_vector[neighbor_id]
    for client_id in nets_this_round.keys():
        cluster_dict_list[client_id] = tmp_client_state_dict[client_id]


def cal_model_difference_fedamp(n_client, nets_this_round, global_parameters):
    index_clientid = list(nets_this_round.keys())
    model_similarity_matrix = torch.zeros((n_client, n_client))
    global_vec = weight_flatten_all(global_parameters)
    for i in index_clientid:
        self_vec = weight_flatten_all(nets_this_round[i].state_dict())-global_vec
        for j in index_clientid:
            neighbor_vec = weight_flatten_all(nets_this_round[j].state_dict())-global_vec
            sim = torch.nn.functional.cosine_similarity(self_vec.unsqueeze(0), neighbor_vec.unsqueeze(0), eps=1e-8)
            model_similarity_matrix[i, j] = sim
    return model_similarity_matrix

def cal_model_difference_fedamp_llm(n_client, nets_this_round, global_parameters):
    index_clientid = list(nets_this_round.keys())
    model_similarity_matrix = torch.zeros((n_client, n_client))
    global_vec = weight_flatten_all(global_parameters)
    for i in index_clientid:
        self_vec = weight_flatten_all(nets_this_round[i])-global_vec
        for j in index_clientid:
            neighbor_vec = weight_flatten_all(nets_this_round[j])-global_vec
            sim = torch.nn.functional.cosine_similarity(self_vec.unsqueeze(0), neighbor_vec.unsqueeze(0), eps=1e-8)
            model_similarity_matrix[i, j] = sim
    return model_similarity_matrix

def weight_flatten_all(model):
    params = []
    for k in model:
        params.append(model[k].reshape(-1))
    params = torch.cat(params)
    return params
