import torch
import numpy as np
import copy
import cvxpy as cp
import torch.nn.functional as F

def update_graph_pfedgraph(graph_matrix, nets_this_round, global_parameters, fed_avg_freqs, alpha):
    index_clientid = list(nets_this_round.keys())
    model_difference_matrix = cal_model_difference_pfedgraph(graph_matrix.shape[0], nets_this_round, global_parameters)
    graph_matrix = optimizing_graph_pfedgraph(graph_matrix, index_clientid, model_difference_matrix, alpha, fed_avg_freqs)

    return graph_matrix

def update_graph_pfedgraph_llm(graph_matrix, nets_this_round, global_parameters, fed_avg_freqs, alpha):
    index_clientid = list(nets_this_round.keys())
    model_difference_matrix = cal_model_difference_pfedgraph_llm(graph_matrix.shape[0], nets_this_round, global_parameters)
    graph_matrix = optimizing_graph_pfedgraph(graph_matrix, index_clientid, model_difference_matrix, alpha, fed_avg_freqs)

    return graph_matrix

# 之所以要传原来的graph_matrix，是因为只有部分client参与训练，所以graph_matrix中只有部分行列需要更新
def optimizing_graph_pfedgraph(graph_matrix, index_clientid, model_difference_matrix, lamba, fed_avg_freqs):
    n = len(index_clientid)
    p = np.array(fed_avg_freqs)
    P = lamba * np.identity(n)
    P = cp.atoms.affine.wraps.psd_wrap(P)
    G = - np.identity(n)
    h = np.zeros(n)
    A = np.ones((1, n))
    b = np.ones(1)
    for id in index_clientid:
        model_difference_vector = model_difference_matrix[id, index_clientid]
        d = model_difference_vector.numpy()
        q = d - 2 * lamba * p
        x = cp.Variable(n)
        prob = cp.Problem(cp.Minimize(cp.quad_form(x, P) + q.T @ x),
                  [G @ x <= h,
                   A @ x == b]
                  )
        prob.solve()
        # 如果x.value未出现 nan
        if np.isnan(x.value).any():
            print("x.value is nan")
            continue   # 不更新
        # x.value中小于1/n的值置0
        x.value[x.value < (1/n/1e5)] = 0
        graph_matrix[id, index_clientid] = F.normalize(torch.tensor(x.value), p=1, dim=0)
            
    return graph_matrix

def aggregation_pfedgraph(graph_matrix, nets_this_round, global_w, cluster_models):
    tmp_client_state_dict = {}
    for client_id in nets_this_round.keys():
        tmp_client_state_dict[client_id] = copy.deepcopy(global_w)
        for key in tmp_client_state_dict[client_id]:
            tmp_client_state_dict[client_id][key] = torch.zeros_like(tmp_client_state_dict[client_id][key]).float()

    for client_id in nets_this_round.keys():
        tmp_client_state = tmp_client_state_dict[client_id]
        aggregation_weight_vector = graph_matrix[client_id]

        for neighbor_id in nets_this_round.keys():
            net_para = nets_this_round[neighbor_id].state_dict()
            for key in tmp_client_state:
                tmp_client_state[key] += net_para[key] * aggregation_weight_vector[neighbor_id]

    for client_id in nets_this_round.keys():
        cluster_models[client_id].load_state_dict(tmp_client_state_dict[client_id])
        nets_this_round[client_id].load_state_dict(tmp_client_state_dict[client_id])

def aggregation_pfedgraph_llm(graph_matrix, nets_this_round, global_w, cluster_dict_list):
    tmp_client_state_dict = {}
    for client_id in nets_this_round.keys():
        tmp_client_state_dict[client_id] = copy.deepcopy(global_w)
        for key in tmp_client_state_dict[client_id]:
            tmp_client_state_dict[client_id][key] = torch.zeros_like(tmp_client_state_dict[client_id][key]).float()

    for client_id in nets_this_round.keys():
        tmp_client_state = tmp_client_state_dict[client_id]
        aggregation_weight_vector = graph_matrix[client_id]

        for neighbor_id in nets_this_round.keys():
            net_para = nets_this_round[neighbor_id]
            for key in tmp_client_state:
                tmp_client_state[key] += net_para[key] * aggregation_weight_vector[neighbor_id]

    for client_id in nets_this_round.keys():
        cluster_dict_list[client_id] = tmp_client_state_dict[client_id]
        nets_this_round[client_id] = copy.deepcopy(tmp_client_state_dict[client_id])

def cal_model_difference_pfedgraph(n_client, nets_this_round, global_parameters):
    index_clientid = list(nets_this_round.keys())
    model_difference_matrix = torch.zeros((n_client, n_client))
    global_vec = weight_flatten_all(global_parameters)
    for i in index_clientid:
        self_vec = weight_flatten_all(nets_this_round[i].state_dict())-global_vec
        for j in index_clientid:
            neighbor_vec = weight_flatten_all(nets_this_round[j].state_dict())-global_vec
            diff = - F.cosine_similarity(self_vec.unsqueeze(0), neighbor_vec.unsqueeze(0), eps=1e-8)
            model_difference_matrix[i, j] = diff
    return model_difference_matrix

def cal_model_difference_pfedgraph_llm(n_client, nets_this_round, global_parameters):
    index_clientid = list(nets_this_round.keys())
    model_difference_matrix = torch.zeros((n_client, n_client))
    global_vec = weight_flatten_all(global_parameters)
    for i in index_clientid:
        self_vec = weight_flatten_all(nets_this_round[i])-global_vec
        for j in index_clientid:
            neighbor_vec = weight_flatten_all(nets_this_round[j])-global_vec
            diff = - F.cosine_similarity(self_vec.unsqueeze(0), neighbor_vec.unsqueeze(0), eps=1e-8)
            model_difference_matrix[i, j] = diff
    return model_difference_matrix

def weight_flatten_all(model):
    params = []
    for k in model:
        params.append(model[k].reshape(-1))
    params = torch.cat(params)
    return params