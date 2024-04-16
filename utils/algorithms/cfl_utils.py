import torch
import numpy as np
from sklearn.cluster import AgglomerativeClustering

def cal_model_sim_cluster_llm(n_client, nets_this_round, nets_param_start):
    model_similarity_matrix = torch.zeros((n_client, n_client))
    index_clientid = list(nets_this_round.keys())
    dW = [{} for i in range(n_client)]
    for id in index_clientid:
        model_i = nets_this_round[id]
        model_i_start = nets_param_start[id]
        for key in model_i:
            dW[id][key] =  model_i[key] - model_i_start[key]
        nets_this_round[id] = model_i_start
    for i in index_clientid:
        for j in index_clientid:
            sim = torch.dot(weight_flatten(dW[i]), weight_flatten(dW[j])) / (torch.norm(weight_flatten(dW[i])) * torch.norm(weight_flatten(dW[j]))+1e-8)
            model_similarity_matrix[i, j] = sim                                          
    return model_similarity_matrix, dW

def cal_model_sim_cluster(n_client, nets_this_round, nets_param_start):
    model_similarity_matrix = torch.zeros((n_client, n_client))
    index_clientid = list(nets_this_round.keys())
    dW = [{} for i in range(n_client)]
    for id in index_clientid:
        model_i = nets_this_round[id].state_dict()
        model_i_start = nets_param_start[id].state_dict()
        for key in nets_this_round[id].state_dict():
            dW[id][key] =  model_i[key] - model_i_start[key]
        nets_this_round[id].load_state_dict(model_i_start)
    for i in index_clientid:
        for j in index_clientid:
            sim = torch.dot(weight_flatten(dW[i]), weight_flatten(dW[j])) / (torch.norm(weight_flatten(dW[i])) * torch.norm(weight_flatten(dW[j]))+1e-8)
            model_similarity_matrix[i, j] = sim                                          
    return model_similarity_matrix, dW

def compute_max_update_norm(cluster):
    return np.max([torch.norm(weight_flatten(client_dw)).item() for client_dw in cluster])

def compute_mean_update_norm(cluster):
    return torch.norm(torch.mean(torch.stack([weight_flatten(client_dw) for client_dw in cluster]), dim=0)).item()

def weight_flatten(model):
    params = []
    for k in model:
        params.append(model[k].reshape(-1))
    params = torch.cat(params)
    return params

def cluster_clients(S):
    clustering = AgglomerativeClustering(affinity="precomputed", linkage="complete").fit(-S)
    c1 = np.argwhere(clustering.labels_ == 0).flatten() 
    c2 = np.argwhere(clustering.labels_ == 1).flatten() 
    return c1, c2


def reduce_add_average(targets, sources):
    for target in targets:
        for k, v in target.named_parameters():
            tmp = torch.mean(torch.stack([source[k].data for source in sources]), dim=0).clone()
            v.data += tmp

def reduce_add_average_llm(targets, sources):
    for target in targets:
        for k, v in target.items():
            tmp = torch.mean(torch.stack([source[k].data for source in sources]), dim=0).clone()
            v.data += tmp

def update_graph_cfl_llm(args, graph_matrix, nets_this_round, nets_param_start, cluster_indices):
    similarity, dW = cal_model_sim_cluster_llm(args.n_parties, nets_this_round, nets_param_start)
    cluster_indices_new = []
    index_clientid = list(nets_this_round.keys())
    # remove the id in cluster_indices if the client is not in index_clientid
    for idc in cluster_indices:
        cluster_indices_new += [np.array([i for i in idc if i in index_clientid])]
    cluster_indices = cluster_indices_new
    cluster_indices_new = []
    for idc in cluster_indices: # 对每个cluster
        max_norm = compute_max_update_norm([dW[i] for i in idc])
        mean_norm = compute_mean_update_norm([dW[i] for i in idc])
        print("max_norm", max_norm, "mean_norm", mean_norm)
        if mean_norm < 2.0 and max_norm > 2.5 and len(idc) > 2: # eps_1 = 2.0, eps_2 = 2.5
            c1, c2 = cluster_clients(similarity[idc][:,idc])
            print("new split", idc, idc[c1], idc[c2])
            cluster_indices_new += [idc[c1], idc[c2]]
        else:
            cluster_indices_new += [idc]
        
    cluster_indices = cluster_indices_new
    client_clusters = [[nets_this_round[i] for i in idcs] for idcs in cluster_indices]
    gradient_clusters = [[dW[i] for i in idcs] for idcs in cluster_indices]
    for i in range(len(cluster_indices)):
        reduce_add_average_llm(client_clusters[i], gradient_clusters[i])
    # update the graph_matrix
    graph_matrix = np.zeros((args.n_parties, args.n_parties))
    for idc in cluster_indices:
        for i in idc:
            for j in idc:
                graph_matrix[i, j] = 1/len(idc)

    return graph_matrix, cluster_indices

def update_graph_cfl(args, graph_matrix, nets_this_round, nets_param_start, cluster_indices):
    similarity, dW = cal_model_sim_cluster(args.n_parties, nets_this_round, nets_param_start)
    cluster_indices_new = []
    index_clientid = list(nets_this_round.keys())
    # remove the id in cluster_indices if the client is not in index_clientid
    for idc in cluster_indices:
        cluster_indices_new += [np.array([i for i in idc if i in index_clientid])]
    cluster_indices = cluster_indices_new
    cluster_indices_new = []
    for idc in cluster_indices: # 对每个cluster
        max_norm = compute_max_update_norm([dW[i] for i in idc])
        mean_norm = compute_mean_update_norm([dW[i] for i in idc])
        print("max_norm", max_norm, "mean_norm", mean_norm)
        if mean_norm < 2.0 and max_norm > 2.5 and len(idc) > 2: # eps_1 = 2.0, eps_2 = 2.5
            c1, c2 = cluster_clients(similarity[idc][:,idc])
            print("new split", idc, idc[c1], idc[c2])
            cluster_indices_new += [idc[c1], idc[c2]]
        else:
            cluster_indices_new += [idc]
        
    cluster_indices = cluster_indices_new
    client_clusters = [[nets_this_round[i] for i in idcs] for idcs in cluster_indices]
    gradient_clusters = [[dW[i] for i in idcs] for idcs in cluster_indices]
    for i in range(len(cluster_indices)):
        reduce_add_average(client_clusters[i], gradient_clusters[i])
    # update the graph_matrix
    graph_matrix = np.zeros((args.n_parties, args.n_parties))
    for idc in cluster_indices:
        for i in idc:
            for j in idc:
                graph_matrix[i, j] = 1/len(idc)

    return graph_matrix, cluster_indices