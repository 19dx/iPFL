import numpy as np
import torch
import copy

def update_graph_fedfomo(M, graph_matrix, P, nets_this_round):
    index_clientid = list(nets_this_round.keys())
    for id in index_clientid:
        pid = np.array(P[id])[index_clientid]/sum(P[id, index_clientid])
        m = min(M, len(index_clientid))
        # select top m clients' model  
        selected_clients = []
        while len(selected_clients) < m:
            selected_clients.append(index_clientid[np.random.choice(len(index_clientid), 1, p=pid)[0]])
            selected_clients = list(set(selected_clients))
        graph_matrix[id] = np.zeros(graph_matrix.shape[0])
        graph_matrix[id][selected_clients] = 1
            
def aggregation_fedfomo(graph_matrix_old, nets_this_round, train_local_dls, nets_param_start, P):
    graph_matrix = copy.deepcopy(graph_matrix_old)
    index_clientid = list(nets_this_round.keys())
    for id in index_clientid:
        data_loader = train_local_dls[id]
        loss_prev = compute_loss(nets_param_start[id], data_loader)
        neighbors = np.where(graph_matrix[id] !=0)[0]
        normalize_flag = False
        for nid in neighbors:
            # calculate the weight
            loss_nid = compute_loss(nets_this_round[nid], data_loader)
            if loss_nid >= loss_prev:
                graph_matrix[id][nid] = 0
            else:
                graph_matrix[id][nid] = (loss_prev-loss_nid)/(torch.norm(weight_flatten_all(nets_this_round[id])-weight_flatten_all(nets_this_round[nid])+1e-6))
                normalize_flag = True
        if normalize_flag:
            graph_matrix[id] = graph_matrix[id] / graph_matrix[id].sum()
            P[id] += graph_matrix[id]
            
    # aggregate model
    for id in index_clientid:
        aggregation_weight_vector = graph_matrix[id]
        neighbors = np.where(graph_matrix[id] !=0)[0]
        if len(neighbors > 0):
            tmp_client_state = copy.deepcopy(nets_this_round[id].state_dict())
            for key in tmp_client_state.keys():
                tmp_client_state[key] = torch.zeros_like(tmp_client_state[key]).float()
            for nid in neighbors:
                net_para = nets_this_round[nid].state_dict()
                for key in tmp_client_state:
                    tmp_client_state[key] += net_para[key] * aggregation_weight_vector[nid]
            nets_this_round[id].load_state_dict(tmp_client_state)
        else: # 不改变
            graph_matrix[id, id] = 1
            nets_this_round[id].load_state_dict(copy.deepcopy(nets_param_start[id].state_dict()))

def compute_loss(net, test_data_loader):
    net.eval()
    criterion = torch.nn.CrossEntropyLoss()
    net.cuda()
    loss_collector = []
    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(test_data_loader):
            x, target = x.cuda(), target.to(dtype=torch.int64).cuda()
            out = net(x)
            loss = criterion(out, target)
            loss_collector.append(loss.item())
    net.to('cpu')
    return np.array(loss_collector).mean()


def weight_flatten_all(model):
    model_parm = model.state_dict()
    params = []
    for k in model_parm:
        params.append(model_parm[k].reshape(-1))
    params = torch.cat(params)
    return params