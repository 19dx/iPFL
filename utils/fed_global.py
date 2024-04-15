import random
import torch
import sys
from .algorithms.pfedgraph_utils import aggregation_pfedgraph_llm, update_graph_pfedgraph_llm
from .algorithms.fedamp_utils import aggregation_fedamp_llm, update_graph_fedamp_llm
from .algorithms.ipfl_utils import aggregation_ipfl_cos_llm, update_graph_ipfl
from .algorithms.cfl_utils import update_graph_cfl_llm

# ========= For Instruction-tuning tasks =========
def Model_aggregation(args, graph_matrix, nets_dict_this_round, fed_avg_freqs, global_dict, cluster_dict_list):
    prox_vectors = [None] * args.n_parties
    if args.alg in ['fedavg', 'fedprox']:
        for index, net in enumerate(list(nets_dict_this_round.values())):
            if index == 0:
                for key in net:
                    global_dict[key] = net[key] * fed_avg_freqs[index]
            else:
                for key in net:
                    global_dict[key] += net[key] * fed_avg_freqs[index]
    elif args.alg == 'pfedgraph':
        aggregation_pfedgraph_llm(graph_matrix, nets_dict_this_round, global_dict, cluster_dict_list) 
    elif args.alg == 'fedamp':
        aggregation_fedamp_llm(graph_matrix, nets_dict_this_round, cluster_dict_list)      
    elif args.alg == 'ipfl':
        prox_vectors = aggregation_ipfl_cos_llm(args, graph_matrix, nets_dict_this_round, global_dict)  
    return prox_vectors                            

def Update_graph_matrix(args, graph_matrix, nets_this_round, global_dict, nets_param_start, cluster_indices, fed_avg_freqs, data_num_list, cost_list, gain_list, model_difference_matrix):
    if args.alg in ['fedavg', 'fedprox']:
        index_clientid = list(nets_this_round.keys())
        for id in index_clientid:
            graph_matrix[id] = torch.zeros(args.n_parties)
            graph_matrix[id, index_clientid] = fed_avg_freqs
        return graph_matrix, cluster_indices
    elif args.alg == 'local':
        return torch.eye(args.n_parties), cluster_indices
    elif args.alg == 'cfl':
        graph_matrix, cluster_indices = update_graph_cfl_llm(args, graph_matrix, nets_this_round, nets_param_start, cluster_indices)
        return graph_matrix, cluster_indices
    elif args.alg == 'pfedgraph':
        return update_graph_pfedgraph_llm(graph_matrix, nets_this_round, global_dict, fed_avg_freqs, 0.8), cluster_indices
    elif args.alg == 'fedamp':
        return update_graph_fedamp_llm(graph_matrix, nets_this_round, global_dict), cluster_indices
    elif args.alg == 'ipfl':
        return update_graph_ipfl(args, graph_matrix, nets_this_round, gain_list, data_num_list, cost_list, model_difference_matrix), cluster_indices
    else:
        raise NotImplementedError
