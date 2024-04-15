import torch
import random

def Dropout(args, graph_matrix, utility_this_round, party_list, benign_client_list):
    if args.alg == 'local':
        return
    if args.dropout: 
        if args.dropout_type == "IR-strict": # remove the negative utility clients
            for id in range(args.n_parties):
                if utility_this_round[id] < 0:
                    graph_matrix[id] = torch.zeros(args.n_parties)
                    party_list.remove(id)
                    benign_client_list.remove(id)
        elif args.dropout_type == 'IR-prob':
            for id in range(args.n_parties):
                if utility_this_round[id] < 0:
                    if random.random() < args.dropout_p:
                        graph_matrix[id] = torch.zeros(args.n_parties)
                        party_list.remove(id)
                        benign_client_list.remove(id)

