import numpy as np
import random

def Cal_Utility(args, graph_matrix, nets_this_round, gain_list, data_num_list, data_num_list_reported, cost_list, model_difference_matrix, utility, benign_client_list):
    index_clientid = list(nets_this_round.keys())
    utility_this_round = np.zeros(graph_matrix.shape[0])
    payment_matrix = np.zeros_like(graph_matrix)
    for id in index_clientid:
        selected_clients = np.where(abs(graph_matrix[id]) != 0)[0]
        if gain_list[id] == 0:
            gain = 0
        else:
            gain = Get_Gain(gain_list[id], selected_clients, data_num_list, id)
        payment_matrix += Get_Payment(args, graph_matrix, gain_list, data_num_list_reported, model_difference_matrix, id)
        cost = Get_Cost(cost_list, graph_matrix, id)
        utility_id = gain - np.sum(payment_matrix[id]) - cost
        utility_this_round[id] = utility_id
        print("Client {}: gain={:.5f}, payment={:.5f}, cost={:.2f}, utility={:.5f}".format(id, gain, np.sum(payment_matrix[id]), cost, utility_id))
    utility.append(utility_this_round)
    return np.mean(utility_this_round[benign_client_list]),payment_matrix

def Init_incentive_properties(args, client_num_samples):
    utility = []
    data_num_list_reported = np.array(client_num_samples.copy())
    data_num_list_real = client_num_samples.copy()
    cost_list_real = [args.C] * args.n_parties
    cost_list_reported = [args.C] * args.n_parties
    gain_k_list = [args.K] * args.n_parties
    # initialize the graph matrix
    if args.alg == 'pfedgraph':
        graph_matrix = np.ones((args.n_parties, args.n_parties))/(args.n_parties-1)                 # Collaboration Graph
        graph_matrix[range(args.n_parties), range(args.n_parties)] = 0
    elif args.alg == 'local':
        graph_matrix = np.identity(args.n_parties)
    else:
        fed_avg_freqs = data_num_list_reported / sum(data_num_list_reported)
        graph_matrix = np.tile(fed_avg_freqs, (args.n_parties, 1))
    # initialize the utility, gain_k_list, data_num_list, cost_list
    if not args.diverse:
        malicious_clients = malicious_clients_generator(args, list(range(args.n_parties)), gain_k_list, data_num_list_reported, cost_list_reported, cost_list_real) # 改变数据量或者cost
    else:
        """
        define different types of agents
        1. BB(balanced buyer): gain()=\sqrt{K/n_i}, cost=[0.8, 1.2] gaussian; 
        2. SB(stingy buyer): gain()=\sqrt{K/n_i}, cost=100 (very large)
        3. GS(generous seller): gain()=0, cost=0; 
        4. BS(balanced seller): gain()=0, cost=[0.8, 1.2] gaussian
        5. MC-liar: gain()=\sqrt{K/n_i}, cost=C*exg_rate; 
        6. MC-freeride: gain()=\sqrt{K/n_i}, cost=0; 
        7. MC-attack: gain()=0, cost=0
        Note, type 7 is from GS
        for all agents, we randomly assign them to BB, SB, GS, BS
        """
        assert args.n_parties == 12
        BB_list, SB_list, GS_list, BS_list = [3, 4, 6], [1, 2, 8, 9], [0, 5, 7, 10, 11], []
        gain_k_list = [args.K] * args.n_parties
        for agent in BB_list:
            gain_k_list[agent] = np.random.uniform(args.K/2, args.K)
            cost_list_real[agent] = np.random.uniform(args.C/2, args.C)
            cost_list_reported[agent] = cost_list_real[agent]
        for agent in SB_list:
            gain_k_list[agent] = np.random.uniform(args.K/2, args.K)
            cost_list_real[agent] = 1000
            cost_list_reported[agent] = cost_list_real[agent]
        for agent in GS_list:
            gain_k_list[agent] = 0
            cost_list_real[agent] = 0
            cost_list_reported[agent] = 0
        for agent in BS_list:
            gain_k_list[agent] = 0
            cost_list_real[agent] = np.random.uniform(0.8, 1.2)
            cost_list_reported[agent] = cost_list_real[agent]
        liars, freeriders, attackers = [], [], [11]
        malicious_clients = (liars, freeriders, attackers)
        print('>> ---BB: {} | SB: {} | GS: {} | BS: {}---'.format(BB_list, SB_list, GS_list, BS_list))
        print('>> ---liars: {} | freeriders: {} | attackers: {}---'.format(liars, freeriders, attackers))
    print(">> ---gain_k_list: {}, cost_list:{} ---".format(gain_k_list, cost_list_real))
    return utility, gain_k_list, data_num_list_real, data_num_list_reported, cost_list_real, cost_list_reported, graph_matrix, malicious_clients

def malicious_clients_generator(args, client_list, gain_k_list, data_num_list, cost_list_reported, cost_list_real):
    # determine the liars, freeriders and attackers
    liars, freeriders, attackers = [], [], []
    if args.liar_ratio > 0:
        liars = random.sample(client_list, int(len(client_list) * args.liar_ratio))
        liars.sort()
        exaggerate_ratio = np.ones(len(liars)) * args.liar_exaggerate_ratio
        if args.liar_type == "size":
            for liar in liars:
                data_num_list[liar] = int(data_num_list[liar] * exaggerate_ratio[liars.index(liar)])
        elif args.liar_type == 'cost':
            for liar in liars:
                cost_list_reported[liar] = cost_list_reported[liar] * exaggerate_ratio[liars.index(liar)]
        print('>> -------- Liars on {}: {} --------'.format(args.liar_type, liars))
    elif args.freerider_ratio > 0:
        freeriders = random.sample(client_list, int(len(client_list) * args.freerider_ratio))
        for id in freeriders:
            cost_list_real[id] = 0
        print('>> -------- Freeriders: {} --------'.format(freeriders))
    elif args.attack_ratio > 0:
        attackers = random.sample(client_list, int(len(client_list) * args.attack_ratio))
        attackers.sort()
        for id in attackers:
            gain_k_list[id] = 0
            cost_list_reported[id] = 0
            cost_list_real[id] = 0
        print('>> -------- Attackers: {} --------'.format(attackers))

    return (liars, freeriders, attackers)

# FL进程结束后，计算malicious clients的utility
def Cal_MC_Utility(utility_this_round, best_test_acc_list, liars, freeriders, attackers):
    if len(liars) > 0:
        print('>> -------- Liars {} (mean) Utility: {} | Acc: {} --------'.format(liars, np.mean(utility_this_round[liars]), np.mean(best_test_acc_list[liars])))
    if len(freeriders) > 0:
        print('>> -------- Freeriders {} (mean) Utility: {} --------'.format(freeriders, np.mean(utility_this_round[freeriders])))
    if len(attackers) > 0:
        print('>> -------- Attackers {} (mean) Utility: {} --------'.format(attackers, np.mean(utility_this_round[attackers])))


def Cal_Social_Welfare(args, graph_matrix, data_num_list, cost_list):
    social_welfare = 0
    for i in range(graph_matrix.shape[0]):
        selected_clients = np.where(abs(graph_matrix[i]) != 0)[0]
        social_welfare += Get_Gain(args.K, selected_clients, data_num_list, i) - Get_Cost(cost_list, graph_matrix, i)
    return social_welfare

def Get_Gain(K, selected_clients, data_num_list, self_id): # selected_clients包含了自己
    data_amount = sum([data_num_list[j] for j in selected_clients])
    return np.sqrt(K/data_num_list[self_id]) - np.sqrt(K/data_amount)

def Get_Payment(args, graph_matrix, gain_list, data_num_list, model_difference_matrix, self_id):
    payment_matrix = np.zeros_like(graph_matrix)
    if args.alg != 'ipfl':
        return payment_matrix
    
    # === expense ===
    selected_clients = list(np.where(abs(graph_matrix[self_id]) != 0)[0])

    if len(selected_clients) > 1: # 该客户端有合作 Local Training
        for j in selected_clients:
            if j == self_id:
                continue
            # remove id from selected_clients
            selected_clients_i = selected_clients.copy()
            selected_clients_i.remove(j)
            payment_matrix[self_id][j] += Get_Gain(gain_list[self_id], selected_clients, data_num_list, self_id)
            payment_matrix[self_id][j] -= Get_Gain(gain_list[self_id], selected_clients_i, data_num_list, self_id)
            payment_matrix[self_id][j] -= model_difference_matrix[self_id,j]*data_num_list[j]/data_num_list[self_id]*args.ipfl_lam

    # === revenue ===
    for j in range(graph_matrix.shape[0]):
        if j == self_id or graph_matrix[j, self_id] == 0: # j 没有与 i 合作
            continue
        j_selected_clients = list(np.where(abs(graph_matrix[j]) != 0)[0])
        payment_matrix[self_id][j] += model_difference_matrix[j, self_id]*data_num_list[self_id]/data_num_list[j] * args.ipfl_lam
        payment_matrix[self_id][j] -= Get_Gain(gain_list[j], j_selected_clients, data_num_list, j)
        j_selected_clients.remove(self_id)
        payment_matrix[self_id][j] += Get_Gain(gain_list[j], j_selected_clients, data_num_list, j) 
    return payment_matrix

def Get_Cost(cost_list, graph_matrix, self_id):
    num_export = np.count_nonzero(graph_matrix[:, self_id])
    if graph_matrix[self_id, self_id] != 0:
        num_export -= 1
    return cost_list[self_id] * num_export

def Get_Marginal_Cost(K, chosen_clients, index_clientid, data_num_list, cost_list, model_div_list, self_id, type='ADD'): # model_div_list长度为index_clientid的长度
    marginal_cost = [0] * len(data_num_list) 
    selected_clients = chosen_clients.copy()
    selected_clients.append(self_id)
    if type == 'ADD':
        for id in index_clientid:
            if id != self_id and id not in chosen_clients:
                selected_clients_i = selected_clients.copy()
                selected_clients_i.append(id)
                delta_gain = -Get_Gain(K, selected_clients_i, data_num_list, self_id) + Get_Gain(K, selected_clients, data_num_list, self_id)
                marginal_cost[id] = delta_gain + cost_list[id] + model_div_list[id]*data_num_list[id]/data_num_list[self_id]

    elif type == 'POP':
        for id in index_clientid:
            if id in chosen_clients:
                selected_clients_i = selected_clients.copy()
                selected_clients_i.remove(id)
                delta_gain = -Get_Gain(K, selected_clients_i, data_num_list, self_id) + Get_Gain(K, selected_clients, data_num_list, self_id)
                marginal_cost[id] = delta_gain - cost_list[id] - model_div_list[id]*data_num_list[id]/data_num_list[self_id]

    return marginal_cost