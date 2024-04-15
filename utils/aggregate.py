
def simple_model_aggregation(nets_this_round, global_model, fed_avg_freqs, graph_matrix):
    global_w = {}
    index_clientid = list(nets_this_round.keys())
    for index, net in enumerate(list(nets_this_round.values())):
        net_para = net.state_dict()
        if index == 0:
            for key in net_para:
                global_w[key] = net_para[key] * fed_avg_freqs[index]
        else:
            for key in net_para:
                global_w[key] += net_para[key] * fed_avg_freqs[index]
        graph_matrix[index_clientid[index]] = [0] * graph_matrix.shape[0]
        graph_matrix[index_clientid[index], index_clientid] = fed_avg_freqs
    global_model.load_state_dict(global_w)          # Update the global model