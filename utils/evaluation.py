import torch
import numpy as np

def evaluate_local_models(nets_this_round, test_dl, data_distributions, best_test_acc_list, benign_client_list):
    benign_current_acc = []
    for net_id, net in nets_this_round.items():
        if net_id in benign_client_list:
            data_distribution = data_distributions[net_id]
            personalized_test_acc, generalized_test_acc = compute_local_test_accuracy(net_id, net, test_dl, data_distribution, len(best_test_acc_list))
            benign_current_acc.append(personalized_test_acc)
            if personalized_test_acc > best_test_acc_list[net_id]:
                best_test_acc_list[net_id] = personalized_test_acc
            print('>> Client {} | Personalized Test Acc: ({:.5f}) | Generalized Test Acc: {:.5f}'.format(net_id, personalized_test_acc, generalized_test_acc))
    return np.array(benign_current_acc).mean()

def compute_local_test_accuracy(net_id, model, dataloader, data_distribution, n_client):
    num_classes = len(data_distribution)
    if isinstance(dataloader, tuple):   # leaf
        testloader, test_distribution = dataloader
        test_distribution = test_distribution * num_classes
        test_distribution[test_distribution == 0] = 1
    elif isinstance(dataloader, list):  # pacs
        testloader, test_distribution = dataloader[int(net_id/(n_client/len(dataloader)))]
        test_distribution = test_distribution * num_classes
        test_distribution[test_distribution == 0] = 1
    else:
        testloader = dataloader
        test_distribution = np.ones(num_classes)
    model.eval()
    toatl_label_num = np.zeros(len(data_distribution))
    correct_label_num = np.zeros(len(data_distribution))
    model.cuda()
    generalized_total, generalized_correct = 0, 0
    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(testloader):
            x, target = x.cuda(), target.to(dtype=torch.int64).cuda()
            out = model(x)
            _, pred_label = torch.max(out.data, 1)
            correct_filter = (pred_label == target.data)
            generalized_total += x.data.size()[0]
            generalized_correct += correct_filter.sum().item()
            for i, true_label in enumerate(target.data):
                toatl_label_num[true_label] += 1
                if correct_filter[i]:
                    correct_label_num[true_label] += 1

    personalized_correct = (correct_label_num * data_distribution / test_distribution).sum()
    personalized_total = (toatl_label_num * data_distribution / test_distribution).sum()

    model.to('cpu')
    return personalized_correct / personalized_total, generalized_correct / generalized_total

def compute_acc(net, test_data_loader):
    net.eval()
    correct, total = 0, 0
    net.cuda()
    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(test_data_loader):
            x, target = x.cuda(), target.to(dtype=torch.int64).cuda()
            out = net(x)
            _, pred_label = torch.max(out.data, 1)
            total += x.data.size()[0]
            correct += (pred_label == target.data).sum().item()
    net.to('cpu')
    return correct / float(total)

def evaluate_global_model(nets_this_round, global_model, test_dl, data_distributions, best_test_acc_list, benign_client_list):
    current_acc = []
    for net_id, _ in nets_this_round.items():
        if net_id in benign_client_list:
            data_distribution = data_distributions[net_id]
            personalized_test_acc, generalized_test_acc = compute_local_test_accuracy(net_id, global_model, test_dl, data_distribution, len(best_test_acc_list))
            current_acc.append(personalized_test_acc)
            if personalized_test_acc > best_test_acc_list[net_id]:
                best_test_acc_list[net_id] = personalized_test_acc
            print('>> Client {} | Personalized Test Acc: {:.5f} | Generalized Test Acc: {:.5f}'.format(net_id, personalized_test_acc, generalized_test_acc))
    return np.array(current_acc).mean()

# ===============  Model Difference ===============
def cal_model_difference(n_client, nets_this_round, global_parameters, metric='cos'):
    index_clientid = list(nets_this_round.keys())
    model_difference_matrix = torch.zeros((n_client, n_client))
    global_vec = weight_flatten_all(global_parameters)
    if metric == 'cos':
        for i in index_clientid:
            self_vec = weight_flatten_all(nets_this_round[i].state_dict()) - global_vec
            for j in index_clientid:
                neighbor_vec = weight_flatten_all(nets_this_round[j].state_dict()) - global_vec
                
                diff = 1 - torch.nn.functional.cosine_similarity(self_vec, neighbor_vec, dim=0, eps=1e-8)
                model_difference_matrix[i, j] = diff
  
    elif metric == 'l2':
        for i in index_clientid:
            self_vec = weight_flatten_all(nets_this_round[i].state_dict()) - global_vec
            for j in index_clientid:
                neighbor_vec = weight_flatten_all(nets_this_round[j].state_dict()) - global_vec
                model_difference_matrix[i, j] = torch.norm(self_vec-neighbor_vec) ** 2
    return model_difference_matrix

def cal_model_difference_llm(n_client, nets_this_round, global_parameters, metric='cos'):
    index_clientid = list(nets_this_round.keys())
    model_difference_matrix = torch.zeros((n_client, n_client))
    global_vec = weight_flatten_all(global_parameters)
    if metric == 'cos':
        for i in index_clientid:
            self_vec = weight_flatten_all(nets_this_round[i]) - global_vec
            for j in index_clientid:
                neighbor_vec = weight_flatten_all(nets_this_round[j]) - global_vec
                
                diff = 1 - torch.nn.functional.cosine_similarity(self_vec, neighbor_vec, dim=0, eps=1e-8)
                model_difference_matrix[i, j] = diff
  
    elif metric == 'l2':
        for i in index_clientid:
            self_vec = weight_flatten_all(nets_this_round[i]) - global_vec
            for j in index_clientid:
                neighbor_vec = weight_flatten_all(nets_this_round[j]) - global_vec
                model_difference_matrix[i, j] = torch.norm(self_vec-neighbor_vec) ** 2
    return model_difference_matrix

def weight_flatten(model):
    params = []
    for k in model:
        if 'fc' in k:
            params.append(model[k].reshape(-1))
    params = torch.cat(params)
    return params

def weight_flatten_all(model):
    params = []
    for k in model:
        params.append(model[k].reshape(-1))
    params = torch.cat(params)
    return params
