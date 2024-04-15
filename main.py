import torch
import copy
import random
import time
from config import get_args
args = get_args()

from utils.utils import set_seed
from utils.model import get_model
from utils.dataset.dataset import get_dataloader
from utils.algorithms.incentive_utils import Init_incentive_properties, Cal_MC_Utility, Cal_Utility
from utils.algorithms.fedfomo_utils import update_graph_fedfomo, aggregation_fedfomo
from utils.algorithms.fedamp_utils import update_graph_fedamp, aggregation_fedamp
from utils.algorithms.cfl_utils import update_graph_cfl
from utils.algorithms.ipfl_utils import aggregation_ipfl_cos, update_graph_ipfl
from utils.algorithms.pfedgraph_utils import update_graph_pfedgraph, aggregation_pfedgraph

from utils.aggregate import *
from utils.evaluation import evaluate_global_model, evaluate_local_models, cal_model_difference
from utils.local_training import *
from utils.dropout import *
from utils.attack import *
import warnings
warnings.filterwarnings("ignore")

Start_time = time.time()
set_seed(args.init_seed)

# Set up dataloader
if args.dataset in ("shakespeare", 'femnist'):
    train_local_dls, test_dl, client_num_samples, traindata_cls_counts, data_distributions = get_dataloader(args)
elif args.dataset =='pacs': # or args.dataset == 'officehome' or args.dataset == 'vlcs':
    train_local_dls, test_dl, client_num_samples, data_distributions, domain = get_dataloader(args) # test_dl is a list consisting 4 dataloader here
else:
    train_local_dls, test_dl, client_num_samples, traindata_cls_counts, data_distributions = get_dataloader(args)

utility, gain_list, data_num_list_real, data_num_list_reported, cost_list_real, cost_list_reported, graph_matrix, malicious_clients = Init_incentive_properties(args, client_num_samples)
(liars, freeriders, attackers) = malicious_clients
total_payment_matrix = np.zeros_like(graph_matrix)
total_graph_matrix = np.zeros_like(graph_matrix)
# Set up model
global_model = get_model(args)
global_model.load_state_dict(torch.load(args.load_path)) if args.load_path is not None else None    # Load saved model if exists
global_parameters = global_model.state_dict()
local_models = [copy.deepcopy(global_model) for _ in range(args.n_parties)]
ditto_models = [copy.deepcopy(global_model) for _ in range(args.n_parties)] if args.alg == 'ditto' else None
cluster_models = [copy.deepcopy(global_model) for _ in range(args.n_parties)] if args.alg in ('fedamp', 'ipfl', 'pfedgraph') else None
prox_vectors, model_difference_matrix = None, None
cluster_indices =  [np.arange(args.n_parties).astype("int")] if args.alg == 'cfl' else None

party_list = [i for i in range(args.n_parties)]
benign_client_list = list(set(party_list) - set(freeriders) - set(attackers)) # calculate the acc
best_test_acc_list = np.zeros(args.n_parties)
# ========== Federated Learning ==========
print(f"{'='*20} Start Federated Learning {'='*20}")  
for round in range(args.comm_round):
    if len(party_list) < args.n_parties:
        print('>> Round {} | Available Clients: {} <<'.format(round, party_list))
    # Federated Process: sync model / local train / global aggregate / global evaluate
    nets_this_round = {k: local_models[k] for k in party_list}
    ditto_this_round = {k: ditto_models[k] for k in party_list} if args.alg=='ditto' else None
    nets_param_start = {k: copy.deepcopy(local_models[k]) for k in party_list}
    total_num_samples = sum(data_num_list_reported[k] for k in party_list)
    fed_avg_freqs = [data_num_list_reported[k] / total_num_samples for k in party_list]

    if args.alg in ('fedavg', 'fedprox'):  # not personalized FL
        global_w = global_model.state_dict()
        for net in nets_this_round.values():
            net.load_state_dict(global_w)
        # Local Model Training
        if args.alg == 'fedprox':
            local_train_fedprox(args, nets_this_round, global_model, train_local_dls, freeriders)
        else:
            local_train_fedavg(args, nets_this_round, train_local_dls, freeriders)
        manipulate_gradient(args, nets_this_round, (freeriders, attackers))
        simple_model_aggregation(nets_this_round, global_model, fed_avg_freqs, graph_matrix)
        mean_personalized_acc = evaluate_global_model(nets_this_round, global_model, test_dl, data_distributions, best_test_acc_list, benign_client_list)
        
    elif args.alg == 'local':
        mean_personalized_acc = local_train(args, nets_this_round, train_local_dls, test_dl, data_distributions, best_test_acc_list)
        graph_matrix = np.identity(args.n_parties)
    
    elif args.alg == 'fedfomo':
        if round == 0:
            P = np.ones((args.n_parties, args.n_parties))
        local_train_fedavg(args, nets_this_round, train_local_dls, freeriders)
        mean_personalized_acc = evaluate_local_models(nets_this_round, test_dl, data_distributions, best_test_acc_list, benign_client_list)
        manipulate_gradient(args, nets_this_round, (freeriders, attackers)) 
        update_graph_fedfomo(args.fedfomo_M, graph_matrix, P, nets_this_round)
        aggregation_fedfomo(graph_matrix, nets_this_round, train_local_dls, nets_param_start, P)
     
    elif args.alg == 'cfl':
        local_train_fedavg(args, nets_this_round, train_local_dls, freeriders)
        mean_personalized_acc = evaluate_local_models(nets_this_round, test_dl, data_distributions, best_test_acc_list, benign_client_list)
        manipulate_gradient(args, nets_this_round, (freeriders, attackers)) 
        graph_matrix, cluster_indices = update_graph_cfl(args, graph_matrix, nets_this_round, nets_param_start, cluster_indices)

    elif args.alg == 'ditto':
        global_w = global_model.state_dict()
        for net in nets_this_round.values():
            net.load_state_dict(global_w)
        local_train_ditto(args, nets_this_round, ditto_this_round, train_local_dls, freeriders)
        mean_personalized_acc = evaluate_local_models(ditto_this_round, test_dl, data_distributions, best_test_acc_list, benign_client_list)
        manipulate_gradient(args, nets_this_round, (freeriders, attackers))
        simple_model_aggregation(nets_this_round, global_model, fed_avg_freqs, graph_matrix)
    
    elif args.alg == 'fedamp':
        local_train_fedamp(args, round, nets_this_round, cluster_models, train_local_dls, freeriders)
        mean_personalized_acc = evaluate_local_models(nets_this_round, test_dl, data_distributions, best_test_acc_list, benign_client_list)
        manipulate_gradient(args, nets_this_round, (freeriders, attackers))
        graph_matrix = update_graph_fedamp(graph_matrix, nets_this_round, global_parameters)   # Graph Matrix is not normalized yet
        aggregation_fedamp(graph_matrix, nets_this_round, cluster_models)                                                    # Aggregation weight is normalized here
    
    elif args.alg == 'ipfl':
        local_train_ipfl(args, round, nets_this_round, prox_vectors, train_local_dls, freeriders)
        mean_personalized_acc = evaluate_local_models(nets_this_round, test_dl, data_distributions, best_test_acc_list, benign_client_list)
        manipulate_gradient(args, nets_this_round, (freeriders, attackers))
        model_difference_matrix = cal_model_difference(args.n_parties, nets_this_round, global_parameters, args.difference_metric)
        graph_matrix = update_graph_ipfl(args, graph_matrix, nets_this_round, gain_list, data_num_list_reported, cost_list_reported, model_difference_matrix)
        prox_vectors = aggregation_ipfl_cos(args, graph_matrix, nets_this_round, global_model)                                                    # Aggregation weight is normalized here
        
    elif args.alg == 'pfedgraph':
        local_train_pfedgraph(args, round, nets_this_round, cluster_models, train_local_dls, freeriders)
        mean_personalized_acc = evaluate_local_models(nets_this_round, test_dl, data_distributions, best_test_acc_list, benign_client_list)
        manipulate_gradient(args, nets_this_round, (freeriders, attackers))
        graph_matrix = update_graph_pfedgraph(graph_matrix, nets_this_round, global_parameters, fed_avg_freqs, args.pfedgraph_alpha)
        aggregation_pfedgraph(graph_matrix, nets_this_round, global_parameters, cluster_models)                            
    
    mean_utility, payment_martix = Cal_Utility(args, graph_matrix, nets_this_round, gain_list, data_num_list_real, data_num_list_reported, cost_list_real, model_difference_matrix, utility, benign_client_list)
    Cal_MC_Utility(utility[-1], best_test_acc_list, liars, freeriders, attackers) # 计算malicious clients的utility
    Dropout(args, graph_matrix, utility[-1], party_list, benign_client_list)
    total_payment_matrix += payment_martix
    total_graph_matrix += graph_matrix

    print('Graph:')
    print(graph_matrix)
    print('Payment')
    print(payment_martix)
    print('>> (Current) Round {} | Local Per: {:.5f} Best Per: {:.5f}, Avg Utility: {:.5f}'.format(round, mean_personalized_acc, np.array(best_test_acc_list)[benign_client_list].mean(), mean_utility))
    print('-'*80)
    
print('Graph:')
print(total_graph_matrix)
print('Payment')
print(total_payment_matrix)
best_test_acc_list_str = [str(format(i, '.5f')) for i in best_test_acc_list]
utility_str = [str(format(i, '.3f')) for i in np.sum(utility, axis=0)]
print('>> (Final) Personalized Acc: ', ", ".join(best_test_acc_list_str), '| Avg: {:.5f}'.format(sum(best_test_acc_list)/len(best_test_acc_list)))
print('>> (Final) Utility: ', ", ".join(utility_str), '| Avg: {:.5f}'.format(np.sum(utility)/args.n_parties))
print('>> Time: {:.2f} s'.format(time.time() - Start_time))
print('>> -------- End of {} Training --------'.format(args.alg))