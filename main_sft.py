import time
Start_time = time.time()
import copy
import os
from tqdm import tqdm
import numpy as np
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM
from peft import get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict, prepare_model_for_kbit_training, PeftModel

from config_llm import get_config, save_config, get_model_config, get_training_args
from utils.algorithms.incentive_utils import Cal_Utility, Init_incentive_properties
from utils.dataset.dataset_llm import get_dataset, process_sft_dataset, get_eval_dataset
from utils.utils import split_dataset, get_dataset_this_round, cosine_learning_rate
from utils.template import get_formatting_prompts_func
from utils.fed_local import get_fed_local_sft_trainer
from utils.fed_global import Update_graph_matrix, Model_aggregation
from utils.evaluation import cal_model_difference_llm

import warnings
warnings.filterwarnings("ignore", category=UserWarning) 
warnings.filterwarnings("ignore", category=FutureWarning)
from trl import SFTTrainer

# ===== Define the arguments =====
script_args, fed_args, peft_config = get_config()
training_args = get_training_args(script_args, script_args.learning_rate)
save_config(script_args, fed_args)
# ===== Set the Logger =====
output_subdir = script_args.output_dir+f'/{fed_args.alg}_{time.strftime("%Y%m%d%H%M%S", time.localtime())}'
os.makedirs(output_subdir, exist_ok=True)
logging.basicConfig(filename=f'{output_subdir}/training.log', 
        format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%m-%d %H:%M', level=logging.INFO, filemode='w')
logger = logging.getLogger()

# ===== Load the dataset =====
dataset = get_dataset(script_args.dataset_name, script_args.local_data_dir)
dataset = process_sft_dataset(script_args.dataset_name, dataset, script_args.dataset_sample, script_args.dataset_ratio)
eval_dataset = get_eval_dataset(script_args.dataset_name, script_args.local_data_dir, script_args.dataset_sample_eval)
print('>> Dataset: ', dataset)
logger.info("-"*30+"Dataset"+"-"*30)
logger.info(f"| Train: {dataset} | Eval: {eval_dataset} |")
logger.info("-"*70)

# ===== Split the dataset into clients =====
local_datasets = split_dataset(fed_args, script_args, dataset, prefix='Training')
local_eval_datasets = split_dataset(fed_args, script_args, eval_dataset, prefix='Evaluation') if fed_args.local_eval else [None]*fed_args.n_parties
sample_num_list = [len(local_datasets[i]) for i in range(fed_args.n_parties)]
print('>> Sample list: {}'.format(sample_num_list))
logger.info(">> Sample list: {}".format(sample_num_list))
# ===== Initial incentive properties =====
utility, gain_list, data_num_list_real, data_num_list_reported, cost_list_real, cost_list_reported, graph_matrix, malicious_clients = Init_incentive_properties(fed_args, sample_num_list)
(liars, freeriders, attackers) = malicious_clients
party_list = [i for i in range(fed_args.n_parties)]
benign_client_list = list(set(party_list) - set(freeriders) - set(attackers)) # calculate the acc

# ===== Get model config =====
device_map, quantization_config, torch_dtype = get_model_config(script_args)

model = AutoModelForCausalLM.from_pretrained(
    script_args.model_name,
    quantization_config=quantization_config,
    device_map=device_map, # auto
    trust_remote_code=script_args.trust_remote_code,
    torch_dtype=torch_dtype
)

if script_args.load_in_8bit or script_args.load_in_4bit:
    model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=training_args.gradient_checkpointing
            )

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# print(f">> Init: fixed param: {model.state_dict()['base_model.model.model.layers.0.self_attn.q_proj.weight'][0]}, learned: {model.state_dict()['base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight'][0]}")

# ===== Define the global and local models =====
global_dict = copy.deepcopy(get_peft_model_state_dict(model))
local_dict_list = [copy.deepcopy(global_dict) for i in range(fed_args.n_parties)]
cluster_indices = [np.arange(fed_args.n_parties).astype("int")] if fed_args.alg == 'cfl' else None
cluster_dict_list = [copy.deepcopy(global_dict) for i in range(fed_args.n_parties)] if fed_args.alg in ('fedamp', 'ipfl', 'pfedgraph') else [None]*fed_args.n_parties
prox_vectors = [None]*fed_args.n_parties
# ===== Define the tokenizer =====
tokenizer = AutoTokenizer.from_pretrained(script_args.model_name, use_fast=False, padding_side="right")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.unk_token   # following vicuna

# ===== Define the formatting function =====
formatting_prompts_func, response_template = get_formatting_prompts_func(script_args.template, tokenizer.eos_token)
response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)[2:]  # Now we have it like in the dataset texts: `[2277, 29937, 4007, 22137, 29901]`
data_collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

# ===== Start federated training =====
training_loss = [[] for i in range(fed_args.n_parties)]
best_loss_list = [1e2] * fed_args.n_parties
print('>> -------- Start {} Training --------'.format(fed_args.alg))
logger.info('>> -------- Start of {} Training --------'.format(fed_args.alg))
for round in tqdm(range(fed_args.num_rounds)):
    print(f">> ==================== Round {round+1} : {party_list} ====================")
    logger.info(f">> ==================== Round {round+1} : {party_list} ====================")
    for client in range(fed_args.n_parties):

        if client not in party_list or client in freeriders:
            training_loss[client].append(-1)        # -1 is an indicator of not training
            continue
        total_num_samples = sum(data_num_list_reported[k] for k in party_list)
        fed_avg_freqs = [data_num_list_reported[k] / total_num_samples for k in party_list]
        if fed_args.alg in ['fedavg', 'fedprox']:
            set_peft_model_state_dict(model, global_dict)   # sync the global model to the local model
        else:
            set_peft_model_state_dict(model, local_dict_list[client])  # sync the local model
        local_dict_start = {client: copy.deepcopy(local_dict_list[client]) for client in party_list}
        
        sub_dataset = get_dataset_this_round(local_datasets[client], round, fed_args, script_args)  # get the required sub-dataset for this round
        new_lr = cosine_learning_rate(round, fed_args.num_rounds, script_args.learning_rate, 1e-6) # manually schedule the learning rate
        training_args = get_training_args(script_args, new_lr)

        if fed_args.local_eval and ((round+1) % 5 == 0 or round == 1):
            sub_eval_dataset = local_eval_datasets[client].shuffle(seed=round).select(range(min(script_args.dataset_sample_eval, len(local_eval_datasets[client]))))
            trainer_eval = SFTTrainer(
                model=model,
                tokenizer=tokenizer,
                args=training_args,
                max_seq_length=script_args.seq_length,
                eval_dataset=sub_eval_dataset,
                formatting_func=formatting_prompts_func,
                data_collator=data_collator,
            )
            eval_loss = trainer_eval.evaluate()["eval_loss"]
            if eval_loss < best_loss_list[client]:
                best_loss_list[client] = eval_loss
                trainer_eval.save_model(os.path.join(output_subdir, f"client{client}-checkpoint"))

        trainer = get_fed_local_sft_trainer(
            model=model,
            round=round,
            tokenizer=tokenizer,
            training_args=training_args,
            local_dataset=sub_dataset,
            formatting_prompts_func=formatting_prompts_func,
            data_collator=data_collator,
            global_dict=global_dict,
            cluster_dict=cluster_dict_list[client],
            proxy_vec=prox_vectors[client],
            fed_args=fed_args,
            script_args=script_args,
        )

        results = trainer.train()
        training_loss[client].append(results.training_loss)
        
        print(f">> Client {client} Training Loss: {results.training_loss:.5f} | Eval Loss: {best_loss_list[client]:.5f}")
        logger.info(f">> Client {client} Training Loss: {results.training_loss:.5f} | Eval Loss: {best_loss_list[client]:.5f}")
        local_dict_list[client] = copy.deepcopy(get_peft_model_state_dict(model))   # copy is needed!

    # ===== Manipulate the gradients =====
    nets_this_round = {client: local_dict_list[client] for client in party_list}
    
    # ===== Updating the graph =====
    model_difference_matrix = cal_model_difference_llm(fed_args.n_parties, nets_this_round, global_dict)
    graph_matrix, cluster_indices = Update_graph_matrix(fed_args, graph_matrix, nets_this_round, global_dict, local_dict_start, cluster_indices, fed_avg_freqs, data_num_list_reported, cost_list_reported, gain_list, model_difference_matrix)
    # ===== Aggregate the local models =====
    prox_vectors = Model_aggregation(fed_args, graph_matrix, nets_this_round, fed_avg_freqs, global_dict, cluster_dict_list)
    # set_peft_model_state_dict(model, global_dict)   # update global model

    # ===== Calculate the utility =====
    mean_utility, payment_martix = Cal_Utility(fed_args, graph_matrix, nets_this_round, gain_list, data_num_list_real, data_num_list_reported, cost_list_real, model_difference_matrix, utility, benign_client_list)
    print('>> (Current) Round {} Loss: {:.5f} | Utility: {:.2f}'.format(round+1, np.mean(best_loss_list), mean_utility))
    logger.info('>> (Current) Round {} Loss: {:.5f} | Utility: {:.2f}'.format(round+1, np.mean(best_loss_list), mean_utility))
    logger.info('Graph:')
    logger.info(graph_matrix)
    logger.info('Payment:')
    logger.info(payment_martix)

print('Graph:')
print(graph_matrix) 
logger.info('Graph:')
logger.info(graph_matrix)
logger.info('Payment:')
logger.info(payment_martix)
best_test_acc_list_str = [str(format(i, '.3f')) for i in best_loss_list]
utility_str = [str(format(i, '.3f')) for i in np.sum(utility, axis=0)]
print('>> (Final) Personalized Loss: ', ", ".join(best_test_acc_list_str), '| Avg: {:.5f}'.format(np.mean(best_loss_list)))
print('>> (Final) Utility: ', ", ".join(utility_str), '| Avg: {:.5f}'.format(np.sum(utility)/fed_args.n_parties))
print('>> Time: {:.2f} s'.format(time.time() - Start_time))
print('>> -------- End of {} Training --------'.format(fed_args.alg))
logger.info('>> (Final) Personalized Loss: ' + ", ".join(best_test_acc_list_str) + '| Avg: {:.5f}'.format(np.mean(best_loss_list)))
logger.info('>> (Final) Utility: ' + ", ".join(utility_str), '| Avg: {:.5f}'.format(np.sum(utility)/fed_args.n_parties))
logger.info('>> Time: {:.2f} s'.format(time.time() - Start_time))
logger.info('>> -------- End of {} Training --------'.format(fed_args.alg))
# remove the cache file of datasets
for ds in local_datasets:
    ds.cleanup_cache_files()
for ds in local_eval_datasets:
    ds.cleanup_cache_files()


