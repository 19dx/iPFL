import argparse
import os
import torch
import numpy as np
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
from tqdm import tqdm
import sys 
from ..utils.dataset.dataset_llm import get_eval_dataset, split_dataset

TEMPLATE = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{}

### Response:"""


@torch.no_grad()
def eval(model, tokenizer, test_dataset, device='cuda'):
    cors = 0
    with torch.no_grad():
        for sample in tqdm(test_dataset):
            input_ids = tokenizer.encode(TEMPLATE.format(sample['instruction']), return_tensors='pt').to(device)
            outputs = model.generate(inputs=input_ids, max_length=2048)  # do_sample=True, top_p=1.0, temperature=0.7
            outputs = outputs[0][len(input_ids[0]):]
            decoded_outputs = tokenizer.decode(outputs, skip_special_tokens=True)
            while decoded_outputs.startswith(' '):
                decoded_outputs = decoded_outputs[1:]
            cors += int(decoded_outputs == sample['response'])
    return cors/len(test_dataset)

def get_options(data):
    if "option" in data:
        return list(data['option'].values())
    else:
        options = []
        for key in ['opa', 'opb', 'opc', 'opd']:
            options.append(data[key])
        return options


def main(args):
    # ====== Dataset ======
    eval_dataset = get_eval_dataset(args.dataset_name, ratio=1)
    local_eval_datasets = split_dataset(args, args, eval_dataset, prefix='Evaluation')
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name)
    # ====== Model Config ======
    device = 'cuda'
    
    all_cors = []
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model_name, torch_dtype=torch.bfloat16).to(device)
    if args.clients_selection:
        client_list = [int(c) for c in args.clients_selection.split(',')]
    else:
        client_list = list(range(args.n_parties))
    for client in client_list:
        Start_time = time.time()
        if args.adapter_model_path:
            if len(client_list) == 3:
                adapter_model_name = os.path.join(args.adapter_model_path, "client{}-r50-checkpoint".format(client))
            else:
                adapter_model_name = os.path.join(args.adapter_model_path, "client{}-checkpoint".format(client))
            model = PeftModel.from_pretrained(base_model, adapter_model_name)
        else:
            model = base_model
        eval_dataset = local_eval_datasets[client].shuffle(seed=27).select(range(min(args.eval_sample_size, len(local_eval_datasets[client]))))
        cors = eval(model, tokenizer, eval_dataset, device=device)
        print("Client {} accuracy: {:.5f} | Time: {:.3f}".format(client, cors, time.time()-Start_time))
        all_cors.append(cors)
    if args.split_strategy == 'cluster' and not args.clients_selection:
        for c in range(args.n_parties//2):
            print("Cluster {} accuracy: {:.5f}".format(c, np.mean(all_cors[c*2:c*2+2])))

    weighted_acc = np.mean(all_cors)
    print(args.adapter_model_path)
    print("Average accuracy: {:.5f}".format(weighted_acc))
    print('='*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_name", "-bm", type=str, default="/GPFS/public/6fdf2e60f86ff2481f2241aaee459f85b5b0bbb9")
    parser.add_argument("--adapter_model_path", "-am", type=str, default="")
    parser.add_argument("--dataset_name", "-d", type=str, default="medqa")
    parser.add_argument("--n_parties", "-n", type=int, default=6)
    parser.add_argument("--seed", "-s", type=int, default=2023)
    parser.add_argument("--split_strategy", "-ss", type=str, default="cluster")
    parser.add_argument("--eval_sample_size", "-ess", type=int, default=800)
    parser.add_argument("--clients_selection", "-cs", type=str, default=None)

    args = parser.parse_args()
    model_root = 'Your_model_path'
    model_name = args.adapter_model_path.split('/')[-2].split('_')[1]
    args.base_model_name = os.path.join(model_root, model_name)
    args.dataset_name = args.adapter_model_path.split('/')[-2].split('_')[0]
    args.n_parties = int(args.adapter_model_path.split('/')[-2].split('_')[2].split('c')[1])
    print(args)
    main(args)