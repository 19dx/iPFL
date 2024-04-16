import datasets
datasets.disable_progress_bar()   # disable the progress bar
from datasets import load_dataset
import pandas as pd
from ..template import MODEL2TEMPLATE

def get_dataset(dataset_name, local_data_dir):
    if dataset_name == 'CoFinance': # alpaca + NWGI
        code_dataset = load_dataset(local_data_dir+"/CodeAlpaca-20k", split="train")
        fin_dataset = load_dataset("json", data_files=local_data_dir+"/NWGI/trainNWGI_12138.jsonl", split="train") 
        dataset = (code_dataset, fin_dataset)
    elif dataset_name == 'finance': # FIQA, TFNS, NWGI
        fiqa_dataset = load_dataset('json', data_files=local_data_dir+'/FIQA/trainFIQA_938.jsonl', split="train")
        tfns_dataset = load_dataset('json', data_files=local_data_dir+'/TFNS/trainTFNS_14772.jsonl', split="train")
        nwgi_dataset = load_dataset('json', data_files=local_data_dir+'/NWGI/trainNWGI_12138.jsonl', split="train")
        dataset = (fiqa_dataset, tfns_dataset, nwgi_dataset)
    elif dataset_name == 'gpt4': # vicgalle/alpaca-gpt4
        dataset_name = local_data_dir + "vicgalle/alpaca-gpt4" if local_data_dir is not None else "vicgalle/alpaca-gpt4"
        dataset = load_dataset(dataset_name, split="train")
    elif dataset_name == 'dromedary':  # 'zhiqings/dromedary-65b-verbose-clone-v0'
        dataset_name = local_data_dir + 'zhiqings/dromedary-65b-verbose-clone-v0' if local_data_dir is not None else 'zhiqings/dromedary-65b-verbose-clone-v0'
        data_files = dataset_name + '/' + 'merged_behavior_clone.json'
        dataset = load_dataset('json', data_files=data_files, split='train')
    else:
        dataset_name = local_data_dir + dataset_name if local_data_dir is not None else dataset_name
        dataset = load_dataset(dataset_name, split="train")
    return dataset

def get_eval_dataset(dataset_name, local_data_dir, data_sample_eval):
    if dataset_name == 'finance': # FIQA, TFNS, NWGI
        fiqa_dataset = load_dataset('json', data_files=local_data_dir+'/FIQA/testFIQA_275.jsonl', split="train")
        fiqa_dataset = fiqa_dataset.map(alpaca_format, remove_columns=['input', 'output'], desc=f"Preprocessing {dataset_name} for unified format.")
        tfns_dataset = load_dataset('json', data_files=local_data_dir+'/TFNS/testTFNS_4314.jsonl', split="train")
        tfns_dataset = tfns_dataset.map(alpaca_format, remove_columns=['input', 'output'], desc=f"Preprocessing {dataset_name} for unified format.")
        nwgi_dataset = load_dataset('json', data_files=local_data_dir+'/NWGI/testNWGI_4046.jsonl', split="train")
        nwgi_dataset = nwgi_dataset.map(alpaca_format, remove_columns=['input', 'output'], desc=f"Preprocessing {dataset_name} for unified format.")
        print(f'EVALUATION: FIQA: {fiqa_dataset}, TFNS: {tfns_dataset}, NWGI: {nwgi_dataset}')
        dataset = (fiqa_dataset, tfns_dataset, nwgi_dataset)
    elif dataset_name == 'gpt4': # vicgalle/alpaca-gpt4
        dataset = load_dataset(local_data_dir + "vicgalle/alpaca-gpt4", split="train")
        dataset = dataset.map(alpaca_format, remove_columns=['input', 'output', 'text'], desc=f"Preprocessing {dataset_name} for unified format.")
        dataset = dataset.shuffle(seed=2023).select(range(len(dataset)-data_sample_eval, len(dataset)))
        print(f'EVALUATION: Alpaca-GPT4: {dataset}')
    elif dataset_name == 'dromedary':  # 'zhiqings/dromedary-65b-verbose-clone-v0'
        dataset_name = local_data_dir + 'zhiqings/dromedary-65b-verbose-clone-v0' if local_data_dir is not None else 'zhiqings/dromedary-65b-verbose-clone-v0'
        data_files = dataset_name + '/' + 'merged_behavior_clone.json'
        dataset = load_dataset('json', data_files=data_files, split='train')
        dataset = dataset.map(alpaca_format, remove_columns=['input', 'output'], desc=f"Preprocessing {dataset_name} for unified format.")
        dataset = dataset.shuffle(seed=2023).select(range(len(dataset)-data_sample_eval, len(dataset)))
        print(f'EVALUATION: Dromedary: {dataset}')
    else:
        dataset = None
    return dataset

def process_sft_dataset(dataset_name, dataset, dataset_sample, ratio):
    if dataset_name == 'CoFinance': # alpaca + NWGI
        code_dataset, fin_dataset = dataset
        code_dataset = code_dataset.shuffle(seed=2023).select(range(dataset_sample))
        fin_dataset = fin_dataset.shuffle(seed=1).select(range(600))
        code_dataset = code_dataset.map(alpaca_format, remove_columns=['output'], desc=f"Preprocessing {dataset_name} for unified format.")
        fin_dataset = fin_dataset.map(alpaca_format, remove_columns=['input', 'output'], desc=f"Preprocessing {dataset_name} for unified format.")
        print(f"Code: {code_dataset}, Fin: {fin_dataset}")
        dataset = (code_dataset, fin_dataset)
        return dataset
    elif dataset_name == 'finance': # FIQA, TFNS, NWGI
        fiqa_dataset, tfns_dataset, nwgi_dataset = dataset
        num_sample = min(len(fiqa_dataset), dataset_sample)
        fiqa_dataset = fiqa_dataset.shuffle(seed=2023).select(range(num_sample))
        fiqa_dataset = fiqa_dataset.map(alpaca_format, remove_columns=['input', 'output'], desc=f"Preprocessing {dataset_name} for unified format.")
        num_sample = min(len(tfns_dataset), dataset_sample)
        tfns_dataset = tfns_dataset.shuffle(seed=2023).select(range(num_sample))
        tfns_dataset = tfns_dataset.map(alpaca_format, remove_columns=['input', 'output'], desc=f"Preprocessing {dataset_name} for unified format.")
        num_sample = min(len(nwgi_dataset), dataset_sample)
        nwgi_dataset = nwgi_dataset.shuffle(seed=2023).select(range(num_sample))
        nwgi_dataset = nwgi_dataset.map(alpaca_format, remove_columns=['input', 'output'], desc=f"Preprocessing {dataset_name} for unified format.")
        print(f"FIQA: {fiqa_dataset}, TFNS: {tfns_dataset}, NWGI: {nwgi_dataset}")
        dataset = (fiqa_dataset, tfns_dataset, nwgi_dataset)
        return dataset
    elif dataset_name == 'gpt4': # vicgalle/alpaca-gpt4
        dataset = dataset.map(alpaca_format, remove_columns=['input', 'output', 'text'], desc=f"Preprocessing {dataset_name} for unified format.")
        dataset = dataset.shuffle(seed=2023).select(range(dataset_sample))
        return dataset
    elif dataset_name == 'dromedary':  # 'zhiqings/dromedary-65b-verbose-clone-v0'
        def dromedary_format(example):
            if example['input'] == "":
                example["instruction"] = example["instruction"]
            else:
                example["instruction"] = example["instruction"] + " " + example['input']
            example["response"] = example["output"].replace("\n\n### User", "")
            return example   

        dataset = dataset.map(dromedary_format, remove_columns=['input', 'output'], desc=f"Preprocessing {dataset_name} for unified format.")
        dataset = dataset.shuffle(seed=2023).select(range(dataset_sample))
        return dataset
    else:
        raise NotImplementedError(f"Dataset {dataset_name} is not supported.")


def alpaca_format(example):
    if 'input' not in example.keys() or example['input'] == "":
        example["instruction"] = example["instruction"]
    else:
        example["instruction"] = example["instruction"] + " " + example['input']
    example["response"] = example['output']
    return example


