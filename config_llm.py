from dataclasses import dataclass, field, asdict
from typing import Optional
from transformers import HfArgumentParser, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig
import os
import json
from accelerate import Accelerator
import torch
from datetime import datetime  
import yaml


# Define and parse arguments.
@dataclass
class FedArguments:
    alg: Optional[str] = field(default="fedavg", metadata={"help": "the algorithm to use"})
    num_rounds: Optional[int] = field(default=500, metadata={"help": "the number of rounds"})
    n_parties: Optional[int] = field(default=2, metadata={"help": "the number of clients"})
    sample_clients: Optional[int] = field(default=2, metadata={"help": "the number of clients to sample"})
    split_strategy: Optional[str] = field(default="iid", metadata={"help": "the split strategy"})
    split_strategy: Optional[str] = field(default="iid", metadata={"help": "the split strategy"})
    prox_mu: Optional[float] = field(default=0.01, metadata={"help": "the mu parameter of FedProx"})
    fed_momentum: Optional[float] = field(default=0.5, metadata={"help": "the momentum parameter at the server"})
    fedopt_tau: Optional[float] = field(default=1e-3, metadata={"help": "the tau parameter of FedAdagrad, FedYogi and FedAdam"})
    fedopt_eta: Optional[float] = field(default=1e-2, metadata={"help": "the global learning rate parameter of FedAdagrad, FedYogi and FedAdam"})
    fedopt_beta1: Optional[float] = field(default=0.9, metadata={"help": "the beta1 parameter of FedYogi and FedAdam"})
    fedopt_beta2: Optional[float] = field(default=0.99, metadata={"help": "the beta2 parameter of FedYogi and FedAdam"})
    local_eval: Optional[bool] = field(default=False, metadata={"help": "whether to evaluate locally"})
    # ==== incentive related ====
    C: Optional[float] = field(default=1, metadata={"help": "the cost for sharing its model for wach agent"})
    K: Optional[float] = field(default=5e5, metadata={"help": "Hyper-parameter in the utility function"})
    diverse: Optional[bool] = field(default=False, metadata={"help": "whether to use diverse utility"})
    ipfl_lam: Optional[float] = field(default=5, metadata={"help": "the lambda parameter of ipfl"})
    ipfl_eta: Optional[float] = field(default=1, metadata={"help": "the eta parameter of ipfl"})
    # attack
    attack_type: Optional[str] = field(default="inv_grad", metadata={"help": "the attack type"})
    attack_ratio: Optional[float] = field(default=0.0, metadata={"help": "the attack ratio"})
    # free-riding
    freeride_type: Optional[str] = field(default=None, metadata={"help": "the freeride type"})
    freerider_ratio: Optional[float] = field(default=0.0, metadata={"help": "ratio to sample for freeriders"})
    # liar
    liar_ratio: Optional[float] = field(default=0.0, metadata={"help": "the liar ratio"})
    liar_type: Optional[str] = field(default="size", metadata={"help": "the liar type"})
    liar_exaggerate_ratio: Optional[float] = field(default=2.0, metadata={"help": "the liar exaggerate ratio"})
    # dropout
    dropout: Optional[bool] = field(default=False, metadata={"help": "whether to choose dropout if negative utility occurs"})
    dropout_type: Optional[str] = field(default='IR-strict', metadata={"help": "the reason for dropping out"})
    dropout_p: Optional[float] = field(default=0.5, metadata={"help": "Dropout probability. Default=0.0"})
    

@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with SFTTrainer
    """

    model_name: Optional[str] = field(default="meta-llama/Llama-2-7b-hf", metadata={"help": "the model name"})
    dataset_name: Optional[str] = field(
        default="finance", metadata={"help": "the dataset name"}
    )
    dataset_text_field: Optional[str] = field(default="text", metadata={"help": "the text field of the dataset"})
    log_with: Optional[str] = field(default="none", metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=2e-5, metadata={"help": "the learning rate"})    # vicuna and alpaca use 2e-5
    batch_size: Optional[int] = field(default=16, metadata={"help": "the batch size"})
    batch_size_eval: Optional[int] = field(default=8, metadata={"help": "the batch size"})
    seq_length: Optional[int] = field(default=512, metadata={"help": "Input sequence length"})
    gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={"help": "the number of gradient accumulation steps"}
    )
    load_in_8bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 8 bits precision"})
    load_in_4bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 4 bits precision"})
    use_peft: Optional[bool] = field(default=False, metadata={"help": "Wether to use PEFT or not to train adapters"})
    trust_remote_code: Optional[bool] = field(default=False, metadata={"help": "Enable `trust_remote_code`"})
    output_dir: Optional[str] = field(default="output", metadata={"help": "the output directory"})
    peft_lora_r: Optional[int] = field(default=8, metadata={"help": "the r parameter of the LoRA adapters"})
    peft_lora_alpha: Optional[int] = field(default=16, metadata={"help": "the alpha parameter of the LoRA adapters"})
    logging_steps: Optional[int] = field(default=100, metadata={"help": "the number of logging steps"})
    use_auth_token: Optional[bool] = field(default=False, metadata={"help": "Use HF auth token to access the model"})   # token and use_auth_token cannot be used together
    num_train_epochs: Optional[int] = field(default=3, metadata={"help": "the number of training epochs"})
    max_steps: Optional[int] = field(default=10, metadata={"help": "the number of training steps"})
    save_steps: Optional[int] = field(
        default=1000, metadata={"help": "Number of updates steps before two checkpoint saves"}
    )
    save_total_limit: Optional[int] = field(default=10, metadata={"help": "Limits total number of checkpoints."})
    push_to_hub: Optional[bool] = field(default=False, metadata={"help": "Push the model to HF Hub"})
    hub_model_id: Optional[str] = field(default=None, metadata={"help": "The name of the model on HF Hub"})
    gradient_checkpointing: Optional[bool] = field(default=True, metadata={"help": "Enable gradient checkpointing"})
    template: Optional[str] = field(default="alpaca", metadata={"help": "the template to use"})
    seed: Optional[int] = field(default=2023, metadata={"help": "the seed to use"})
    dpo_beta: Optional[float] = field(default=0.1, metadata={"help": "the beta parameter of DPO"})
    dataset_sample: Optional[int] = field(default=20000, metadata={"help": "the number of samples to use from the dataset"})
    dataset_sample_eval: Optional[int] = field(default=300, metadata={"help": "the number of samples to use from the dataset"})
    dataset_ratio: Optional[int] = field(default=1, metadata={"help": "the ratio of the dataset to use"})
    local_data_dir: Optional[str] = field(default=None, metadata={"help": "the local data directory"})

parser = HfArgumentParser((ScriptArguments, FedArguments))
script_args, fed_args = parser.parse_args_into_dataclasses()

# ===== Define the LoraConfig =====
if script_args.use_peft:
    peft_config = LoraConfig(
        r=script_args.peft_lora_r,
        lora_alpha=script_args.peft_lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
else:
    peft_config = None

def get_config():
    return script_args, fed_args, peft_config

# ===== Define the training arguments =====
def get_training_args(script_args, new_lr):
    training_args = TrainingArguments(
        disable_tqdm=True,     # disable tqdm
        output_dir=script_args.output_dir,
        per_device_train_batch_size=script_args.batch_size,
        per_device_eval_batch_size=script_args.batch_size_eval,
        # eval_accumulation_steps=4,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        learning_rate=new_lr,
        logging_steps=script_args.logging_steps,
        num_train_epochs=script_args.num_train_epochs,
        max_steps=script_args.max_steps,
        report_to=script_args.log_with,
        save_steps=script_args.save_steps,
        save_total_limit=script_args.save_total_limit,
        push_to_hub=script_args.push_to_hub,
        hub_model_id=script_args.hub_model_id,
        gradient_checkpointing=script_args.gradient_checkpointing,
        lr_scheduler_type="constant",
    )
    return training_args

def get_model_config(script_args):
    if script_args.load_in_8bit and script_args.load_in_4bit:
        raise ValueError("You can't load the model in 8 bits and 4 bits at the same time")
    elif script_args.load_in_8bit or script_args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=script_args.load_in_8bit, load_in_4bit=script_args.load_in_4bit
        )
        # Copy the model to each device
        device_map = {"": Accelerator().local_process_index}
        torch_dtype = torch.bfloat16
    else:
        device_map = None
        quantization_config = None
        torch_dtype = None
    return device_map, quantization_config, torch_dtype

def save_config(script_args, fed_args):
    dataset_name_split = os.path.basename(script_args.dataset_name)
    model_name_split = os.path.basename(script_args.model_name)
    script_args.output_dir = f"{script_args.output_dir}/{dataset_name_split}_{model_name_split}_c{fed_args.n_parties}_s{script_args.dataset_sample}_i{script_args.max_steps}_b{script_args.batch_size}a{script_args.gradient_accumulation_steps}_l{script_args.seq_length}"
    os.makedirs(script_args.output_dir, exist_ok=True)
    with open(os.path.join(script_args.output_dir, f"{fed_args.alg}_args.json"), "w") as f:
        combined_dict = {
            "script_args": asdict(script_args),
            "fed_args": asdict(fed_args),
        }
        json.dump(combined_dict, f, indent=4)