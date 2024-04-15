# ============ For Instruction-tuning tasks ===========
import torch
import copy
from trl import SFTTrainer


def get_fed_local_sft_trainer(script_args, fed_args, model, tokenizer, training_args, local_dataset, formatting_prompts_func, data_collator, global_dict, cluster_dict, proxy_vec, round):
    
    if fed_args.alg == 'fedprox':
        trainer = SFTTrainerFedProx(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            max_seq_length=script_args.seq_length,
            train_dataset=local_dataset,
            formatting_func=formatting_prompts_func,
            data_collator=data_collator,
            global_state=global_dict,
            prox_mu=fed_args.prox_mu,
        )
    elif fed_args.alg in ['fedavg', 'cfl'] or (fed_args.alg).startswith('local'):
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            max_seq_length=script_args.seq_length,
            train_dataset=local_dataset,
            formatting_func=formatting_prompts_func,
            data_collator=data_collator,
        )
    elif fed_args.alg == 'fedamp':
        trainer = SFTTrainerFedAMP(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            max_seq_length=script_args.seq_length,
            train_dataset=local_dataset,
            formatting_func=formatting_prompts_func,
            data_collator=data_collator,
            round=round,
            cluster_dict=cluster_dict,
        )
    elif fed_args.alg == 'ipfl':
        trainer = SFTTraineriPFL(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            max_seq_length=script_args.seq_length,
            train_dataset=local_dataset,
            formatting_func=formatting_prompts_func,
            data_collator=data_collator,
            round=round,
            proxy_vec=proxy_vec,
            lamda=fed_args.ipfl_lam,
        )
    elif fed_args.alg == 'pfedgraph':
        trainer = SFTTrainerpFedGraph(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            max_seq_length=script_args.seq_length,
            train_dataset=local_dataset,
            formatting_func=formatting_prompts_func,
            data_collator=data_collator,
            round=round,
            cluster_dict=cluster_dict,
        )

    return trainer

class SFTTrainerFedProx(SFTTrainer):
    def __init__(self, global_state, prox_mu, **kwargs):
        super(SFTTrainerFedProx, self).__init__(**kwargs)
        self.global_state = global_state
        self.mu = prox_mu
    
    def compute_loss(self, model, inputs, return_outputs=False):

        return_values = super(SFTTrainerFedProx, self).compute_loss(model, inputs, return_outputs=return_outputs)

        if return_outputs:
            loss, outputs = return_values
        else:
            loss = return_values

        # Apply FedProx Loss
        for name, param in model.named_parameters():
            name = name.replace(".default", "")   
            # only trainable parameters
            if not param.requires_grad:
                continue
            else:
                loss += self.mu / 2 * torch.norm(param - self.global_state[name]) ** 2

        return (loss, outputs) if return_outputs else loss

class SFTTrainerFedAMP(SFTTrainer):
    def __init__(self, round, cluster_dict, **kwargs):
        super(SFTTrainerFedAMP, self).__init__(**kwargs)
        self.cluster_dict = cluster_dict
        self.round = round
    
    def compute_loss(self, model, inputs, return_outputs=False):

        return_values = super(SFTTrainerFedAMP, self).compute_loss(model, inputs, return_outputs=return_outputs)

        if return_outputs:
            loss, outputs = return_values
        else:
            loss = return_values

        if self.round > 0:
            # Apply Norm Loss
            for name, param in model.named_parameters():
                name = name.replace(".default", "") 
                # only trainable parameters
                if not param.requires_grad:
                    continue
                else:
                    loss += 0.005 * torch.norm(param - self.cluster_dict[name]) ** 2
        return (loss, outputs) if return_outputs else loss


class SFTTrainerpFedGraph(SFTTrainer):
    def __init__(self, round, cluster_dict, **kwargs):
        super(SFTTrainerpFedGraph, self).__init__(**kwargs)
        self.cluster_dict = cluster_dict
        self.round = round
    
    def compute_loss(self, model, inputs, return_outputs=False):

        return_values = super(SFTTrainerpFedGraph, self).compute_loss(model, inputs, return_outputs=return_outputs)

        if return_outputs:
            loss, outputs = return_values
        else:
            loss = return_values

        if self.round > 0:
            for name, param in model.named_parameters():
                name = name.replace(".default", "")  
                # only trainable parameters
                if not param.requires_grad:
                    continue
                else:
                    loss += 0.01 * torch.nn.functional.cosine_similarity(param.view(-1), self.cluster_dict[name].view(-1), dim=0, eps=1e-8)
        return (loss, outputs) if return_outputs else loss


class SFTTraineriPFL(SFTTrainer):
    def __init__(self, round, proxy_vec, lamda, **kwargs):
        super(SFTTraineriPFL, self).__init__(**kwargs)
        self.round = round
        self.proxy_vec = proxy_vec
        self.lamda = lamda
    
    def compute_loss(self, model, inputs, return_outputs=False):

        return_values = super(SFTTraineriPFL, self).compute_loss(model, inputs, return_outputs=return_outputs)

        if return_outputs:
            loss, outputs = return_values
        else:
            loss = return_values

        if self.round > 0:
            net_param = []
            for param in model.parameters():
                if param.requires_grad:
                    net_param.append(param.reshape(-1))
            net_param = torch.cat(net_param)
            loss += self.lamda * torch.norm(net_param - self.proxy_vec) ** 2
        return (loss, outputs) if return_outputs else loss
    
    