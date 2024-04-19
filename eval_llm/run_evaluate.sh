model_root='BASE-MODEL-DIR'
split_strategy='cluster'
eval_sample_size=600
clients='0,1,2,3,4,5'
for adapter_model_path in "CHECKPOINT-DIR"
do
CUDA_VISIBLE_DEVICES=0 python eval_llm/multi-choice_per_eval.py  --adapter_model_path $adapter_model_path \
    --seed 5 --model_root $model_root --split_strategy $split_strategy --eval_sample_size $eval_sample_size --clients_selection $clients
done