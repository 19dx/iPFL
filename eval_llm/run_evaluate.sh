
split_strategy='cluster'
eval_sample_size=600
clients='0,1,2,3,4,5'
for adapter_model_path in "output_finance/finance_llama2_c6_s400_i10_b8a1_l512/local_20231222152133"
do
CUDA_VISIBLE_DEVICES=3 python eval_llm/multi-choice_per_eval.py  --adapter_model_path $adapter_model_path \
    --split_strategy $split_strategy --eval_sample_size $eval_sample_size --clients_selection $clients
done