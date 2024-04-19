max_steps=10
num_rounds=50
batch_size=8
gradient_accumulation_steps=1
seq_length=512
data_sample=400
data_sample_eval=300
local_eval=true
n_parties=6
lora_r=32
lora_alpha=64
local_data_dir="data/"
model_name="MODEL_PATH" 
dataset_name="finance" 
split_strategy='cluster'
lr=5e-5
K=2e4
output_dir=./output/mix-finance
if [ ! -d $output_dir ]; then
    mkdir -p $output_dir
else
    echo "Output dir $output_dir exists, skipping..."
fi
lam=5
eta=1
CUDA_VISIBLE_DEVICES=0 nohup python main_sft.py --alg "local" --K $K --local_eval $local_eval --dataset_sample $data_sample --dataset_sample_eval $data_sample_eval --peft_lora_r $lora_r --peft_lora_alpha $lora_alpha --learning_rate $lr --model_name $model_name --local_data_dir $local_data_dir --dataset_name $dataset_name --split_strategy $split_strategy --n_parties $n_parties --max_steps $max_steps --num_rounds $num_rounds --batch_size $batch_size --gradient_accumulation_steps $gradient_accumulation_steps --seq_length $seq_length --use_peft --load_in_8bit --output_dir $output_dir >$output_dir/local.log &
CUDA_VISIBLE_DEVICES=6 nohup python main_sft.py --alg "fedavg" --K $K --local_eval $local_eval --dataset_sample $data_sample --dataset_sample_eval $data_sample_eval --peft_lora_r $lora_r --peft_lora_alpha $lora_alpha --learning_rate $lr --model_name $model_name --local_data_dir $local_data_dir --dataset_name $dataset_name --split_strategy $split_strategy --n_parties $n_parties --max_steps $max_steps --num_rounds $num_rounds --batch_size $batch_size --gradient_accumulation_steps $gradient_accumulation_steps --seq_length $seq_length --use_peft --load_in_8bit --output_dir $output_dir >$output_dir/fedavg.log &
CUDA_VISIBLE_DEVICES=7 nohup python main_sft.py --alg "fedprox" --K $K --local_eval $local_eval --dataset_sample $data_sample --dataset_sample_eval $data_sample_eval --peft_lora_r $lora_r --peft_lora_alpha $lora_alpha --learning_rate $lr --model_name $model_name --local_data_dir $local_data_dir --dataset_name $dataset_name --split_strategy $split_strategy --n_parties $n_parties --max_steps $max_steps --num_rounds $num_rounds --batch_size $batch_size --gradient_accumulation_steps $gradient_accumulation_steps --seq_length $seq_length --use_peft --load_in_8bit --output_dir $output_dir >$output_dir/fedprox.log &
CUDA_VISIBLE_DEVICES=0 nohup python main_sft.py --alg "cfl" --K $K --local_eval $local_eval --dataset_sample $data_sample --dataset_sample_eval $data_sample_eval --peft_lora_r $lora_r --peft_lora_alpha $lora_alpha --learning_rate $lr --model_name $model_name --local_data_dir $local_data_dir --dataset_name $dataset_name --split_strategy $split_strategy --n_parties $n_parties --max_steps $max_steps --num_rounds $num_rounds --batch_size $batch_size --gradient_accumulation_steps $gradient_accumulation_steps --seq_length $seq_length --use_peft --load_in_8bit --output_dir $output_dir >$output_dir/cfl.log &
CUDA_VISIBLE_DEVICES=4 nohup python main_sft.py --alg "fedamp" --K $K --local_eval $local_eval --dataset_sample $data_sample --dataset_sample_eval $data_sample_eval --peft_lora_r $lora_r --peft_lora_alpha $lora_alpha --learning_rate $lr --model_name $model_name --local_data_dir $local_data_dir --dataset_name $dataset_name --split_strategy $split_strategy --n_parties $n_parties --max_steps $max_steps --num_rounds $num_rounds --batch_size $batch_size --gradient_accumulation_steps $gradient_accumulation_steps --seq_length $seq_length --use_peft --load_in_8bit --output_dir $output_dir >$output_dir/fedamp.log &
CUDA_VISIBLE_DEVICES=5 nohup python main_sft.py --alg "pfedgraph" --K $K --local_eval $local_eval --dataset_sample $data_sample --dataset_sample_eval $data_sample_eval --peft_lora_r $lora_r --peft_lora_alpha $lora_alpha --learning_rate $lr --model_name $model_name --local_data_dir $local_data_dir --dataset_name $dataset_name --split_strategy $split_strategy --n_parties $n_parties --max_steps $max_steps --num_rounds $num_rounds --batch_size $batch_size --gradient_accumulation_steps $gradient_accumulation_steps --seq_length $seq_length --use_peft --load_in_8bit --output_dir $output_dir >$output_dir/pfedgraph.log &
CUDA_VISIBLE_DEVICES=4 nohup python main_sft.py --alg "ipfl" --ipfl_lam $lam --ipfl_eta $eta --K $K --local_eval $local_eval --dataset_sample $data_sample --dataset_sample_eval $data_sample_eval --peft_lora_r $lora_r --peft_lora_alpha $lora_alpha --learning_rate $lr --model_name $model_name --local_data_dir $local_data_dir --dataset_name $dataset_name --split_strategy $split_strategy --n_parties $n_parties --max_steps $max_steps --num_rounds $num_rounds --batch_size $batch_size --gradient_accumulation_steps $gradient_accumulation_steps --seq_length $seq_length --use_peft --load_in_8bit --output_dir $output_dir >$output_dir/ipfl.log &

