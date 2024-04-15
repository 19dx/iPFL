START_TIME=`date +%s`
dataset='fashionmnist'
data_partition="noniid"
beta=0.1
client=10
model='simplecnn-mnist'
iternum=200
C=1
sample_fraction=1
lam=0.001
eta=5
dir_path=./output/${dataset}_${data_partition}_beta${beta}_${model}_it${iternum}_c${client}_p${sample_fraction}_C${C}
if [ ! -d $dir_path ]; then
    mkdir $dir_path
else
    echo "dir exists"
fi
dir_dir=${dir_path}/baselines
if [ ! -d $dir_dir ]; then
    mkdir $dir_dir
else
    echo "dir exists"
fi

# Baselines
nohup python -u main.py --alg "local" --gpu "0" --C $C --dataset $dataset --model $model --partition $data_partition --n_parties $client --num_local_iterations $iternum --beta $beta > $dir_dir/local_${START_TIME}.log &
nohup python -u main.py --alg "fedavg" --gpu "1" --C $C --dataset $dataset --model $model --partition $data_partition --n_parties $client --num_local_iterations $iternum --beta $beta > $dir_dir/fedavg_${START_TIME}.log &
nohup python -u main.py --alg "fedprox" --gpu "1" --C $C --dataset $dataset --model $model --partition $data_partition --n_parties $client --num_local_iterations $iternum --beta $beta > $dir_dir/fedprox_${START_TIME}.log &
nohup python -u main.py --alg "cfl" --gpu "5" --C $C --dataset $dataset --model $model --partition $data_partition --n_parties $client --num_local_iterations $iternum --beta $beta > $dir_dir/cfl_${START_TIME}.log &
nohup python -u main.py --alg "fedamp" --gpu "5" --C $C --dataset $dataset --model $model --partition $data_partition --n_parties $client --num_local_iterations $iternum --beta $beta > $dir_dir/fedamp_${START_TIME}.log &
nohup python -u main.py --alg "ditto" --gpu "3" --C $C --dataset $dataset --model $model --partition $data_partition --n_parties $client --num_local_iterations $iternum --beta $beta > $dir_dir/ditto_${START_TIME}.log &
nohup python -u main.py --alg "fedfomo" --gpu "5" --C $C --dataset $dataset --model $model --partition $data_partition --n_parties $client --num_local_iterations $iternum --beta $beta > $dir_dir/fedfomo_${START_TIME}.log &
nohup python -u main.py --alg "pfedgraph" --gpu "5" --C $C --dataset $dataset --model $model --partition $data_partition --n_parties $client --num_local_iterations $iternum --beta $beta > $dir_dir/pfedgraph_cosine_${START_TIME}.log &
nohup python -u main.py --alg "ipfl" --gpu "7" --C $C --ipfl_lam $lam --ipfl_eta $eta --dataset $dataset --model $model --partition $data_partition --n_parties $client --num_local_iterations $iternum --beta $beta > $dir_dir/ipfl_${START_TIME}.log &

