START_TIME=`date +%s`
dataset='cifar10'
data_partition="cluster-3-10"
beta=0.1
client=9
model='simplecnn'
iternum=200
C=1
sample_fraction=1
dir_path=./output_test/${dataset}_${data_partition}_beta${beta}_${model}_it${iternum}_c${client}_p${sample_fraction}_C${C}
if [ ! -d $dir_path ]; then
    mkdir $dir_path
else
    echo "dir exists"
fi
lam=2
eta=5

## ================== Attack on model ==================
attack_type='sign_flip'
attack_ratio=0.2

dir_dir=${dir_path}/baselines_attack
if [ ! -d $dir_dir ]; then
    mkdir $dir_dir
else
    echo "dir exists"
fi
# python main.py --alg "fedamp" --gpu "4" --attack_type $attack_type --attack_ratio $attack_ratio --C $C --dataset $dataset --model $model --partition $data_partition --n_parties $client --num_local_iterations $iternum --beta $beta 
for attack_type in 'shuffle' 'sign_flip'
do
    nohup python -u main.py --alg "fedavg" --gpu "2" --attack_type $attack_type --attack_ratio $attack_ratio --C $C --dataset $dataset --model $model --partition $data_partition --n_parties $client --num_local_iterations $iternum --beta $beta > $dir_dir/fedavg_r${attack_ratio}_${attack_type}_${START_TIME}.log &
    nohup python -u main.py --alg "fedprox" --gpu "2" --attack_type $attack_type --attack_ratio $attack_ratio --C $C --dataset $dataset --model $model --partition $data_partition --n_parties $client --num_local_iterations $iternum --beta $beta > $dir_dir/fedprox_r${attack_ratio}_${attack_type}_${START_TIME}.log &
    nohup python -u main.py --alg "fedamp" --gpu "4" --attack_type $attack_type --attack_ratio $attack_ratio --C $C --dataset $dataset --model $model --partition $data_partition --n_parties $client --num_local_iterations $iternum --beta $beta > $dir_dir/fedamp_r${attack_ratio}_${attack_type}_${START_TIME}.log &
    nohup python -u main.py --alg "ditto" --gpu "7" --attack_type $attack_type --attack_ratio $attack_ratio --C $C --dataset $dataset --model $model --partition $data_partition --n_parties $client --num_local_iterations $iternum --beta $beta > $dir_dir/ditto_r${attack_ratio}_${attack_type}_${START_TIME}.log &
    nohup python -u main.py --alg "cfl" --gpu "3" --attack_type $attack_type --attack_ratio $attack_ratio --C $C --dataset $dataset --model $model --partition $data_partition --n_parties $client --num_local_iterations $iternum --beta $beta > $dir_dir/cfl_r${attack_ratio}_${attack_type}_${START_TIME}.log &
    nohup python -u main.py --alg "fedfomo" --gpu "7" --attack_type $attack_type --attack_ratio $attack_ratio --C $C --dataset $dataset --model $model --partition $data_partition --n_parties $client --num_local_iterations $iternum --beta $beta > $dir_dir/fedfomo_r${attack_ratio}_${attack_type}_${START_TIME}.log &
    nohup python -u main.py --alg "pfedgraph" --gpu "7" --attack_type $attack_type --attack_ratio $attack_ratio --C $C --dataset $dataset --model $model --partition $data_partition --n_parties $client --num_local_iterations $iternum --beta $beta > $dir_dir/pfedgraph_cosine_r${attack_ratio}_${attack_type}_${START_TIME}.log &
    nohup python -u main.py --alg "ipfl" --gpu "4" --ipfl_lam $lam --ipfl_eta $eta --attack_type $attack_type --attack_ratio $attack_ratio --C $C --dataset $dataset --model $model --partition $data_partition --n_parties $client --num_local_iterations $iternum --beta $beta > $dir_dir/ipfl_r${attack_ratio}_${attack_type}_${START_TIME}.log &
done
# ================== Freeriding Clients ==================
freeride_type='same_value'
freerider_ratio=0.2

dir_dir=${dir_path}/baselines_freeride
if [ ! -d $dir_dir ]; then
    mkdir $dir_dir
else
    echo "dir exists"
fi
# python main.py --alg "fedamp" --gpu "6" --freeride_type $freeride_type --freerider_ratio $freerider_ratio --C $C --dataset $dataset --model $model --partition $data_partition --n_parties $client --num_local_iterations $iternum --beta $beta 
for freeride_type in 'same_value' 'gauss'
do
    nohup python -u main.py --alg "fedavg" --gpu "6" --freeride_type $freeride_type --freerider_ratio $freerider_ratio --C $C --dataset $dataset --model $model --partition $data_partition --n_parties $client --num_local_iterations $iternum --beta $beta > $dir_dir/fedavg_r${freerider_ratio}_${freeride_type}_${START_TIME}.log &
    nohup python -u main.py --alg "fedprox" --gpu "6" --freeride_type $freeride_type --freerider_ratio $freerider_ratio --C $C --dataset $dataset --model $model --partition $data_partition --n_parties $client --num_local_iterations $iternum --beta $beta > $dir_dir/fedprox_r${freerider_ratio}_${freeride_type}_${START_TIME}.log &
    nohup python -u main.py --alg "fedamp" --gpu "3" --freeride_type $freeride_type --freerider_ratio $freerider_ratio --C $C --dataset $dataset --model $model --partition $data_partition --n_parties $client --num_local_iterations $iternum --beta $beta > $dir_dir/fedamp_r${freerider_ratio}_${freeride_type}_${START_TIME}.log &
    nohup python -u main.py --alg "ditto" --gpu "0" --freeride_type $freeride_type --freerider_ratio $freerider_ratio --C $C --dataset $dataset --model $model --partition $data_partition --n_parties $client --num_local_iterations $iternum --beta $beta > $dir_dir/ditto_r${freerider_ratio}_${freeride_type}_${START_TIME}.log &
    nohup python -u main.py --alg "cfl" --gpu "3" --freeride_type $freeride_type --freerider_ratio $freerider_ratio --C $C --dataset $dataset --model $model --partition $data_partition --n_parties $client --num_local_iterations $iternum --beta $beta > $dir_dir/cfl_r${freerider_ratio}_${freeride_type}_${START_TIME}.log &
    nohup python -u main.py --alg "fedfomo" --gpu "7" --freeride_type $freeride_type --freerider_ratio $freerider_ratio --C $C --dataset $dataset --model $model --partition $data_partition --n_parties $client --num_local_iterations $iternum --beta $beta > $dir_dir/fedfomo_r${freerider_ratio}_${freeride_type}_${START_TIME}.log &
    nohup python -u main.py --alg "pfedgraph" --gpu "5" --freeride_type $freeride_type --freerider_ratio $freerider_ratio --C $C --dataset $dataset --model $model --partition $data_partition --n_parties $client --num_local_iterations $iternum --beta $beta > $dir_dir/pfedgraph_cosine_r${freerider_ratio}_${freeride_type}_${START_TIME}.log &
    nohup python -u main.py --alg "ipfl" --gpu "4" --ipfl_lam $lam --ipfl_eta $eta --freeride_type $freeride_type --freerider_ratio $freerider_ratio --C $C --dataset $dataset --model $model --partition $data_partition --n_parties $client --num_local_iterations $iternum --beta $beta > $dir_dir/ipfl_r${freerider_ratio}_${freeride_type}_${START_TIME}.log &
done