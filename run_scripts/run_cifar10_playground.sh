START_TIME=`date +%s`
data_partition="cluster-3-10"
beta=0.1
client=12
dataset='cifar10'
iternum=500
comm_round=20
model='simplecnn'
sample_fraction=1
C=1
dir_path=./output/${dataset}_${data_partition}_beta${beta}_${model}_it${iternum}_c${client}_p${sample_fraction}_C${C}
if [ ! -d $dir_path ]; then
    mkdir $dir_path
else
    echo "dir exists"
fi
dir_dir=${dir_path}/playground
if [ ! -d $dir_dir ]; then
    mkdir $dir_dir
else
    echo "dir exists"
fi

lambda=2
eta=10
k=5e5
attack_type='shuffle'

# python main.py --diverse --attack_type $attack_type --attack_ratio $attack_ratio --alg "ipfl" --gpu "0" --C $C --K $k --ipfl_lam $lambda  --ipfl_eta $eta --dataset $dataset --model $model --partition $data_partition --n_parties $client --num_local_iterations $iternum --beta $beta
nohup python -u main.py --alg "ipfl" --gpu "0" --diverse --comm_round $comm_round --attack_type $attack_type --C $C --K $k --ipfl_lam $lambda  --ipfl_eta $eta --dataset $dataset --model $model --partition $data_partition --n_parties $client --num_local_iterations $iternum --beta $beta > $dir_dir/ipfl_K${k}_lam_${lambda}_eta${eta}_${START_TIME}.log &
