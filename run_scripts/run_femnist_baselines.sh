START_TIME=`date +%s`
dataset='femnist'
data_partition="cluster-3-10"
beta=0.1
client=20
test_client=20
model='cnn_femnist'
iternum=50
C=1
sample_fraction=1
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
nohup python -u main.py --alg "local" --gpu "2" --C $C --dataset $dataset --model $model --partition $data_partition --leaf_train_num $client --leaf_test_num $test_client --num_local_iterations $iternum --beta $beta > $dir_dir/local_${START_TIME}.log &
nohup python -u main.py --alg "fedavg" --gpu "2" --C $C --dataset $dataset --model $model --partition $data_partition --leaf_train_num $client --leaf_test_num $test_client --num_local_iterations $iternum --beta $beta > $dir_dir/fedavg_${START_TIME}.log &
nohup python -u main.py --alg "fedprox" --gpu "2" --C $C --dataset $dataset --model $model --partition $data_partition --leaf_train_num $client --leaf_test_num $test_client --num_local_iterations $iternum --beta $beta > $dir_dir/fedprox_${START_TIME}.log &
nohup python -u main.py --alg "cfl" --gpu "3" --C $C --dataset $dataset --model $model --partition $data_partition --leaf_train_num $client --leaf_test_num $test_client --num_local_iterations $iternum --beta $beta > $dir_dir/cfl_${START_TIME}.log &
nohup python -u main.py --alg "fedamp" --gpu "3" --C $C --dataset $dataset --model $model --partition $data_partition --leaf_train_num $client --leaf_test_num $test_client --num_local_iterations $iternum --beta $beta > $dir_dir/fedamp_${START_TIME}.log &
nohup python -u main.py --alg "ditto" --gpu "3" --C $C --dataset $dataset --model $model --partition $data_partition --leaf_train_num $client --leaf_test_num $test_client --num_local_iterations $iternum --beta $beta > $dir_dir/ditto_${START_TIME}.log &
nohup python -u main.py --alg "fedfomo" --gpu "3" --C $C --dataset $dataset --model $model --partition $data_partition --leaf_train_num $client --leaf_test_num $test_client --num_local_iterations $iternum --beta $beta > $dir_dir/fedfomo_${START_TIME}.log &
nohup python -u main.py --alg "pfedgraph" --gpu "3" --C $C --dataset $dataset --model $model --partition $data_partition --leaf_train_num $client --leaf_test_num $test_client --num_local_iterations $iternum --beta $beta > $dir_dir/pfedgraph_cosine_${START_TIME}.log &
nohup python -u main.py --alg "ipfl" --gpu "7" --C $C --dataset $dataset --model $model --partition $data_partition --leaf_train_num $client --leaf_test_num $test_client --num_local_iterations $iternum --beta $beta > $dir_dir/ipfl_${START_TIME}.log &

