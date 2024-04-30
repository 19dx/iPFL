# PFL Incentive Market
## Overview
This repository contains the code for the paper "Incentivizing Inclusive Data Contributions in Personalized Federated Learning". \
We propose a novel inclusive Personalized Federated Learning (*iPFL*) framework, to incentivize data contributions from clients with different personalized model requirements and diverse economic utility. Our iPFL leverages a graph-based optimization process to balance the trade-off between model performance and economic utility. We evaluate iPFL on both classification and instruction-tuning tasks, demonstrating its superior performance compared to state-of-the-art baselines.
![alt text](doc/assets/overview.png)

### Baselines

| Method | Hyper-parameter | Personalized or not |
| :----: |:----: |:----: |
| [FedAvg](https://arxiv.org/abs/1602.05629) | \ | No|
| [FedProx](https://arxiv.org/abs/1812.06127) | $\mu=0.1$ |No|
| [Ditto](https://arxiv.org/abs/2012.04221) | $\lambda=1$ |Yes|
| [FedAMP](https://arxiv.org/abs/2007.03797) | $\lambda=0.01$ |Yes|
| [CFL](https://arxiv.org/abs/1910.01991) | $\epsilon_1=2.0, \epsilon_2=2.5$ |Yes|
| [FedFomo](https://arxiv.org/abs/2012.08565) | $M=6$ | Yes |
| [pFedGraph](https://openreview.net/forum?id=33fj5Ph3ot) | $\alpha=0.8, \lambda=0.01$ | Yes|
### Dataset
- **Classification Tasks**: CIFAR-10, Fashion-MNIST, FEMNIST, Shakespeare, PACS
- **Instruction-tuning Tasks**: mixed-Finance (FIQA, TFNS, NWGI), Cofinance (NWGI+CodeAlpaca-20k)
- All data used in our paper is saved in [data.zip](https://drive.google.com/file/d/1NNcRxARJeTKdRc7u71QC5ZhIJ5kFd6Ub/view?usp=sharing), run the following command to download and unzip the data.zip:
```
cd data
wget https://drive.google.com/uc?id=1NNcRxARJeTKdRc7u71QC5ZhIJ5kFd6Ub&export=download
unzip data.zip
```

## Installation Guide
The developmental version of the package has been tested on the following setting:
- **Operating System:** Linux Ubuntu 18.04
- **GPU:** NVIDIA GeForce RTX 3090
### Prerequisites

Before installing, ensure you have the following prerequisites installed:

- Python 3.x
- CUDA Toolkit (for GPU acceleration)

### Installation
Due to the relative independence of classification and instruction-tuning tasks, we use **two separate environments** for convenience. The latter environment utilizes the off-the-shelf integrated framework [OpenFedLLM](https://github.com/rui-ye/OpenFedLLM).

First, clone our iPFL repository \
```git clone https://github.com/19dx/iPFL.git```. \
Then, install the required packages by following the instructions below.
### Classification Tasks
- Packages
```Pytorch=1.10.1, Cuda=11.3, Python=3.8.15, cvxpy, sklearn```
- You can follow the example below to create the running environment: \
    ```
    conda create -n iPFL python=3.8
    conda activate iPFL
    conda install pytorch=1.10.1 torchvision=0.11.2 torchaudio=0.10.1 cudatoolkit=11.3 -c pytorch
    pip install cvxpy scikit-learn
    ```
- These installation steps takes approximately 20 minutes, with some variability depending on network congestion.
### Instruction-tuning Tasks
We utilize OpenFedLLM framework to implement our iPFL algorithm on instruction-tuning tasks. Please refer to [OpenFedLLM](https://github.com/rui-ye/OpenFedLLM) for more details. \
Set up the environment: ```conda env create -f llm_environment.yml``` and perform ```conda activate fedllm``` to activate the environment. These installation steps takes approximately 30 minutes, with some variability depending on network congestion.
## Demo
### Classification Tasks
#### Dataset
| Dataset | Classes | Num of Agents |Training/Test Samples | Data Partitioning |
| :----: |:----: |:----: |:----: |:----: |
| [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) | 10 | 9/10 | 50,000/10,000 | NIID, Cluster, Skew |
| [Fashion-MNIST](https://www.kaggle.com/datasets/zalando-research/fashionmnist) | 10 | 9/10 | 60,000/10,000 | NIID, Cluster, Skew |
| [FEMNIST](https://leaf.cmu.edu/) | 62 | 20 | 3,697,932/ 1,315,228 | natural |
| [Shakespeare](https://leaf.cmu.edu/) | 80 | 10 | 1,690,000/ 563,000 | natural |
| [PACS](https://dali-dl.github.io/project_iccv2017.html) | 7 | 12 | 999/ 1,917 | Cluster |

- First, create a folder named ```./data/``` to store the datasets by running ```mkdir data```.
- For FEMNIST and Shakespeare datasets, we utilize repo [Leaf](https://github.com/TalwalkarLab/leaf/tree/master) to preprocess the raw data and store them in ```./data/```. For raw PACS dataset, you can download from [PACS](https://dali-dl.github.io/project_iccv2017.html) and move the images folder to ```./data/```.
#### Models
LSTM (for Shakespeare), Resnet20 (for PACS), Simple-CNN (for others).

#### Quick Start
1. **Basic experiments** (Fig.2, Fig.3):
- Activate the environment: ```conda activate ipfl```
- Run the following command to train the model on CIFAR-10 dataset: \
```sh run_scripts/run_cifar10_baselines.sh``` 
- Note that the data partitioning for CIFAR-10 and Fashion-MNIST is the argument *data_partition*, the value can be *noniid*, *cluster-3-10*, *noniid-skew-5*, respectively representing NIID, Cluster and Skew setting.
The training log will be saved in ```./output/```. You can change the dataset and model by modifying the corresponding varibles in script.
- The reference running time of one communication round on CIFAR-10 dataset is shown below(tested with GPU: NVIDIA RTX 4060 and CPU: Intel Core i9). 

|                | FedAvg | FedProx | CFL  | Ditto | FedAMP | FedFomo | pFedGraph | iPFL  |
|----------------|--------|---------|------|-------|--------|---------|-----------|-------|
| Run Time (s)   | 30.44  | 40.81   | 31.84| 68.62 | 36.05  | 66.17   | 34.17     | 34.43 |

2. **Inclusive experiment** (Fig.5) with 12 proposed 4 types (*Trader*, *Buyer*, *Seller*, and *Attacker*) of clients:
- To observe the incentive mechanisms in our inclusive market on CIFAR-10 dataset:
```sh run_scripts/run_cifar10_playground.sh```
3. **Model posioning experiments** (Fig.4):
- Note that the model poisoning strategies considered in our paper is (1) shuffling trained parameters, (2) flipping the sign of updates, (3) uploading random gaussian noise and (4) uploading the same value for all elements. We rather prefer cluster them into 2 groups based on whether local training for the model poisoning is necessary. We call the groups of malicious clients in our code "attacker" and "freerider" respectively and set the corresponding hyperparameters.
- Run script for simplicity: ```sh run_scripts/run_cifar10_robustness.sh```
4. **Liars experiments** (Table 2):
- Use *liar_ratio* to control the ratio of liars hidden in clients who exaggerate its cost or dataset size
- Run ```python main.py --alg "ipfl" --liar_ratio 0.1 --liar_type cost --liar_exaggerate_ratio 2```

### Instruction-tuning Tasks
#### Dataset: 
- mixed-Finance: consists of FIQA, TNFS and NWGI. Every 2 clients possesses one sub dataset and each client has 200 samples. You can download the dataset from Huggingface: [FIQA](https://huggingface.co/datasets/pauri32/fiqa-2018), [TFNS](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment), [NWGI](https://huggingface.co/datasets/oliverwang15/news_with_gpt_instructions).
- Code+Finance (Cofinance): consists of NWGI (3 clients, 200 samples per client) and CodeAlpaca datasets (5 clients, 500 samples per client). You can download the code dataset from Huggingface: [CodeAlpaca](https://huggingface.co/datasets/lucasmccabe-lmi/CodeAlpaca-20k).
- Please move the datasets to a certain folder and pass the path to --local_data_dir.
- Model: We use pretrained [Llama2-7B](https://huggingface.co/meta-llama/Llama-2-7b) as the initial model for federated learning.
#### Quick Start
- Activate the environment: ```conda activate fedllm```
- For mixed-Finance dataset, run ```sh run_scripts/run_finance.sh``` (Table 1)
- For Cofinance dataset, run ```sh run_scripts/run_cofinance.sh``` (Table 1)
- Note: setting for K, c and hyperparameters of iPFL default to the corresponding scripts or refer to our paper.
#### Evaluation
- For finance dataset (FIQA, TNFS, NWGI), use their own test set to evaluate the performance in ```./eval_llm/```. You can simply run ```sh eval_llm/run_evaluate.sh``` to obtain the test accuracy for each client.
- For code dataset, we utilize ```bigcode-evaluation-harness``` repo to evaluate the performance. Please refer to [bigcode-evaluation-harness](https://github.com/bigcode-project/bigcode-evaluation-harness) for more details.


## Citation
Please cite our paper if you find the repository helpful.
