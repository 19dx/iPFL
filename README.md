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
- Any additional dependencies listed in `requirements.txt`

### Installation
Due to the relative independence of classification and instruction-tuning tasks, we use **two separate environments** for convenience. The latter environment utilizes the off-the-shelf integrated framework [OpenFedLLM](https://github.com/rui-ye/OpenFedLLM).

First, clone our iPFL repository \
```git clone https://github.com/19dx/iPFL.git```. \
Then, install the required packages by following the instructions below.
### Classification Tasks
- Packages
```Pytorch=1.10.1, Cuda=11.3, Python=3.8.15, random, cvxpy, numpy, sklearn, copy```
- You can follow the example below to create the running environment: \
    ```
    conda create -n iPFL python=3.8.15
    conda activate iPFL
    conda install pytorch=1.10.1 torchvision=0.11.2 torchaudio=0.10.1 cudatoolkit=11.3 -c
    pip install random cvxpy numpy scikit-learn copy scipy
    pip install -r requirements.txt
    ```

### Instruction-tuning Tasks
We utilize OpenFedLLM framework to implement our iPFL algorithm on instruction-tuning tasks. Please refer to [OpenFedLLM](https://github.com/rui-ye/OpenFedLLM) for more details. \
Set up the environment: ```conda env create -f llm_environment.yml``` and perform ```conda activate fedllm``` to activate the environment.
## Demo
### Classification Tasks
#### Dataset
| Dataset | Classes | Num of Agents |Training/Test Samples | Data Partitioning |
| :----: |:----: |:----: |:----: |:----: |
| [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) | 10 | 9/10 | 50,000/10,000 | synthetic |
| [Fashion-MNIST](https://www.kaggle.com/datasets/zalando-research/fashionmnist) | 10 | 9/10 | 60,000/10,000 | NIID, Cluster, Skew |
| [FEMNIST](https://leaf.cmu.edu/) | 62 | 20 | 3,697,932/ 1,315,228 | natural |
| [Shakespeare](https://leaf.cmu.edu/) | 80 | 10 | 1,690,000/ 563,000 | natural |
| [PACS](https://dali-dl.github.io/project_iccv2017.html) | 7 | 12 | 999/ 1,917 | Cluster |

- First, create a folder named ```./data/``` to store the datasets by running ```mkdir data```.
- For FEMNIST and Shakespeare datasets, we utilize repo [Leaf](https://github.com/TalwalkarLab/leaf/tree/master) to preprocess the raw data and store them in ```./data/```. For PACS dataset, you can download from [PACS](https://dali-dl.github.io/project_iccv2017.html) and move the images folder to ```./data/```.
#### Models
LSTM (for Shakespeare), Resnet20 (for PACS), Simple-CNN (for others).

#### Quick Start
``` sh run_scripts/run_cifar10_baselines.sh ``` \
The training log will be saved in ```./output/```. You can change the dataset and model by modifying the corresponding script.

### Instruction-tuning Tasks
#### Dataset: 
- mixed-Finance: consists of FIQA, TNFS and NWGI. Every 2 clients possesses one sub dataset and each client has 200 samples. You can download the dataset from Huggingface: [FIQA](https://huggingface.co/datasets/pauri32/fiqa-2018), [TFNS](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment), [NWGI](https://huggingface.co/datasets/oliverwang15/news_with_gpt_instructions).
- Code+Finance(Cofinance): consists of NWGI (3 clients, 200 samples per client) and CodeAlpaca datasets (5 clients, 500 samples per client). You can download the code dataset from Huggingface: [CodeAlpaca](https://huggingface.co/datasets/lucasmccabe-lmi/CodeAlpaca-20k).
- Please move the datasets to a certain folder and pass the path to --local_data_dir.
- Model: We use pretrained [Llama2-7B](https://huggingface.co/meta-llama/Llama-2-7b) as the initial model for federated learning.
#### Quick Start
- For mixed-Finance dataset, run ```sh run_scripts/run_finance.sh```
- For Cofinance dataset, run ```sh run_scripts/run_cofinance.sh```
- Note: setting for K, c and hyperparameters of iPFL default to the corresponding scripts or refer to our paper.
#### Evaluation
- For finance dataset (FIQA, TNFS, NWGI), use their own test set to evaluate the performance in ```./eval_llm/```. You can simply run ```sh eval_llm/run_evaluate.sh``` to obtain the test accuracy for each client.
- For code dataset, we utilize ```bigcode-evaluation-harness``` repo to evaluate the performance. Please refer to [bigcode-evaluation-harness](https://github.com/bigcode-project/bigcode-evaluation-harness) for more details.


## Citation
Please cite our paper if you find the repository helpful.
