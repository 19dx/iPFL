import torch.utils.data as data
import numpy as np
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100, FashionMNIST, ImageFolder, DatasetFolder
from torch.utils.data import DataLoader, Dataset
import os

from .partition_data import partition_data

class Cifar_Truncated(data.Dataset):
    def __init__(self, data, labels, transform=None):
        super(Cifar_Truncated, self).__init__()
        self.data = data
        self.labels = labels
        self.transform = transform
        
    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.data)

def cifar_dataset_read(args, dataset, base_path, batch_size, n_parties, partition, beta, skew_class):
    data_path = os.path.join(base_path, dataset)
    if dataset == "cifar10":
        train_dataset = CIFAR10(data_path, True, download=True)
        test_dataset = CIFAR10(data_path, False, download=True)
        
        transform_train=transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

        transform_test=transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    elif dataset == "cifar100":
        train_dataset = CIFAR100(data_path, True, download=True)
        test_dataset = CIFAR100(data_path, False, download=True)
        normalize = transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                             std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
        transform_train=transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize])

        transform_test=transforms.Compose(
            [transforms.ToTensor(),
            normalize])
        
    train_image = train_dataset.data
    train_label = np.array(train_dataset.targets)
    test_image = test_dataset.data
    test_label = np.array(test_dataset.targets)
    n_train = train_label.shape[0]

    net_dataidx_map, client_num_samples, traindata_cls_counts, data_distributions = partition_data(partition, n_train, n_parties, train_label, beta, skew_class)

    train_dataloaders = []
    for i in range(n_parties):
        train_idxs = net_dataidx_map[i]
        local_train_image = train_image[train_idxs]
        local_train_label = train_label[train_idxs]
        train_dataset = Cifar_Truncated(data=local_train_image, labels=local_train_label, transform=transform_train)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        train_dataloaders.append(train_loader)
    
    test_dataset = Cifar_Truncated(data=test_image, labels=test_label, transform=transform_test)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_dataloaders, test_loader, client_num_samples, traindata_cls_counts, data_distributions

def fashionmnist_dataset_read(args, dataset, base_path, batch_size, n_parties, partition, beta, skew_class):
    data_path = os.path.join(base_path, dataset)
    train_dataset = FashionMNIST(data_path, True, download=True)
    test_dataset = FashionMNIST(data_path, False, download=True)
    transform_train=transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(28, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))])

    transform_test=transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))])
        
    train_image = np.array(train_dataset.data)
    train_label = np.array(train_dataset.targets)
    test_image = np.array(test_dataset.data)
    test_label = np.array(test_dataset.targets)
    n_train = train_label.shape[0]
    net_dataidx_map, client_num_samples, traindata_cls_counts, data_distributions = partition_data(partition, n_train, n_parties, train_label, beta, skew_class)
    
    train_dataloaders = []
    for i in range(n_parties):
        train_idxs = net_dataidx_map[i]
        train_dataset = Cifar_Truncated(data=train_image[train_idxs], labels=train_label[train_idxs], transform=transform_train)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        train_dataloaders.append(train_loader)
    
    test_dataset = Cifar_Truncated(data=test_image, labels=test_label, transform=transform_test)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_dataloaders, test_loader, client_num_samples, traindata_cls_counts, data_distributions

class ImageFolder_custom(DatasetFolder):
    def __init__(self, root, dataidxs=None, transform=None):
        self.root = root
        self.dataidxs = dataidxs
        self.transform = transform

        imagefolder_obj = ImageFolder(self.root, self.transform)
        self.loader = imagefolder_obj.loader
        if self.dataidxs is not None:
            self.samples = np.array(imagefolder_obj.samples)[self.dataidxs]
        else:
            self.samples = np.array(imagefolder_obj.samples)

    def __getitem__(self, index):
        path = self.samples[index][0]
        target = self.samples[index][1]
        target = int(target)
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target

    def __len__(self):
        if self.dataidxs is None:
            return len(self.samples)
        else:
            return len(self.dataidxs)

def get_all_pacs_dataloader(args, base_path, batch_size, n_parties, partition, beta, skew_class):
    assert n_parties % 4 == 0
    n_per_domain = int(n_parties/4)
    train_dataloaders = []
    test_dataloaders = []
    domains = []
    client_num_samples = []
    data_distributions, test_distributions = [], []
    transforms_train = transforms.Compose([
        transforms.RandomResizedCrop(64, scale=(0.75, 1)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])
    transforms_test = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])
    real_datadir = os.path.join(base_path,'PACS')
    for domain_name in ['art_painting', 'cartoon', 'photo', 'sketch']:
        dataset_all = ImageFolder(os.path.join(real_datadir, domain_name), transforms_train)
        labels = dataset_all.targets
        dataset_size = len(labels)
        idxs = np.random.permutation(dataset_size)
        train_idxs = idxs[:int(0.8*dataset_size)]
        test_idxs = idxs[int(0.8*dataset_size):]
        test_distribution = record_test_distribution(np.array(labels)[test_idxs])
        test_dataset = ImageFolder_custom(root=os.path.join(real_datadir, domain_name), dataidxs=test_idxs, transform=transforms_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True, shuffle=False) 
        test_dataloaders.append(test_loader)
        test_distributions.append(test_distribution)
        
        net_dataidx_map, client_num_samples_per_domain, _, data_distribution = partition_data(partition, len(train_idxs), n_per_domain, np.array(labels)[train_idxs], beta, skew_class)
        data_distributions.extend(data_distribution.tolist())

        for i in range(n_per_domain):
            idxs = net_dataidx_map[i]
            sketch_train_dataset = ImageFolder_custom(root=os.path.join(real_datadir, domain_name), dataidxs=train_idxs[idxs], transform=transforms_train)
            client_num_samples.append(client_num_samples_per_domain[i])
            train_loader = DataLoader(dataset=sketch_train_dataset, batch_size=batch_size, pin_memory=True, shuffle=True)
            train_dataloaders.append(train_loader)
            domains.append(domain_name)
    test_dataloaders_info = [(a, b) for a, b in zip(test_dataloaders, test_distributions)]
    return train_dataloaders, test_dataloaders_info, np.array(client_num_samples), data_distributions, domains

def record_test_distribution(y_train):
    num_classes = int(y_train.max()) + 1
    test_distribution = [0]*num_classes
    unq, unq_cnt = np.unique(y_train, return_counts=True)
    total = sum(unq_cnt)
    for i in range(len(unq)):
        test_distribution[unq[i]] = unq_cnt[i] / total
    return np.array(test_distribution)
