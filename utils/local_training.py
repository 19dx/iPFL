import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
from .evaluation import compute_local_test_accuracy

def local_train(args, nets_this_round, train_local_dls, test_dl, data_distributions, best_test_acc_list):
    
    for net_id, net in nets_this_round.items():
        train_local_dl = train_local_dls[net_id]
        data_distribution = data_distributions[net_id]

        # Set Optimizer
        if args.optimizer == 'adam':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, weight_decay=args.reg)
        elif args.optimizer == 'amsgrad':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, weight_decay=args.reg,
                                amsgrad=True)
        elif args.optimizer == 'sgd':
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, momentum=0.9,
                                weight_decay=args.reg)
        criterion = torch.nn.CrossEntropyLoss().cuda()
        net.cuda()
        net.train()
            
        iterator = iter(train_local_dl)
        for iteration in range(args.num_local_iterations):
            try:
                x, target = next(iterator)
            except StopIteration:
                iterator = iter(train_local_dl)
                x, target = next(iterator)

            x, target = x.cuda(), target.cuda()
            
            optimizer.zero_grad()
            target = target.long()

            out = net(x)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
        
        personalized_test_acc, generalized_test_acc = compute_local_test_accuracy(net_id, net, test_dl, data_distribution, len(best_test_acc_list))

        if personalized_test_acc > best_test_acc_list[net_id]:
            best_test_acc_list[net_id] = personalized_test_acc
        print('>> Client {} | Personalized Test Acc: {:.5f} | Generalized Test Acc: {:.5f}'.format(net_id, personalized_test_acc, generalized_test_acc))
        net.to('cpu')
    return np.array(best_test_acc_list).mean()

def local_train_fedavg(args, nets_this_round, train_local_dls, freeriders):
    for net_id, net in nets_this_round.items():
        if net_id in freeriders:
            continue
        train_local_dl = train_local_dls[net_id]

        # Set Optimizer
        if args.optimizer == 'adam':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, weight_decay=args.reg)
        elif args.optimizer == 'amsgrad':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, weight_decay=args.reg,
                                amsgrad=True)
        elif args.optimizer == 'sgd':
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, momentum=0.9,
                                weight_decay=args.reg)
        criterion = torch.nn.CrossEntropyLoss().cuda()
        net.cuda()
        net.train()
            
        iterator = iter(train_local_dl)
        for iteration in range(args.num_local_iterations):
            try:
                x, target = next(iterator)
            except StopIteration:
                iterator = iter(train_local_dl)
                x, target = next(iterator)

            x, target = x.cuda(), target.cuda()
            
            optimizer.zero_grad()
            target = target.long()

            out = net(x)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
        net.to('cpu')

def local_train_fedprox(args, nets_this_round, global_model, train_local_dls, freeriders):
    global_model.cuda()
    for net_id, net in nets_this_round.items():
        if net_id in freeriders:
            continue
        train_local_dl = train_local_dls[net_id]

        # Set Optimizer
        if args.optimizer == 'adam':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, weight_decay=args.reg)
        elif args.optimizer == 'amsgrad':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, weight_decay=args.reg,
                                amsgrad=True)
        elif args.optimizer == 'sgd':
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, momentum=0.9,
                                weight_decay=args.reg)
        criterion = torch.nn.CrossEntropyLoss().cuda()
        net.cuda()
        net.train()
            
        iterator = iter(train_local_dl)
        for iteration in range(args.num_local_iterations):
            try:
                x, target = next(iterator)
            except StopIteration:
                iterator = iter(train_local_dl)
                x, target = next(iterator)

            x, target = x.cuda(), target.cuda()
            
            optimizer.zero_grad()
            target = target.long()

            out = net(x)
            loss = criterion(out, target)
            for param_p, param in zip(global_model.parameters(), net.parameters()):
                loss += ((args.mu / 2) * torch.norm((param - param_p)) ** 2)
            loss.backward()
            optimizer.step()
        net.to('cpu')
    global_model.to('cpu')

def local_train_ditto(args, nets_this_round, p_models, train_local_dls, freeriders):
    for net_id, net in nets_this_round.items():
        if net_id in freeriders:
            continue
        vnet = p_models[net_id]
        train_local_dl = train_local_dls[net_id]

        # Pre-Trainging Test Accuracy
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.reg)
        poptimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, vnet.parameters()), lr = args.lr, momentum=0.9, weight_decay=args.reg)
        criterion = torch.nn.CrossEntropyLoss().cuda()
        net.cuda()
        net.train()
        vnet.cuda()
        vnet.train()
        iterator = iter(train_local_dl)
        for iteration in range(args.num_local_iterations):
            try:
                x, target = next(iterator)
            except StopIteration:
                iterator = iter(train_local_dl)
                x, target = next(iterator)

            x, target = x.cuda(), target.cuda()
            
            poptimizer.zero_grad()
            target = target.long()

            out = vnet(x)
            loss = criterion(out, target)
            for param_p, param in zip(vnet.parameters(), net.parameters()):
                loss += ((args.ditto_lamda / 2) * torch.norm((param - param_p)) ** 2)
            loss.backward()
            poptimizer.step()
            
        for iteration in range(args.num_local_iterations):
            try:
                x, target = next(iterator)
            except StopIteration:
                iterator = iter(train_local_dl)
                x, target = next(iterator)

            x, target = x.cuda(), target.cuda()
            
            optimizer.zero_grad()
            target = target.long()

            out = net(x)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
            
        net.to('cpu')
        vnet.to('cpu')

def local_train_fedamp(args, round, nets_this_round, cluster_models, train_local_dls, freeriders):
    for net_id, net in nets_this_round.items():
        if net_id in freeriders:
            continue
        train_local_dl = train_local_dls[net_id]
        cluster_model = cluster_models[net_id]

        # Set Optimizer
        if args.optimizer == 'adam':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, weight_decay=args.reg)
        elif args.optimizer == 'amsgrad':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, weight_decay=args.reg,
                                amsgrad=True)
        elif args.optimizer == 'sgd':
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.reg)
        criterion = torch.nn.CrossEntropyLoss()
        
        cluster_model.cuda()
        net.cuda()
        net.train()
        iterator = iter(train_local_dl)
        for iteration in range(args.num_local_iterations):
            try:
                x, target = next(iterator)
            except StopIteration:
                iterator = iter(train_local_dl)
                x, target = next(iterator)
            x, target = x.cuda(), target.cuda()
            
            optimizer.zero_grad()
            target = target.long()

            out = net(x)
            loss = criterion(out, target)
            
            if round > 0:
                for param_p, param in zip(cluster_model.parameters(), net.parameters()):
                    loss += ((args.fedamp_lam1 / 2) * torch.norm((param - param_p)) ** 2)
                
            loss.backward()
            optimizer.step()
        
        net.to('cpu')
        cluster_model.to('cpu')

def local_train_ipfl(args, round, nets_this_round, prox_models, train_local_dls, freeriders):
    for net_id, net in nets_this_round.items():
        if net_id in freeriders:
            continue
        train_local_dl = train_local_dls[net_id]
        if round > 0:
            prox_vector = prox_models[net_id].cuda()
        # Set Optimizer
        if args.optimizer == 'adam':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, weight_decay=args.reg)
        elif args.optimizer == 'amsgrad':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, weight_decay=args.reg,
                                amsgrad=True)
        elif args.optimizer == 'sgd':
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.reg)
        criterion = torch.nn.CrossEntropyLoss()
        
        net.cuda()
        net.train()
        iterator = iter(train_local_dl)
        for iteration in range(args.num_local_iterations):
            try:
                x, target = next(iterator)
            except StopIteration:
                iterator = iter(train_local_dl)
                x, target = next(iterator)
            x, target = x.cuda(), target.cuda()
            
            optimizer.zero_grad()
            target = target.long()

            out = net(x)
            if torch.isnan(out).any():
                print("Nan of {} with iter={}".format(net_id, iteration))
                exit()
            loss = criterion(out, target)

            if round > 0:
                flatten_model = []
                for param in net.parameters():
                    flatten_model.append(param.reshape(-1))
                flatten_model = torch.cat(flatten_model)
                loss2 = args.ipfl_lam/(2*args.ipfl_eta) * torch.norm(flatten_model-prox_vector)**2 #/len(prox_vector)
                loss2.backward()

            loss.backward()
            optimizer.step()
        
        net.to('cpu')

def local_train_pfedgraph(args, round, nets_this_round, cluster_models, train_local_dls, freeriders):
    for net_id, net in nets_this_round.items():
        if net_id in freeriders:
            continue
        train_local_dl = train_local_dls[net_id]

        # Set Optimizer
        if args.optimizer == 'adam':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, weight_decay=args.reg)
        elif args.optimizer == 'amsgrad':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, weight_decay=args.reg,
                                amsgrad=True)
        elif args.optimizer == 'sgd':
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.reg)
        criterion = torch.nn.CrossEntropyLoss()
        cluster_model = cluster_models[net_id].cuda()
        
        flatten_cluster_model = []
        for param in cluster_model.parameters():
            flatten_cluster_model.append(param.reshape(-1))
        flatten_cluster_model = torch.cat(flatten_cluster_model) 
        
        net.cuda()
        net.train()
        iterator = iter(train_local_dl)
        for iteration in range(args.num_local_iterations):
            try:
                x, target = next(iterator)
            except StopIteration:
                iterator = iter(train_local_dl)
                x, target = next(iterator)
            x, target = x.cuda(), target.cuda()
            
            optimizer.zero_grad()
            target = target.long()

            out = net(x)
            loss = criterion(out, target)

            if round > 0:
                flatten_model = []
                for param in net.parameters():
                    flatten_model.append(param.reshape(-1))
                flatten_model = torch.cat(flatten_model)
                loss2 = args.pfedgraph_lam * torch.nn.functional.cosine_similarity(flatten_cluster_model.unsqueeze(0), flatten_model.unsqueeze(0), eps=1e-8)
                # loss2 = 
                loss2.backward()
                
            loss.backward()
            optimizer.step()
        
        net.to('cpu')
