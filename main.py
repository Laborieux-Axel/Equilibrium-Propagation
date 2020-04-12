import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torch.nn.functional as F
import argparse

import os
from datetime import datetime
import time
import math
from model_utils import *


parser = argparse.ArgumentParser(description='Eqprop')
parser.add_argument('--model',type = str, default = 'MLP', metavar = 'm', help='model')
parser.add_argument('--pool',type = str, default = 'max', metavar = 'p', help='pooling')
parser.add_argument('--task',type = str, default = 'MNIST', metavar = 't', help='task')
parser.add_argument('--archi', nargs='+', type = int, default = [784, 512, 10], metavar = 'A', help='architecture of the network')
parser.add_argument('--act',type = str, default = 'mysig', metavar = 'a', help='activation function')
parser.add_argument('--optim', type = str, default = 'sgd', metavar = 'opt', help='optimizer for training')
parser.add_argument('--lrs', nargs='+', type = float, default = [], metavar = 'l', help='layer wise lr')
parser.add_argument('--mbs',type = int, default = 20, metavar = 'M', help='minibatch size')
parser.add_argument('--T1',type = int, default = 20, metavar = 'T1', help='Time of first phase')
parser.add_argument('--T2',type = int, default = 4, metavar = 'T2', help='Time of second phase')
parser.add_argument('--betas', nargs='+', type = float, default = [0.0, 0.01], metavar = 'Bs', help='Betas')
parser.add_argument('--epochs',type = int, default = 1,metavar = 'EPT',help='Number of epochs per tasks')
parser.add_argument('--check-thm', default = False, action = 'store_true', help='checking the gdu while training')
parser.add_argument('--save', default = False, action = 'store_true', help='saving results')
parser.add_argument('--todo', type = str, default = 'train', metavar = 'tr', help='training or plot gdu curves')
parser.add_argument('--seed',type = int, default = 2, metavar = 's', help='random seed')
parser.add_argument('--device',type = int, default = 0, metavar = 'd', help='device')

args = parser.parse_args()


print('\n')
print('##################################################################')
print('\nargs\tmbs\tT1\tT2\tepochs\tactivation\tbetas')
print('\t',args.mbs,'\t',args.T1,'\t',args.T2,'\t',args.epochs,'\t',args.act, '\t', args.betas)
print('\n')

device = torch.device('cuda:'+str(args.device) if torch.cuda.is_available() else 'cpu')


date = datetime.now().strftime('%Y-%m-%d')
time = datetime.now().strftime('%H-%M-%S')
path = 'results/'+date+'/'+time+'_gpu'+str(args.device)
if not(os.path.exists(path)) and args.save:
    os.makedirs(path)

mbs=args.mbs
torch.manual_seed(args.seed)


if args.task=='MNIST':
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean=(0.0,), std=(1.0,))])

    mnist_dset_train = torchvision.datasets.MNIST('./mnist_pytorch', train=True, transform=transform, target_transform=None, download=True)
    train_loader = torch.utils.data.DataLoader(mnist_dset_train, batch_size=mbs, shuffle=True, num_workers=1)

    mnist_dset_test = torchvision.datasets.MNIST('./mnist_pytorch', train=False, transform=transform, target_transform=None, download=True)
    test_loader = torch.utils.data.DataLoader(mnist_dset_test, batch_size=200, shuffle=False, num_workers=1)

elif args.task=='CIFAR10':

    transform_train = torchvision.transforms.Compose([#torchvision.transforms.RandomHorizontalFlip(0.5),
                                                      #torchvision.transforms.RandomCrop(size=[32,32], padding=4, padding_mode='edge'),
                                                      torchvision.transforms.ToTensor(), 
                                                      torchvision.transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(3*0.2023, 3*0.1994, 3*0.2010))
                                                     ])   

    transform_test = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), 
                                                     torchvision.transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(3*0.2023, 3*0.1994, 3*0.2010)) ]) 

    cifar10_train_dset = torchvision.datasets.CIFAR10('./cifar10_pytorch', train=True, transform=transform_train, download=True)
    cifar10_test_dset = torchvision.datasets.CIFAR10('./cifar10_pytorch', train=False, transform=transform_test, download=True)

    # For Validation set
    val_index = np.random.randint(10)
    val_samples = list(range( 5000 * val_index, 5000 * (val_index + 1) ))
    
    #train_loader = torch.utils.data.DataLoader(cifar10_train_dset, batch_size=mbs, sampler = torch.utils.data.SubsetRandomSampler(val_samples), shuffle=False, num_workers=1)
    train_loader = torch.utils.data.DataLoader(cifar10_train_dset, batch_size=mbs, shuffle=True, num_workers=1)
    test_loader = torch.utils.data.DataLoader(cifar10_test_dset, batch_size=200, shuffle=False, num_workers=1)


if args.act=='mysig':
    activation = my_sigmoid
elif args.act=='sigmoid':
    activation = torch.sigmoid
elif args.act=='tanh':
    activation = torch.tanh
elif args.act=='hard_sigmoid':
    activation = hard_sigmoid
elif args.act=='my_hard_sig':
    activation = my_hard_sig
elif args.act=='ctrd_hard_sig':
    activation = ctrd_hard_sig



criterion = torch.nn.MSELoss(reduction='none')

if args.model=='MLP':
    model = P_MLP(args.archi, activation=activation)
elif args.model=='CNN':

    if args.task=='MNIST':
        pools = [torch.nn.MaxPool2d(2, stride=2), torch.nn.MaxPoll2d(2, stride=2)]
        model = P_CNN(28, [1, 32, 64], [5, 5], [1, 1], [10], pools, activation=activation)

    elif args.task=='CIFAR10':    
        if args.pool=='max':
            pools = [torch.nn.MaxPool2d(2, stride=2), torch.nn.MaxPool2d(2, stride=2), torch.nn.Identity()] 
            strides = [1,1,1]
        elif args.pool=='avg':
            pools = [torch.nn.AvgPool2d(2, stride=2), torch.nn.AvgPool2d(2, stride=2), torch.nn.Identity()]
            strides = [1,1,1]
        elif args.pool=='id':
            pools = [torch.nn.Identity(), torch.nn.Identity(), torch.nn.Identity()]
            strides = [2,2,1]
        model = P_CNN(32, [3, 64, 128, 256], [5, 5, 3], strides, [1024, 10], pools, activation=activation)
    
    print('Poolings =', model.pools)

print('\n')
print(model)
model.to(device)


betas = args.betas[0], args.betas[1]

if args.todo=='train':
    assert(len(args.lrs)==len(model.synapses))
    optim_params = []
    for idx in range(len(model.synapses)):
        optim_params.append(  {'params': model.synapses[idx].parameters(), 'lr': args.lrs[idx]}  )
    if args.optim=='sgd':
        optimizer = torch.optim.SGD( optim_params )
    elif args.optim=='adam':
        optimizer = torch.optim.Adam( optim_params )

    print(optimizer)
    train(model, optimizer, train_loader, test_loader, args.T1, args.T2, betas, device, epochs=args.epochs, criterion=criterion, check_thm=args.check_thm, save=args.save, path=path)

elif args.todo=='gducheck':

    dataiter = iter(train_loader)
    images, labels = dataiter.next()
    images, labels = images.to(device), labels.to(device)

    BPTT, EP = check_gdu(model, images[0:3,:], labels[0:3], args.T1, args.T2, betas, torch.nn.MSELoss(reduction='none'))
    RMSE(BPTT, EP)
    if args.save:
        plot_gdu(BPTT, EP, path)
