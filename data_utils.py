import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')

import os
from datetime import datetime
import time
import math




def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)





def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return 'elapsed time : %s \t (will finish in %s)' % (asMinutes(s), asMinutes(rs))

        
def plot_gdu(BPTT, EP, path):
    N = len(EP.keys())
    fig = plt.figure(figsize=(10,2*N))
    for idx, key in enumerate(EP.keys()):
        fig.add_subplot(N//2+1, 2, idx+1)
        if len(EP[key].size())==3:
            i, j = np.random.randint(EP[key].size(1)), np.random.randint(EP[key].size(2))
            ep = EP[key][:,i,j].cpu().detach().numpy().flatten()
            bptt = BPTT[key][:,i,j].cpu().detach().numpy().flatten()
            plt.plot(ep, label='ep')
            plt.plot(bptt, label='bptt')
            plt.title(key.replace('.','_'))
            plt.legend()
        elif len(EP[key].size())==2:
            i = np.random.randint(EP[key].size(1))
            ep = EP[key][:,i].cpu().detach().numpy().flatten()
            bptt = BPTT[key][:,i].cpu().detach().numpy().flatten()
            plt.plot(ep, label='ep')
            plt.plot(bptt, label='bptt')
            plt.title(key.replace('.','_'))
            plt.legend()
        elif len(EP[key].size())==5:
            i, j = np.random.randint(EP[key].size(1)), np.random.randint(EP[key].size(2))
            k, l = np.random.randint(EP[key].size(3)), np.random.randint(EP[key].size(4))
            ep = EP[key][:,i,j,k,l].cpu().detach().numpy().flatten()
            bptt = BPTT[key][:,i,j,k,l].cpu().detach().numpy().flatten()
            plt.plot(ep, label='ep')
            plt.plot(bptt, label='bptt')
            plt.title(key.replace('.','_'))
            plt.legend()
    fig.savefig(path + '/some_gdu_curves.png')
    plt.close()        





        
def plot_neural_activity(neurons, path):   
    N = len(neurons)
    fig = plt.figure(figsize=(3*N,6))
    for idx in range(N):
        fig.add_subplot(2, N//2+1, idx+1)
        nrn = neurons[idx].cpu().detach().numpy().flatten()
        plt.hist(nrn, 50)
        #plt.xlim((-1.1,1.1))
        plt.title('neurons of layer '+str(idx+1))
    fig.savefig(path)
    plt.close()




    
def plot_synapses(model, path):   
    N = len(model.synapses)
    fig = plt.figure(figsize=(4*N,3))
    for idx in range(N):
        fig.add_subplot(1, N, idx+1)
        nrn = model.synapses[idx].weight.cpu().detach().numpy().flatten()
        plt.hist(nrn, 50)
        plt.title('synapses of layer '+str(idx+1))
    fig.savefig(path)
    plt.close()





def plot_acc(train_acc, test_acc, path):
    fig = plt.figure(figsize=(16,9))
    x_axis = [i for i in range(len(train_acc))]
    plt.plot(x_axis, train_acc, label='train')
    plt.plot(x_axis, test_acc, label='test')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    fig.savefig(path + '/train-test_acc.png')
    plt.close()
 


def createHyperparametersFile(path, args, model):

    hyperparameters = open(path + r"/hyperparameters.txt","w+")
    L = ["- task: {}".format(args.task) + "\n",
        "- data augmentation (if CIFAR10): {}".format(args.data_aug) + "\n",
        "- learning rate decay: {}".format(args.lr_decay) + "\n",
        "- activation: {}".format(args.act) + "\n",
        "- learning rates: {}".format(args.lrs) + "\n",
        "- weight decays: {}".format(args.wds) + "\n",
        "- optimizer: {}".format(args.optim) + "\n",
        "- loss: {}".format(args.loss) + "\n",
        "- alg: {}".format(args.alg) + "\n",
        "- minibatch size: {}".format(args.mbs) + "\n",
        "- T1: {}".format(args.T1) + "\n",
        "- T2: {}".format(args.T2) + "\n", 
        "- betas: {}".format(args.betas) + "\n", 
        "- random beta_2 sign: {}".format(args.random_sign) + "\n", 
        "- thirdphase: {}".format(args.thirdphase) + "\n", 
        "- softmax: {}".format(args.softmax) + "\n", 
        "- epochs: {}".format(args.epochs) + "\n", 
        "- seed: {}".format(args.seed) + "\n", 
        "- device: {}".format(args.device) + "\n"]
   
    hyperparameters.writelines(L)
    print('\nPoolings :', model.pools, '\n', file=hyperparameters)
    print(model, file=hyperparameters)

    hyperparameters.close()








 
