import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torch.nn.functional as F

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


def my_sigmoid(x):
    return 1/(1+torch.exp(-4*(x-0.5)))

def hard_sigmoid(x):
    return (1+F.hardtanh(x-1))*0.5

def ctrd_hard_sig(x):
    return (1+F.hardtanh(x))*0.5

def my_hard_sig(x):
    return 0.5*(my_sigmoid(x)+hard_sigmoid(x))





def grad_or_zero(x):
    if x.grad is None:
        return torch.zeros_like(x).to(x.device)
    else:
        return x.grad

def neurons_zero_grad(neurons):
    for idx in range(len(neurons)):
        if neurons[idx].grad is not None:
            neurons[idx].grad.zero_()

def copy(neurons):
    copy = []
    for n in neurons:
        copy.append(torch.empty_like(n).copy_(n.data).requires_grad_())
    return copy


                
                
# Multi-Layer Perceptron

class P_MLP(torch.nn.Module):
    def __init__(self, archi, activation=torch.tanh):
        super(P_MLP, self).__init__()
        
        self.activation = activation
        self.archi = archi
        
        # Synapses
        self.synapses = torch.nn.ModuleList()
        for idx in range(len(archi)-1):
            self.synapses.append(torch.nn.Linear(archi[idx], archi[idx+1], bias=True))

            
    def Phi(self, x, y, neurons, beta, criterion):
        
        x = x.view(x.size(0),-1)
        y = F.one_hot(y, num_classes=10).double()
        
        layers = [x] + neurons
        
        phi = 0.0
        for idx in range(len(self.synapses)):
            phi += torch.sum( self.synapses[idx](layers[idx]) * layers[idx+1], dim=1).squeeze()
        
        L = 0.5*criterion(layers[-1].double(), y).sum(dim=1).squeeze()     
        phi -= beta*L
        
        return phi
    
    
    def forward(self, x, y, neurons, T, beta=0.0, criterion=torch.nn.MSELoss(reduction='none'), check_thm=False):
        
        mbs = x.size(0)
        
        if not(check_thm):
            
            for t in range(T):
            
                neurons_zero_grad(neurons)
                phi = self.Phi(x, y, neurons, beta=beta, criterion=criterion)
                phi.backward(torch.tensor([1 for i in range(mbs)], dtype=torch.float, device=x.device, requires_grad=True)) 

                for idx in range(len(neurons)):
                    neurons[idx] = self.activation( neurons[idx].grad )
                    neurons[idx].requires_grad = True
            
            return neurons
            
        else:
            neurons_hist = []
            neurons_hist.append(neurons)
            for t in range(T):
                    
                neurons_zero_grad(neurons_hist[-1])
                phi = self.Phi(x, y, neurons_hist[-1], beta=beta, criterion=criterion)
                phi.backward(torch.tensor([1 for i in range(mbs)], dtype=torch.float, device=x.device, requires_grad=True), create_graph=True, retain_graph=True) 
                
                new_neurons = []
                for idx in range(len(neurons)):
                    new_neurons.append(self.activation( neurons_hist[-1][idx].grad.clone() ))
                    new_neurons[-1].retain_grad()
                
                neurons_hist.append(new_neurons)
                  
            return neurons_hist


    def init_neurons(self, mbs, device):
        
        neurons = []
        for size in self.archi[1:]:  
            neurons.append(torch.zeros((mbs, size), requires_grad=True, device=device))
        return neurons


    def compute_syn_grads(self, x, y, neurons_1, neurons_2, betas, criterion, check_thm=False):
        
        beta_1, beta_2 = betas
        
        self.zero_grad()            # p.grad is zero
        if not(check_thm):
            phi_1 = self.Phi(x, y, neurons_1, beta_1, criterion)
        else:
            phi_1 = self.Phi(x, y, neurons_1, beta_2, criterion)
        
        phi_1 = phi_1.mean()
        phi_1.backward()            # p.grad =  d_Phi_1/dp
        self.minus_grad()           # p.grad = -d_Phi_1/dp
        
        phi_2 = self.Phi(x, y, neurons_2, beta_2, criterion)
        phi_2 = phi_2.mean()
        phi_2.backward()            # p.grad = d_Phi_2/dp - d_Phi_1/dp
        
        for p in self.parameters():
            if p.grad is not None:
                p.grad.data.div_(beta_1 - beta_2)    # p.grad = -(d_Phi_2/dp - d_Phi_1/dp)/(beta_2 - beta_1) ----> dL/dp  by the theorem
        
        
    def minus_grad(self):
        for p in self.parameters():
            if p.grad is not None:
                p.grad.data.mul_(-1)
    

    
    
    
# Convolutional Neural Network

class P_CNN(torch.nn.Module):
    def __init__(self, in_size, channels, kernels, strides, fc, pools, activation=hard_sigmoid):
        super(P_CNN, self).__init__()

        # Dimensions used to initialize neurons
        self.in_size = in_size
        self.channels = channels
        self.kernels = kernels
        self.strides = strides
        self.fc = fc
        
        self.activation = activation
        self.pools = pools
        
        self.synapses = torch.nn.ModuleList()
        #self.conv_bias = []
        
        size = in_size
        for idx in range(len(channels)-1): 
            self.synapses.append(torch.nn.Conv2d(channels[idx], channels[idx+1], kernels[idx], 
                                                 stride=strides[idx], bias=True))
            
            size = int( (size - kernels[idx])/strides[idx] + 1 )          # size after conv
            if self.pools[idx].__class__.__name__.find('Pool')!=-1:
                size = int( (size - pools[idx].kernel_size)/pools[idx].stride + 1 )   # size after Pool
            
            #self.conv_bias.append( torch.nn.Parameter(torch.zeros((channels[idx+1], 1, 1), requires_grad=True, device=device)) ) 
            
        size = size * size * channels[-1]
        
        fc_layers = [size] + fc
        for idx in range(len(fc)):
            self.synapses.append(torch.nn.Linear(fc_layers[idx], fc_layers[idx+1], bias=True))
        


    def Phi(self, x, y, neurons, beta, criterion):

        mbs = x.size(0)
        y = F.one_hot(y, num_classes=10).double()
        
        layers = [x] + neurons
        
        phi = 0.0
        for idx in range(len(self.synapses)):
            if self.synapses[idx].__class__.__name__.find('Conv')!=-1:
                phi += torch.sum( self.pools[idx](self.synapses[idx](layers[idx])) * layers[idx+1], dim=(1,2,3)).squeeze()      #phi += torch.sum( self.synapses[idx](layers[idx]) * layers[idx+1], dim=(1,2,3)).squeeze()
                #phi += torch.sum( self.conv_bias[idx] * layers[idx+1], dim=(1,2,3)).squeeze()
            else:
                phi += torch.sum( self.synapses[idx](layers[idx].view(mbs,-1)) * layers[idx+1], dim=1).squeeze()
                    
        L = 0.5*criterion(layers[-1].double(), y).sum(dim=1).squeeze()     
        phi -= beta*L
        
        return phi
    

    def forward(self, x, y, neurons, T, beta=0.0, criterion=torch.nn.MSELoss(reduction='none'), check_thm=False):
        
        mbs = x.size(0)
        
        if not(check_thm):
            
            for t in range(T):
            
                neurons_zero_grad(neurons)
                phi = self.Phi(x, y, neurons, beta=beta, criterion=criterion)
                phi.backward(torch.tensor([1 for i in range(mbs)], dtype=torch.float, device=x.device, requires_grad=True)) 

                for idx in range(len(neurons)):
                    neurons[idx] = self.activation( neurons[idx].grad )
                    neurons[idx].requires_grad = True
            
            return neurons
            
        else:
            neurons_hist = []
            neurons_hist.append(neurons)
            for t in range(T):
                    
                neurons_zero_grad(neurons_hist[-1])
                phi = self.Phi(x, y, neurons_hist[-1], beta=beta, criterion=criterion)
                phi.backward(torch.tensor([1 for i in range(mbs)], dtype=torch.float, device=x.device, requires_grad=True), create_graph=True, retain_graph=True) 
                
                new_neurons = []
                for idx in range(len(neurons)):
                    new_neurons.append(self.activation( neurons_hist[-1][idx].grad.clone() ))
                    new_neurons[-1].retain_grad()
                
                neurons_hist.append(new_neurons)
                  
            return neurons_hist


    def init_neurons(self, mbs, device):
        
        neurons = []
        
        size = self.in_size
        for idx in range(len(self.channels)-1): 
            size = int( (size - self.kernels[idx])/self.strides[idx] + 1 )                     # size after conv
            if self.pools[idx].__class__.__name__.find('Pool')!=-1:
                size = int( (size - self.pools[idx].kernel_size)/self.pools[idx].stride + 1 )  # size after Pool
            neurons.append(torch.zeros((mbs, self.channels[idx+1], size, size), requires_grad=True, device=device))

        size = size * size * self.channels[-1]
        
        for idx in range(len(self.fc)):
            neurons.append(torch.zeros((mbs, self.fc[idx]), requires_grad=True, device=device))
        
        return neurons

    def compute_syn_grads(self, x, y, neurons_1, neurons_2, betas, criterion, check_thm=False):
        
        beta_1, beta_2 = betas
        
        self.zero_grad()            # p.grad = 0
        if not(check_thm):
            phi_1 = self.Phi(x, y, neurons_1, beta_1, criterion)
        else:
            phi_1 = self.Phi(x, y, neurons_1, beta_2, criterion)
        
        phi_1 = phi_1.mean()
        phi_1.backward()            # p.grad =  d_Phi_1/dp
        self.minus_grad()           # p.grad = -d_Phi_1/dp
        
        phi_2 = self.Phi(x, y, neurons_2, beta_2, criterion)
        phi_2 = phi_2.mean()
        phi_2.backward()            # p.grad = d_Phi_2/dp - d_Phi_1/dp
        
        for p in self.parameters():
            if p.grad is not None:
                p.grad.data.div_(beta_1 - beta_2)  # p.grad = -(d_Phi_2/dp - d_Phi_1/dp)/(beta_2 - beta_1) ---> dL/dp  by the theorem
        
        
    def minus_grad(self):
        for p in self.parameters():
            if p.grad is not None:
                p.grad.data.mul_(-1)

        
    
    
    
    


    


def check_gdu(model, x, y, T1, T2, betas, criterion):
    
    # Initialize dictionnaries that will contain BPTT gradients and EP updates
    BPTT, EP = {}, {}
    for name, p in model.named_parameters():
        BPTT[name], EP[name] = [], []
        
    neurons = model.init_neurons(x.size(0), x.device)
    for idx in range(len(neurons)):
        BPTT['neurons_'+str(idx)], EP['neurons_'+str(idx)] = [], []
    
    
    # First phase up to T1-T2
    beta_1, beta_2 = betas
    neurons = model(x, y, neurons, T1-T2, beta=beta_1, criterion=criterion)
    ref_neurons = copy(neurons)
    
    
    # Last steps of the first phase
    for K in range(T2+1):
        
        neurons = model(x, y, neurons, K, beta=beta_1, criterion=criterion) # Running K time step 
        
        # detach data and neurons from the graph
        x = x.detach()
        x.requires_grad = True
        for idx in range(len(neurons)):
            neurons[idx] = neurons[idx].detach()
            neurons[idx].requires_grad = True
        
        neurons_hist = model(x, y, neurons, T2-K, beta=beta_1, criterion=criterion, check_thm=True) # T2-K time step
              
        # final loss
        loss = (1/(2.0*x.size(0)))*criterion(neurons_hist[-1][-1].double(), F.one_hot(y, num_classes=10).double()).sum(dim=1).squeeze()
        
        # setting gradients field to zero before backward
        for idx in range(len(neurons_hist)):
            neurons_zero_grad(neurons_hist[idx])
        model.zero_grad()
        
        # Backpropagation through time
        loss.backward(torch.tensor([1 for i in range(x.size(0))], dtype=torch.float, device=x.device, requires_grad=True))
        
        # Collecting BPTT gradients : for parameters they are partial sums over T2-K time steps, 
        # but for neurons they are all obtained for K==0
        if K!=T2:
            for name, p in model.named_parameters():
                update = torch.empty_like(p).copy_(grad_or_zero(p))
                BPTT[name].append( update.unsqueeze(0) )  # unsqueeze for time dimension
            
            if K==0:
                for idx in range(len(neurons)):
                    for k in range(1,len(neurons_hist)):
                        update = torch.empty_like(neurons_hist[k][idx]).copy_(grad_or_zero(neurons_hist[k][idx]))
                        BPTT['neurons_'+str(idx)].append( update.mul(-x.size(0)).unsqueeze(0) )  # unsqueeze for time dimension
        
            neurons = copy(ref_neurons) # Resetting the neurons to T1-T2 step
        else:
            neurons = neurons_hist[0]   # In this case all BPTT gradients have been collected and neurons are set to the neurons at time step T2

    # Differentiating partial sums to get elementary parameter gradients
    for name, p in model.named_parameters():
        for idx in range(len(BPTT[name]) - 1):
            BPTT[name][idx] = BPTT[name][idx] - BPTT[name][idx+1]
        
    # Reverse the time
    for key in BPTT.keys():
        BPTT[key].reverse()
            
    # Second phase done step by step
    for t in range(T2):
        neurons_pre = copy(neurons)                                          # neurons at time step t
        neurons = model(x, y, neurons, 1, beta=beta_2, criterion=criterion)  # neurons at time step t+1
        
        model.compute_syn_grads(x, y, neurons_pre, neurons, betas, criterion, check_thm=True)  # compute the EP parameter update
        
        # Collect the EP updates forward in time
        for n, p in model.named_parameters():
            update = torch.empty_like(p).copy_(grad_or_zero(p))
            EP[n].append( update.unsqueeze(0) )                    # unsqueeze for time dimension
        for idx in range(len(neurons)):
            update = (neurons[idx] - neurons_pre[idx])/(beta_2 - beta_1)
            EP['neurons_'+str(idx)].append( update.unsqueeze(0) )  # unsqueeze for time dimension
        
    # Concatenating with respect to time dimension
    for key in BPTT.keys():
        BPTT[key] = torch.cat(BPTT[key], dim=0)
        EP[key] = torch.cat(EP[key], dim=0)
        
    return BPTT, EP





def RMSE(BPTT, EP):
    print('\nGDU check :')
    for key in BPTT.keys():
        K = BPTT[key].size(0)
        f_g = (EP[key] - BPTT[key]).pow(2).sum(dim=0).div(K).pow(0.5)
        f =  EP[key].pow(2).sum(dim=0).div(K).pow(0.5)
        g = BPTT[key].pow(2).sum(dim=0).div(K).pow(0.5)
        comp = f_g/(1e-10+torch.max(f,g))
        sign = torch.where(EP[key].sign() != BPTT[key].sign(), torch.ones_like(EP[key]), torch.zeros_like(EP[key]))
        sign = sign.sum().item()/sign.numel()
        print(key.replace('.','_'), '\t RMSE =', round(comp.mean().item(), 4), '\t SIGN err =', round(sign, 4))
    print('\n')

    

        
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
    #plt.show()
    plt.close()        
        
def plot_neural_activity(neurons, path):   
    N = len(neurons)
    fig = plt.figure(figsize=(4*N,3))
    for idx in range(N):
        fig.add_subplot(1, N, idx+1)
        nrn = neurons[idx].cpu().detach().numpy().flatten()
        plt.hist(nrn, 50)
        plt.xlim((-1.1,1.1))
        plt.title('neurons of layer '+str(idx+1))
    fig.savefig(path)
    #plt.show()
    plt.close()
    

def plot_synapses(model):   
    N = len(model.synapses)
    fig = plt.figure(figsize=(4*N,3))
    for idx in range(N):
        fig.add_subplot(1, N, idx+1)
        nrn = model.synapses[idx].weight.cpu().detach().numpy().flatten()
        plt.hist(nrn, 50)
        plt.title('synapses of layer '+str(idx+1))
    #plt.show()



    
    

        
def train(model, optimizer, train_loader, test_loader, T1, T2, betas, device, epochs=1, criterion=torch.nn.MSELoss(reduction='none'), check_thm=False, path=''):
    
    model.train()
    mbs = train_loader.batch_size
    start = time.time()
    iter_per_epochs = len(train_loader.dataset)//mbs
    beta_1, beta_2 = betas
    
    best = 0.0
    train_acc = [0.1]
    
    for epoch in range(epochs):
        run_correct = 0
        run_total = 0
        
        for idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            
            # First phase
            neurons = model.init_neurons(x.size(0), device)
            neurons = model(x, y, neurons, T1, beta=beta_1, criterion=criterion)
            neurons_1 = copy(neurons)
            
            # Predictions for running accuracy
            with torch.no_grad():
                pred = torch.argmax(neurons_1[-1], dim=1).squeeze()
                run_correct += (y == pred).sum().item()
                run_total += x.size(0)
                if (idx%(iter_per_epochs//10)==0) or (idx==iter_per_epochs-1):
                    plot_neural_activity(neurons_1, path + '/ep-'+str(epoch+1)+'_iter-'+str(idx+1)+'_neural_activity.png')
            
            # Second phase
            neurons = model(x, y, neurons, T2, beta=beta_2, criterion=criterion)
            neurons_2 = copy(neurons)
            
            model.compute_syn_grads(x, y, neurons_1, neurons_2, betas, criterion)
            
            optimizer.step()
            
            if (idx%(iter_per_epochs//10)==0) or (idx==iter_per_epochs-1):
                run_acc = run_correct/run_total
                print('Epoch :', round(epoch+(idx/iter_per_epochs), 2),
                      '\tRun train acc :', round(run_acc,3),'\t('+str(run_correct)+'/'+str(run_total)+')\t',
                      timeSince(start, ((idx+1)+epoch*iter_per_epochs)/(epochs*iter_per_epochs)))
                
                if check_thm:
                    BPTT, EP = check_gdu(model, x[0:5,:], y[0:5], T1, T2, betas, criterion)
                    RMSE(BPTT, EP)
    
                run_correct = 0
                run_total = 0
        
        train_acc.append(run_acc)
        fig = plt.figure(figsize=(16,9))
        plt.plot(train_acc)
        fig.savefig(path + '/train_acc.png')
        plt.close()
        test_acc = evaluate(model, test_loader, T1, device)
        if test_acc > best:
            torch.save({'model_state_dict': model.state_dict(), 'opt': optimizer.state_dict() },  path + '/checkpoint.tar')
            torch.save(model, path + '/model.pt')
            best = test_acc
            
            
def evaluate(model, loader, T, device):
    
    model.eval()
    correct=0
    phase = 'Train' if loader.dataset.train else 'Test'
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        neurons = model.init_neurons(x.size(0), device)
        neurons = model(x, y, neurons, T)
        pred = torch.argmax(neurons[-1], dim=1).squeeze()
        correct += (y == pred).sum().item()

    acc = correct/len(loader.dataset) 
    print(phase+' accuracy :\t', acc)   
    return correct


            
