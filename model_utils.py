import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torch.nn.functional as F

import os
from datetime import datetime
import time
import math
from data_utils import *

from itertools import repeat
from torch.nn.parameter import Parameter
import collections
import matplotlib
matplotlib.use('Agg')



def my_sigmoid(x):
    return 1/(1+torch.exp(-4*(x-0.5)))

def hard_sigmoid(x):
    return (1+F.hardtanh(x-1))*0.5

def ctrd_hard_sig(x):
    return (1+F.hardtanh(x))*0.5

def my_hard_sig(x):
    return 0.5*(my_sigmoid(x)+hard_sigmoid(x))



def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)


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

def make_pools(letters):
    pools = []
    for p in range(len(letters)):
        if letters[p]=='m':
            pools.append( torch.nn.MaxPool2d(2, stride=2) )
        elif letters[p]=='a':
            pools.append( torch.nn.AvgPool2d(2, stride=2) )
        elif letters[p]=='i':
            pools.append( torch.nn.Identity() )
    return pools
                
                
# Multi-Layer Perceptron

class P_MLP(torch.nn.Module):
    def __init__(self, archi, activation=torch.tanh):
        super(P_MLP, self).__init__()
        
        self.activation = activation
        self.archi = archi
        self.softmax = False        

        # Synapses
        self.synapses = torch.nn.ModuleList()
        for idx in range(len(archi)-1):
            self.synapses.append(torch.nn.Linear(archi[idx], archi[idx+1], bias=True))

            
    def Phi(self, x, y, neurons, beta, criterion):
        
        x = x.view(x.size(0),-1)
        
        layers = [x] + neurons
        
        phi = 0.0
        for idx in range(len(self.synapses)):
            phi += torch.sum( self.synapses[idx](layers[idx]) * layers[idx+1], dim=1).squeeze()
        
        if beta!=0.0:
            if criterion.__class__.__name__.find('MSE')!=-1:
                y = F.one_hot(y, num_classes=10).double()
                L = 0.5*criterion(layers[-1].double(), y).sum(dim=1).squeeze()   
            else:
                L = criterion(layers[-1].double(), y).squeeze()     
            phi -= beta*L
        
        return phi
    
    
    def forward(self, x, y, neurons, T, beta=0.0, criterion=torch.nn.MSELoss(reduction='none'), check_thm=False):
        
        not_mse = (criterion.__class__.__name__.find('MSE')==-1)
        mbs = x.size(0)
        device = x.device

        for t in range(T):
            phi = self.Phi(x, y, neurons, beta, criterion)
            init_grads = torch.tensor([1 for i in range(mbs)], dtype=torch.float, device=device, requires_grad=True)
            grads = torch.autograd.grad(phi, neurons, grad_outputs=init_grads, create_graph=check_thm)

            for idx in range(len(neurons)-1):
                neurons[idx] = self.activation( grads[idx] )
                if check_thm:
                    neurons[idx].retain_grad()
                else:
                    neurons[idx].requires_grad = True
             
            if not_mse:
                neurons[-1] = grads[-1]
            else:
                neurons[-1] = self.activation( grads[-1] )

            if check_thm:
                neurons[-1].retain_grad()
            else:
                neurons[-1].requires_grad = True

        return neurons


    def init_neurons(self, mbs, device):
        
        neurons = []
        append = neurons.append
        for size in self.archi[1:]:  
            append(torch.zeros((mbs, size), requires_grad=True, device=device))
        return neurons


    def compute_syn_grads(self, x, y, neurons_1, neurons_2, betas, criterion, check_thm=False):
        
        beta_1, beta_2 = betas
        
        self.zero_grad()            # p.grad is zero
        if not(check_thm):
            phi_1 = self.Phi(x, y, neurons_1, beta_1, criterion)
        else:
            phi_1 = self.Phi(x, y, neurons_1, beta_2, criterion)
        phi_1 = phi_1.mean()
        
        phi_2 = self.Phi(x, y, neurons_2, beta_2, criterion)
        phi_2 = phi_2.mean()
        
        delta_phi = (phi_2 - phi_1)/(beta_1 - beta_2)        
        delta_phi.backward() # p.grad = -(d_Phi_2/dp - d_Phi_1/dp)/(beta_2 - beta_1) ----> dL/dp  by the theorem
        

         
# Vector Field Multi-Layer Perceptron

class VF_MLP(torch.nn.Module):
    def __init__(self, archi, activation=torch.tanh):
        super(VF_MLP, self).__init__()
        
        self.activation = activation
        self.archi = archi
        self.softmax = False        

        # Forward synapses
        self.synapses = torch.nn.ModuleList()
        for idx in range(len(archi)-1):
            self.synapses.append(torch.nn.Linear(archi[idx], archi[idx+1], bias=True))

        # Backward synapses
        self.B_syn = torch.nn.ModuleList()
        for idx in range(1, len(archi)-1):            
            self.B_syn.append(torch.nn.Linear(archi[idx+1], archi[idx], bias=False))


    def Phi(self, x, y, neurons, beta, criterion):
        
        x = x.view(x.size(0),-1)
        
        layers = [x] + neurons
        
        phis = []
        for idx in range(len(self.synapses)-1):
            phi = torch.sum( self.synapses[idx](layers[idx]) * layers[idx+1], dim=1).squeeze()
            phi += torch.sum( self.B_syn[idx](layers[idx+2]) * layers[idx+1], dim=1).squeeze()
            phis.append(phi)        

        phi = torch.sum( self.synapses[-1](layers[-2]) * layers[-1], dim=1).squeeze()
        if beta!=0.0:
            if criterion.__class__.__name__.find('MSE')!=-1:
                y = F.one_hot(y, num_classes=10).double()
                L = 0.5*criterion(layers[-1].double(), y).sum(dim=1).squeeze()   
            else:
                L = criterion(layers[-1].double(), y).squeeze()     
            phi -= beta*L
        
        phis.append(phi)

        return phis
    
    
    def forward(self, x, y, neurons, T, beta=0.0, criterion=torch.nn.MSELoss(reduction='none'), check_thm=False):
        
        not_mse = (criterion.__class__.__name__.find('MSE')==-1)
        mbs = x.size(0)
        device = x.device

        for t in range(T):
            phis = self.Phi(x, y, neurons, beta, criterion)
            for idx in range(len(neurons)-1):
                init_grad = torch.tensor([1 for i in range(mbs)], dtype=torch.float, device=device, requires_grad=True)
                grad = torch.autograd.grad(phis[idx], neurons[idx], grad_outputs=init_grad, create_graph=check_thm)

                neurons[idx] = self.activation( grad[0] )
                if check_thm:
                    neurons[idx].retain_grad()
                else:
                    neurons[idx].requires_grad = True
             
            init_grad = torch.tensor([1 for i in range(mbs)], dtype=torch.float, device=device, requires_grad=True)
            grad = torch.autograd.grad(phis[-1], neurons[-1], grad_outputs=init_grad, create_graph=check_thm)
            if not_mse:
                neurons[-1] = grad[0]
            else:
                neurons[-1] = self.activation( grad[0] )

            if check_thm:
                neurons[-1].retain_grad()
            else:
                neurons[-1].requires_grad = True

        return neurons


    def init_neurons(self, mbs, device):
        
        neurons = []
        append = neurons.append
        for size in self.archi[1:]:  
            append(torch.zeros((mbs, size), requires_grad=True, device=device))
        return neurons


    def compute_syn_grads(self, x, y, neurons_1, neurons_2, betas, criterion, check_thm=False):
        
        beta_1, beta_2 = betas
        
        self.zero_grad()            # p.grad is zero
        if not(check_thm):
            phis_1 = self.Phi(x, y, neurons_1, beta_1, criterion)
        else:
            phis_1 = self.Phi(x, y, neurons_1, beta_2, criterion)
        
        phis_2 = self.Phi(x, y, neurons_2, beta_2, criterion)
     
        for idx in range(len(neurons_1)):
            phi_1 = phis_1[idx].mean()
            phi_2 = phis_2[idx].mean()
            delta_phi = (phi_2 - phi_1)/(beta_1 - beta_2)        
            delta_phi.backward() # p.grad = -(d_Phi_2/dp - d_Phi_1/dp)/(beta_2 - beta_1)
        


#Locally connected layer
class Conv2dLocal(torch.nn.Module):
 
    def __init__(self, in_height, in_width, in_channels, out_channels,
                 kernel_size, stride=1, padding=0, bias=True, dilation=1):
        super(Conv2dLocal, self).__init__()
 
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
 
        self.in_height = in_height
        self.in_width = in_width
        self.out_height = int(math.floor(
            (in_height + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1))
        self.out_width = int(math.floor(
            (in_width + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1))
        self.weight = Parameter(torch.Tensor(
            self.out_height, self.out_width,
            out_channels, in_channels, *self.kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(
                out_channels, self.out_height, self.out_width))
        else:
            self.register_parameter('bias', None)
 
        self.reset_parameters()
 
    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
 
    def __repr__(self):
        s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)
 
    def forward(self, input):
        return self.conv2d_local(
            input, self.weight, bias = self.bias, stride=self.stride,
            padding=self.padding, dilation=self.dilation)


    def conv2d_local(self, input, weight, bias=None, padding=0, stride=1, dilation=1):

        outH, outW, outC, inC, kH, kW = weight.size()
        kernel_size = (kH, kW)
     
        # N x [inC * kH * kW] x [outH * outW]
        cols = F.unfold(input, kernel_size, dilation=dilation, padding=padding, stride=stride)
        cols = cols.view(cols.size(0), cols.size(1), cols.size(2), 1).permute(0, 2, 3, 1)
     
        out = torch.matmul(cols, weight.view(outH * outW, outC, inC * kH * kW).permute(0, 2, 1))
        out = out.view(cols.size(0), outH, outW, outC).permute(0, 3, 1, 2)
     
        if bias is not None:
            out = out + bias.expand_as(out)
        return out

    
    
    
# Convolutional Neural Network

class P_CNN(torch.nn.Module):
    def __init__(self, in_size, channels, kernels, strides, fc, pools, activation=hard_sigmoid, local = False, softmax = False):
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
        

        self.softmax = softmax

        """
        softmax is a boolean function which tells whether we use the parametrized implementation 
        of the softmax prediction (section 2.9 of overleaf document). In this case, the last layer
        is *NO LONGER* part of the system: it is *ONLY* used for prediction and it does not interact
        with the rest of the system, except during nudging through beta*l. So the Phi function is
        computed until the *PENULTIMATE* layer. 

        This option affects:
        - Phi method
        - init_neurons method
        - train function
        """

        size = in_size

        if not local:
            for idx in range(len(channels)-1): 
                self.synapses.append(torch.nn.Conv2d(channels[idx], channels[idx+1], kernels[idx], 
                                                     stride=strides[idx], bias=True))
                
                size = int( (size - kernels[idx])/strides[idx] + 1 )          # size after conv
                if self.pools[idx].__class__.__name__.find('Pool')!=-1:
                    size = int( (size - pools[idx].kernel_size)/pools[idx].stride + 1 )   # size after Pool

        else:
            #input height and width (CIFAR-10)
            in_height = 32
            in_width = 32
            for idx in range(len(channels)-1): 

                self.synapses.append(Conv2dLocal(in_height, in_width, channels[idx], channels[idx+1], kernels[idx], 
                                                     stride=strides[idx], bias=True))
               
                #update input height and width (after convolution)

                in_height = self.synapses[idx].out_height
                in_width = self.synapses[idx].out_width

                if self.pools[idx].__class__.__name__.find('Pool')!=-1:
                    in_height = int( (in_height - pools[idx].kernel_size)/pools[idx].stride + 1 )   # height after Pool
                    in_width = int( (in_width - pools[idx].kernel_size)/pools[idx].stride + 1 )   # width after Pool

            size = in_height             

            
        size = size * size * channels[-1]
        
        fc_layers = [size] + fc

        for idx in range(len(fc)):
            self.synapses.append(torch.nn.Linear(fc_layers[idx], fc_layers[idx+1], bias=True))
        


    def Phi(self, x, y, neurons, beta, criterion):

        mbs = x.size(0)       
        conv_len = len(self.kernels)
        tot_len = len(self.synapses)

        layers = [x] + neurons        
        phi = 0.0

        #Phi computation changes depending on softmax == True or not
        if not self.softmax:
            for idx in range(conv_len):    
                phi += torch.sum( self.pools[idx](self.synapses[idx](layers[idx])) * layers[idx+1], dim=(1,2,3)).squeeze()     
                #phi += torch.sum( self.conv_bias[idx] * layers[idx+1], dim=(1,2,3)).squeeze()
            for idx in range(conv_len, tot_len):
                phi += torch.sum( self.synapses[idx](layers[idx].view(mbs,-1)) * layers[idx+1], dim=1).squeeze()
             
            if beta!=0.0:
                if criterion.__class__.__name__.find('MSE')!=-1:
                    y = F.one_hot(y, num_classes=10).double()
                    L = 0.5*criterion(layers[-1].double(), y).sum(dim=1).squeeze()   
                else:
                    L = criterion(layers[-1].double(), y).squeeze()             
                phi -= beta*L

        else:
            #WATCH OUT: the output layer used for the prediction is no longer part of the system ! Summing until len(self.synapses) - 1 only
            for idx in range(conv_len):
                phi += torch.sum( self.pools[idx](self.synapses[idx](layers[idx])) * layers[idx+1], dim=(1,2,3)).squeeze()     
                #phi += torch.sum( self.conv_bias[idx] * layers[idx+1], dim=(1,2,3)).squeeze()
            for idx in range(conv_len, tot_len-1):
                phi += torch.sum( self.synapses[idx](layers[idx].view(mbs,-1)) * layers[idx+1], dim=1).squeeze()
             
            #WATCH OUT: the prediction is made with softmax[last weights[penultimate layer]]
            if beta!=0.0:
                L = criterion(self.synapses[-1](layers[-1].view(mbs,-1)).double(), y).squeeze()             
                phi -= beta*L            
        
        return phi
    

    def forward(self, x, y, neurons, T, beta=0.0, criterion=torch.nn.MSELoss(reduction='none'), check_thm=False):
 
        not_mse = (criterion.__class__.__name__.find('MSE')==-1)
        mbs = x.size(0)
        device = x.device     
        
        if check_thm:
            for t in range(T):
                phi = self.Phi(x, y, neurons, beta, criterion)
                init_grads = torch.tensor([1 for i in range(mbs)], dtype=torch.float, device=device, requires_grad=True)
                grads = torch.autograd.grad(phi, neurons, grad_outputs=init_grads, create_graph=True)

                for idx in range(len(neurons)-1):
                    neurons[idx] = self.activation( grads[idx] )
                    neurons[idx].retain_grad()
             
                if not_mse and not(self.softmax):
                    neurons[-1] = grads[-1]
                else:
                    neurons[-1] = self.activation( grads[-1] )

                neurons[-1].retain_grad()
        else:
             for t in range(T):
                phi = self.Phi(x, y, neurons, beta, criterion)
                init_grads = torch.tensor([1 for i in range(mbs)], dtype=torch.float, device=device, requires_grad=True)
                grads = torch.autograd.grad(phi, neurons, grad_outputs=init_grads, create_graph=False)

                for idx in range(len(neurons)-1):
                    neurons[idx] = self.activation( grads[idx] )
                    neurons[idx].requires_grad = True
             
                if not_mse and not(self.softmax):
                    neurons[-1] = grads[-1]
                else:
                    neurons[-1] = self.activation( grads[-1] )

                neurons[-1].requires_grad = True

        return neurons
       

    def init_neurons(self, mbs, device):
        
        neurons = []
        append = neurons.append
        size = self.in_size
        for idx in range(len(self.channels)-1): 
            size = int( (size - self.kernels[idx])/self.strides[idx] + 1 )                     # size after conv
            if self.pools[idx].__class__.__name__.find('Pool')!=-1:
                size = int( (size - self.pools[idx].kernel_size)/self.pools[idx].stride + 1 )  # size after Pool
            append(torch.zeros((mbs, self.channels[idx+1], size, size), requires_grad=True, device=device))

        size = size * size * self.channels[-1]
        
        if not self.softmax:
            for idx in range(len(self.fc)):
                append(torch.zeros((mbs, self.fc[idx]), requires_grad=True, device=device))
        else:
            #WATCH OUT: we *REMOVE* the output layer from the system
            for idx in range(len(self.fc) - 1):
                append(torch.zeros((mbs, self.fc[idx]), requires_grad=True, device=device))            
            
        return neurons

    def compute_syn_grads(self, x, y, neurons_1, neurons_2, betas, criterion, check_thm=False):
        
        beta_1, beta_2 = betas
        
        self.zero_grad()            # p.grad is zero
        if not(check_thm):
            phi_1 = self.Phi(x, y, neurons_1, beta_1, criterion)
        else:
            phi_1 = self.Phi(x, y, neurons_1, beta_2, criterion)
        phi_1 = phi_1.mean()
        
        phi_2 = self.Phi(x, y, neurons_2, beta_2, criterion)
        phi_2 = phi_2.mean()
        
        delta_phi = (phi_2 - phi_1)/(beta_1 - beta_2)        
        delta_phi.backward() # p.grad = -(d_Phi_2/dp - d_Phi_1/dp)/(beta_2 - beta_1) ----> dL/dp  by the theorem
 
           
   
 
# Vector Field Convolutional Neural Network

class VF_CNN(torch.nn.Module):
    def __init__(self, in_size, channels, kernels, strides, fc, pools, activation=hard_sigmoid, softmax = False):
        super(VF_CNN, self).__init__()

        # Dimensions used to initialize neurons
        self.in_size = in_size
        self.channels = channels
        self.kernels = kernels
        self.strides = strides
        self.fc = fc
        
        self.activation = activation
        self.pools = pools
        
        self.synapses = torch.nn.ModuleList()
        self.B_syn = torch.nn.ModuleList()

        self.softmax = softmax

        """
        softmax is a boolean function which tells whether we use the parametrized implementation 
        of the softmax prediction (section 2.9 of overleaf document). In this case, the last layer
        is *NO LONGER* part of the system: it is *ONLY* used for prediction and it does not interact
        with the rest of the system, except during nudging through beta*l. So the Phi function is
        computed until the *PENULTIMATE* layer. 

        This option affects:
        - Phi method
        - init_neurons method
        - train function
        """

        size = in_size

        for idx in range(len(channels)-1): 
            self.synapses.append(torch.nn.Conv2d(channels[idx], channels[idx+1], kernels[idx], 
                                                 stride=strides[idx], bias=True))
                
            if idx>0:  # backward synapses except for first layer
                self.B_syn.append(torch.nn.Conv2d(channels[idx], channels[idx+1], kernels[idx],
                                                      stride=strides[idx], bias=False))

            size = int( (size - kernels[idx])/strides[idx] + 1 )          # size after conv
            if self.pools[idx].__class__.__name__.find('Pool')!=-1:
                size = int( (size - pools[idx].kernel_size)/pools[idx].stride + 1 )   # size after Pool

        size = size * size * channels[-1]
        
        fc_layers = [size] + fc

        for idx in range(len(fc)):
            self.synapses.append(torch.nn.Linear(fc_layers[idx], fc_layers[idx+1], bias=True))
            if not(self.softmax and (idx==(len(fc)-1))):
                self.B_syn.append(torch.nn.Linear(fc_layers[idx], fc_layers[idx+1], bias=False))


    def Phi(self, x, y, neurons, beta, criterion):

        mbs = x.size(0)       
        conv_len = len(self.kernels)
        tot_len = len(self.synapses)
        bck_len = len(self.B_syn)

        layers = [x] + neurons        
        phis = []

        #Phi computation changes depending on softmax == True or not
        if not self.softmax:

            for idx in range(conv_len-1):    
                phi = torch.sum( self.pools[idx](self.synapses[idx](layers[idx])) * layers[idx+1], dim=(1,2,3)).squeeze()     
                phi += torch.sum( self.pools[idx+1](self.B_syn[idx](layers[idx+1])) * layers[idx+2], dim=(1,2,3)).squeeze()
                phis.append(phi)

            phi = torch.sum( self.pools[conv_len-1](self.synapses[conv_len-1](layers[conv_len-1])) * layers[conv_len], dim=(1,2,3)).squeeze()
            phi += torch.sum( self.B_syn[conv_len-1](layers[conv_len].view(mbs,-1)) * layers[conv_len+1], dim=1).squeeze()            
            phis.append(phi)            

            for idx in range(conv_len+1, tot_len-1):
                phi = torch.sum( self.synapses[idx](layers[idx].view(mbs,-1)) * layers[idx+1], dim=1).squeeze()
                phi += torch.sum( self.B_syn[idx](layers[idx+1].view(mbs,-1)) * layers[idx+2], dim=1).squeeze()             
                phis.append(phi)

            phi = torch.sum( self.synapses[-1](layers[-2].view(mbs,-1)) * layers[-1], dim=1).squeeze()
            if beta!=0.0:
                if criterion.__class__.__name__.find('MSE')!=-1:
                    y = F.one_hot(y, num_classes=10).double()
                    L = 0.5*criterion(layers[-1].double(), y).sum(dim=1).squeeze()   
                else:
                    L = criterion(layers[-1].double(), y).squeeze()             
                phi -= beta*L
            phis.append(phi)

        else:
            #WATCH OUT: the output layer used for the prediction is no longer part of the system ! Summing until len(self.synapses) - 1 only
            for idx in range(conv_len-1):    
                phi = torch.sum( self.pools[idx](self.synapses[idx](layers[idx])) * layers[idx+1], dim=(1,2,3)).squeeze()     
                phi += torch.sum( self.pools[idx+1](self.B_syn[idx](layers[idx+1])) * layers[idx+2], dim=(1,2,3)).squeeze()
                phis.append(phi)
            
            phi = torch.sum( self.pools[conv_len-1](self.synapses[conv_len-1](layers[conv_len-1])) * layers[conv_len], dim=(1,2,3)).squeeze()
            if bck_len>=conv_len:
                phi += torch.sum( self.B_syn[conv_len-1](layers[conv_len].view(mbs,-1)) * layers[conv_len+1], dim=1).squeeze()            
                phis.append(phi)            

            for idx in range(conv_len+1, tot_len-2):
                phi = torch.sum( self.synapses[idx](layers[idx].view(mbs,-1)) * layers[idx+1], dim=1).squeeze()
                phi += torch.sum( self.B_syn[idx](layers[idx+1].view(mbs,-1)) * layers[idx+2], dim=1).squeeze()             
                phis.append(phi)

            #WATCH OUT: the prediction is made with softmax[last weights[penultimate layer]]
            if beta!=0.0:
                L = criterion(self.synapses[-1](layers[-1].view(mbs,-1)).double(), y).squeeze()             
                phi -= beta*L            
            phis.append(phi)           
        
        return phis
    

    def forward(self, x, y, neurons, T, beta=0.0, criterion=torch.nn.MSELoss(reduction='none'), check_thm=False):
 
        not_mse = (criterion.__class__.__name__.find('MSE')==-1)
        mbs = x.size(0)
        device = x.device     
        
        if check_thm:
            for t in range(T):
                phis = self.Phi(x, y, neurons, beta, criterion)
                for idx in range(len(neurons)-1):
                    init_grad = torch.tensor([1 for i in range(mbs)], dtype=torch.float, device=device, requires_grad=True)
                    grad = torch.autograd.grad(phis[idx], neurons[idx], grad_outputs=init_grad, create_graph=True)

                    neurons[idx] = self.activation( grad[0] )
                    neurons[idx].retain_grad()
             
                init_grad = torch.tensor([1 for i in range(mbs)], dtype=torch.float, device=device, requires_grad=True)
                grad = torch.autograd.grad(phis[-1], neurons[-1], grad_outputs=init_grad, create_graph=True)
                if not_mse and not(self.softmax):
                    neurons[-1] = grad[0]
                else:
                    neurons[-1] = self.activation( grad[0] )

                neurons[-1].retain_grad()
        else:
             for t in range(T):
                phis = self.Phi(x, y, neurons, beta, criterion)
                for idx in range(len(neurons)-1):
                    init_grad = torch.tensor([1 for i in range(mbs)], dtype=torch.float, device=device, requires_grad=True)
                    grad = torch.autograd.grad(phis[idx], neurons[idx], grad_outputs=init_grad, create_graph=False)

                    neurons[idx] = self.activation( grad[0] )
                    neurons[idx].requires_grad = True
             
                init_grad = torch.tensor([1 for i in range(mbs)], dtype=torch.float, device=device, requires_grad=False)
                grad = torch.autograd.grad(phis[-1], neurons[-1], grad_outputs=init_grad, create_graph=False)
                if not_mse and not(self.softmax):
                    neurons[-1] = grad[0]
                else:
                    neurons[-1] = self.activation( grad[0] )

                neurons[-1].requires_grad = True

        return neurons
       

    def init_neurons(self, mbs, device):
        
        neurons = []
        append = neurons.append
        size = self.in_size
        for idx in range(len(self.channels)-1): 
            size = int( (size - self.kernels[idx])/self.strides[idx] + 1 )                     # size after conv
            if self.pools[idx].__class__.__name__.find('Pool')!=-1:
                size = int( (size - self.pools[idx].kernel_size)/self.pools[idx].stride + 1 )  # size after Pool
            append(torch.zeros((mbs, self.channels[idx+1], size, size), requires_grad=True, device=device))

        size = size * size * self.channels[-1]
        
        if not self.softmax:
            for idx in range(len(self.fc)):
                append(torch.zeros((mbs, self.fc[idx]), requires_grad=True, device=device))
        else:
            #WATCH OUT: we *REMOVE* the output layer from the system
            for idx in range(len(self.fc) - 1):
                append(torch.zeros((mbs, self.fc[idx]), requires_grad=True, device=device))            
            
        return neurons

    def compute_syn_grads(self, x, y, neurons_1, neurons_2, betas, criterion, check_thm=False):
        
        beta_1, beta_2 = betas
        
        self.zero_grad()            # p.grad is zero
        if not(check_thm):
            phis_1 = self.Phi(x, y, neurons_1, beta_1, criterion)
        else:
            phis_1 = self.Phi(x, y, neurons_1, beta_2, criterion)
        
        phis_2 = self.Phi(x, y, neurons_2, beta_2, criterion)
     
        for idx in range(len(neurons_1)):
            phi_1 = phis_1[idx].mean()
            phi_2 = phis_2[idx].mean()
            delta_phi = (phi_2 - phi_1)/(beta_1 - beta_2)        
            delta_phi.backward() # p.grad = -(d_Phi_2/dp - d_Phi_1/dp)/(beta_2 - beta_1)
 
       

         

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
        leaf_neurons = []
        for idx in range(len(neurons)):
            neurons[idx] = neurons[idx].detach()
            neurons[idx].requires_grad = True
            leaf_neurons.append(neurons[idx])

        neurons = model(x, y, neurons, T2-K, beta=beta_1, criterion=criterion, check_thm=True) # T2-K time step
        
        # final loss
        if criterion.__class__.__name__.find('MSE')!=-1:
            loss = (1/(2.0*x.size(0)))*criterion(neurons[-1].double(), F.one_hot(y, num_classes=10).double()).sum(dim=1).squeeze()
        else:
            if not model.softmax:
                loss = (1/(x.size(0)))*criterion(neurons[-1].double(), y).squeeze()
            else:
                loss = (1/(x.size(0)))*criterion(model.synapses[-1](neurons[-1].view(x.size(0),-1)).double(), y).squeeze()

        # setting gradients field to zero before backward
        neurons_zero_grad(leaf_neurons)
        model.zero_grad()

        # Backpropagation through time
        loss.backward(torch.tensor([1 for i in range(x.size(0))], dtype=torch.float, device=x.device, requires_grad=True))

        # Collecting BPTT gradients : for parameters they are partial sums over T2-K time steps
        if K!=T2:
            for name, p in model.named_parameters():
                update = torch.empty_like(p).copy_(grad_or_zero(p))
                BPTT[name].append( update.unsqueeze(0) )  # unsqueeze for time dimension
                neurons = copy(ref_neurons) # Resetting the neurons to T1-T2 step
        if K!=0:
            for idx in range(len(leaf_neurons)):
                update = torch.empty_like(leaf_neurons[idx]).copy_(grad_or_zero(leaf_neurons[idx]))
                BPTT['neurons_'+str(idx)].append( update.mul(-x.size(0)).unsqueeze(0) )  # unsqueeze for time dimension

                                
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
        sign = torch.where(EP[key]*BPTT[key] < 0, torch.ones_like(EP[key]), torch.zeros_like(EP[key]))
        print(key.replace('.','_'), '\t RMSE =', round(comp.mean().item(), 4), '\t SIGN err =', round(sign.mean().item(), 4))
    print('\n')

    

        
def train(model, optimizer, train_loader, test_loader, T1, T2, betas, device, epochs, criterion, alg='EP', 
                          random_sign=False, save=False, check_thm=False, path='', checkpoint=None, thirdphase = False, save_nrn=False):
    
    model.train()
    mbs = train_loader.batch_size
    start = time.time()
    iter_per_epochs = math.ceil(len(train_loader.dataset)/mbs)
    beta_1, beta_2 = betas
    
    if checkpoint is None:
        train_acc = [10.0]
        test_acc = [10.0]
        best = 0.0
        epoch_sofar = 0
    else:
        train_acc = checkpoint['train_acc']
        test_acc = checkpoint['test_acc']    
        best = checkpoint['best']
        epoch_sofar = checkpoint['epoch']

    for epoch in range(epochs):
        run_correct = 0
        run_total = 0
        
        for idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            
            neurons = model.init_neurons(x.size(0), device)
            if alg=='EP':
                # First phase
                neurons = model(x, y, neurons, T1, beta=beta_1, criterion=criterion)
                neurons_1 = copy(neurons)
            elif alg=='BPTT':
                neurons = model(x, y, neurons, T1-T2, beta=0.0, criterion=criterion)           
                # detach data and neurons from the graph
                x = x.detach()
                x.requires_grad = True
                for k in range(len(neurons)):
                    neurons[k] = neurons[k].detach()
                    neurons[k].requires_grad = True

                neurons = model(x, y, neurons, T2, beta=0.0, criterion=criterion, check_thm=True) # T2 time step

            # Predictions for running accuracy
            with torch.no_grad():
                if not model.softmax:
                    pred = torch.argmax(neurons[-1], dim=1).squeeze()
                else:
                    #WATCH OUT: prediction is different when softmax == True
                    pred = torch.argmax(F.softmax(model.synapses[-1](neurons[-1].view(x.size(0),-1)), dim = 1), dim = 1).squeeze()

                run_correct += (y == pred).sum().item()
                run_total += x.size(0)
                if ((idx%(iter_per_epochs//10)==0) or (idx==iter_per_epochs-1)) and save_nrn:
                    plot_neural_activity(neurons, path + '/ep-'+str(epoch_sofar+epoch+1)+'_iter-'+str(idx+1)+'_neural_activity.png')
            
            if alg=='EP':
                # Second phase
                if random_sign and (beta_1==0.0):
                    rnd_sgn = 2*np.random.randint(2) - 1
                    betas = beta_1, rnd_sgn*beta_2
                    beta_1, beta_2 = betas
            
                neurons = model(x, y, neurons, T2, beta = beta_2, criterion=criterion)
                neurons_2 = copy(neurons)

                # Third phase (if we approximate f' as f'(x) = (f(x+h) - f(x-h))/2h)
                if thirdphase:
                    #come back to the first equilibrium
                    neurons = copy(neurons_1)
                    neurons = model(x, y, neurons, T2, beta = - beta_2, criterion=criterion)
                    neurons_3 = copy(neurons)
                    model.compute_syn_grads(x, y, neurons_2, neurons_3, (beta_2, - beta_2), criterion)
                else:
                    model.compute_syn_grads(x, y, neurons_1, neurons_2, betas, criterion)

                optimizer.step()            

            elif alg=='BPTT':
         
                # final loss
                if criterion.__class__.__name__.find('MSE')!=-1:
                    loss = 0.5*criterion(neurons[-1].double(), F.one_hot(y, num_classes=10).double()).sum(dim=1).mean().squeeze()
                else:
                    if not model.softmax:
                        loss = criterion(neurons[-1].double(), y).mean().squeeze()
                    else:
                        loss = criterion(model.synapses[-1](neurons[-1].view(x.size(0),-1)).double(), y).mean().squeeze()
                # setting gradients field to zero before backward
                model.zero_grad()

                # Backpropagation through time
                loss.backward()
                optimizer.step()
                        
            if ((idx%(iter_per_epochs//10)==0) or (idx==iter_per_epochs-1)):
                run_acc = run_correct/run_total
                print('Epoch :', round(epoch_sofar+epoch+(idx/iter_per_epochs), 2),
                      '\tRun train acc :', round(run_acc,3),'\t('+str(run_correct)+'/'+str(run_total)+')\t',
                      timeSince(start, ((idx+1)+epoch*iter_per_epochs)/(epochs*iter_per_epochs)))
                
                if check_thm and alg=='EP':
                    BPTT, EP = check_gdu(model, x[0:5,:], y[0:5], T1, T2, betas, criterion)
                    RMSE(BPTT, EP)
    
        
        test_correct = evaluate(model, test_loader, T1, device)
        test_acc_t = test_correct/(len(test_loader.dataset))
        if save:
            test_acc.append(100*test_acc_t)
            train_acc.append(100*run_acc)
            if test_correct > best:
                best = test_correct
                torch.save({'model_state_dict': model.state_dict(), 'opt': optimizer.state_dict(),
                            'train_acc': train_acc, 'test_acc': test_acc, 
                            'best': best, 'epoch': epoch_sofar+epoch+1},  path + '/checkpoint.tar')
                torch.save(model, path + '/model.pt')
            plot_acc(train_acc, test_acc, path)        


            
def evaluate(model, loader, T, device):
    
    model.eval()
    correct=0
    phase = 'Train' if loader.dataset.train else 'Test'
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        neurons = model.init_neurons(x.size(0), device)
        neurons = model(x, y, neurons, T)

        if not model.softmax:
            pred = torch.argmax(neurons[-1], dim=1).squeeze()
        else:
            #WATCH OUT: prediction is different when softmax == True
            pred = torch.argmax(F.softmax(model.synapses[-1](neurons[-1].view(x.size(0),-1)), dim = 1), dim = 1).squeeze()

        correct += (y == pred).sum().item()

    acc = correct/len(loader.dataset) 
    print(phase+' accuracy :\t', acc)   
    return correct


            










