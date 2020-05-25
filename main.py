import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torch.nn.functional as F
import argparse
import matplotlib
matplotlib.use('Agg')

import glob
from PIL import Image
import os
from datetime import datetime
import time
import math
import sys
from model_utils import *
from data_utils import *

parser = argparse.ArgumentParser(description='Eqprop')
parser.add_argument('--model',type = str, default = 'MLP', metavar = 'm', help='model')
parser.add_argument('--task',type = str, default = 'MNIST', metavar = 't', help='task')

parser.add_argument('--pools', type = str, default = 'mm', metavar = 'p', help='pooling')
parser.add_argument('--archi', nargs='+', type = int, default = [784, 512, 10], metavar = 'A', help='architecture of the network')
parser.add_argument('--channels', nargs='+', type = int, default = [32, 64], metavar = 'C', help='channels of the convnet')
parser.add_argument('--kernels', nargs='+', type = int, default = [5, 5], metavar = 'K', help='kernels sizes of the convnet')
parser.add_argument('--strides', nargs='+', type = int, default = [1, 1], metavar = 'S', help='strides of the convnet')
parser.add_argument('--paddings', nargs='+', type = int, default = [0, 0], metavar = 'P', help='paddings of the conv layers')
parser.add_argument('--fc', nargs='+', type = int, default = [10], metavar = 'S', help='linear classifier of the convnet')

parser.add_argument('--act',type = str, default = 'mysig', metavar = 'a', help='activation function')
parser.add_argument('--optim', type = str, default = 'sgd', metavar = 'opt', help='optimizer for training')
parser.add_argument('--lrs', nargs='+', type = float, default = [], metavar = 'l', help='layer wise lr')
parser.add_argument('--wds', nargs='+', type = float, default = None, metavar = 'l', help='layer weight decays')
parser.add_argument('--mmt',type = float, default = 0.0, metavar = 'mmt', help='Momentum for sgd')
parser.add_argument('--loss', type = str, default = 'mse', metavar = 'lss', help='loss for training')
parser.add_argument('--alg', type = str, default = 'EP', metavar = 'al', help='EP or BPTT')
parser.add_argument('--mbs',type = int, default = 20, metavar = 'M', help='minibatch size')
parser.add_argument('--T1',type = int, default = 20, metavar = 'T1', help='Time of first phase')
parser.add_argument('--T2',type = int, default = 4, metavar = 'T2', help='Time of second phase')
parser.add_argument('--betas', nargs='+', type = float, default = [0.0, 0.01], metavar = 'Bs', help='Betas')
parser.add_argument('--epochs',type = int, default = 1,metavar = 'EPT',help='Number of epochs per tasks')
parser.add_argument('--check-thm', default = False, action = 'store_true', help='checking the gdu while training')
parser.add_argument('--random-sign', default = False, action = 'store_true', help='randomly switch beta_2 sign')
parser.add_argument('--data-aug', default = False, action = 'store_true', help='enabling data augmentation for cifar10')
parser.add_argument('--lr-decay', default = False, action = 'store_true', help='enabling learning rate decay')
parser.add_argument('--scale',type = float, default = None, metavar = 'g', help='scal factor for weight init')
parser.add_argument('--save', default = False, action = 'store_true', help='saving results')
parser.add_argument('--todo', type = str, default = 'train', metavar = 'tr', help='training or plot gdu curves')
parser.add_argument('--load-path', type = str, default = '', metavar = 'l', help='load a model')
parser.add_argument('--seed',type = int, default = None, metavar = 's', help='random seed')
parser.add_argument('--device',type = int, default = 0, metavar = 'd', help='device')
parser.add_argument('--thirdphase', default = False, action = 'store_true', help='add third phase for higher order evaluation of the gradient (default: False)')
parser.add_argument('--softmax', default = False, action = 'store_true', help='softmax loss with parameters (default: False)')
parser.add_argument('--same-update', default = False, action = 'store_true', help='same update is applied for VFCNN back and forward')

args = parser.parse_args()
command_line = ' '.join(sys.argv)

print('\n')
print(command_line)
print('\n')
print('##################################################################')
print('\nargs\tmbs\tT1\tT2\tepochs\tactivation\tbetas')
print('\t',args.mbs,'\t',args.T1,'\t',args.T2,'\t',args.epochs,'\t',args.act, '\t', args.betas)
print('\n')

device = torch.device('cuda:'+str(args.device) if torch.cuda.is_available() else 'cpu')


if args.save:
    date = datetime.now().strftime('%Y-%m-%d')
    time = datetime.now().strftime('%H-%M-%S')
    if args.load_path=='':
        path = 'results/'+args.alg+'/'+args.loss+'/'+date+'/'+time+'_gpu'+str(args.device)
    else:
        path = args.load_path
    if not(os.path.exists(path)):
        os.makedirs(path)
else:
    path = ''



mbs=args.mbs
if args.seed is not None:
    torch.manual_seed(args.seed)




if args.task=='MNIST':
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), 
                                                torchvision.transforms.Normalize(mean=(0.0,), std=(1.0,))])

    mnist_dset_train = torchvision.datasets.MNIST('./mnist_pytorch', train=True, transform=transform, target_transform=None, download=True)
    train_loader = torch.utils.data.DataLoader(mnist_dset_train, batch_size=mbs, shuffle=True, num_workers=0)

    mnist_dset_test = torchvision.datasets.MNIST('./mnist_pytorch', train=False, transform=transform, target_transform=None, download=True)
    test_loader = torch.utils.data.DataLoader(mnist_dset_test, batch_size=200, shuffle=False, num_workers=0)

elif args.task=='CIFAR10':
    if args.data_aug:
        transform_train = torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(0.5),
                                                          torchvision.transforms.RandomCrop(size=[32,32], padding=4, padding_mode='edge'),
                                                          torchvision.transforms.ToTensor(), 
                                                          torchvision.transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), 
                                                                                           std=(3*0.2023, 3*0.1994, 3*0.2010)) ])   
    else:
         transform_train = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), 
                                                          torchvision.transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), 
                                                                                           std=(3*0.2023, 3*0.1994, 3*0.2010)) ])   

    transform_test = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), 
                                                     torchvision.transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), 
                                                                                      std=(3*0.2023, 3*0.1994, 3*0.2010)) ]) 

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




if args.loss=='mse':
    criterion = torch.nn.MSELoss(reduction='none').to(device)
elif args.loss=='cel':
    criterion = torch.nn.CrossEntropyLoss(reduction='none').to(device)
print('loss =', criterion, '\n')



if args.load_path=='':

    if args.model=='MLP':
        model = P_MLP(args.archi, activation=activation)

    elif args.model=='VFMLP':
        model = VF_MLP(args.archi, activation=activation)

    elif args.model.find('CNN')!=-1:

        if args.task=='MNIST':
            pools = make_pools(args.pools)
            channels = [1]+args.channels 
            if args.model=='CNN':
                model = P_CNN(28, channels, args.kernels, args.strides, args.fc, pools, args.paddings, 
                                  activation=activation, softmax=args.softmax)
            elif args.model=='VFCNN':
                model = VF_CNN(28, channels, args.kernels, args.strides, args.fc, pools, args.paddings,
                                   activation=activation, softmax=args.softmax, same_update=args.same_update)

        elif args.task=='CIFAR10':    
           pools = make_pools(args.pools)
           channels = [3]+args.channels
           if args.model=='CNN':
                model = P_CNN(32, channels, args.kernels, args.strides, args.fc, pools, args.paddings,
                              activation=activation, softmax=args.softmax)
           elif args.model=='VFCNN':
                model = VF_CNN(32, channels, args.kernels, args.strides, args.fc, pools, args.paddings,
                              activation=activation, softmax = args.softmax, same_update=args.same_update)
        
        elif args.task=='imagenet':   #only for gducheck
            pools = make_pools(args.pools)
            channels = [3]+args.channels 
            model = P_CNN(224, channels, args.kernels, args.strides, args.fc, pools, args.paddings, 
                            activation=activation, softmax=args.softmax)
                       

        print('\n')
        print('Poolings =', model.pools)
    if args.scale is not None:
        model.apply(my_init(args.scale))
else:
    model = torch.load(args.load_path + '/model.pt')

model.to(device)
print(model)

betas = args.betas[0], args.betas[1]


if args.todo=='train':
    assert(len(args.lrs)==len(model.synapses))

    # Constructing the optimizer
    optim_params = []
    for idx in range(len(model.synapses)):
        if args.wds is None:
            optim_params.append(  {'params': model.synapses[idx].parameters(), 'lr': args.lrs[idx]}  )
        else:
            optim_params.append(  {'params': model.synapses[idx].parameters(), 'lr': args.lrs[idx], 'weight_decay': args.wds[idx]}  )
    if hasattr(model, 'B_syn'):
        for idx in range(len(model.B_syn)):
            if args.wds is None:
                optim_params.append( {'params': model.B_syn[idx].parameters(), 'lr': args.lrs[idx+1]} )
            else:
                optim_params.append( {'params': model.B_syn[idx].parameters(), 'lr': args.lrs[idx+1], 'weight_decay': args.wds[idx+1]} )


    if args.optim=='sgd':
        optimizer = torch.optim.SGD( optim_params, momentum=args.mmt )
    elif args.optim=='adam':
        optimizer = torch.optim.Adam( optim_params )

    # Constructing the scheduler
    if args.lr_decay:
        #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40,80,120], gamma=0.1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 100, eta_min=1e-5)
    else:
        scheduler = None

    # Loading the state when resuming a run
    if args.load_path!='':
        checkpoint = torch.load(args.load_path + '/checkpoint.tar')
        optimizer.load_state_dict(checkpoint['opt'])
        if checkpoint['scheduler'] is not None and args.lr_decay:
            scheduler.load_state_dict(checkpoint['scheduler'])
    else: 
        checkpoint = None
    
    print(optimizer)
    print('\ntraining algorithm : ',args.alg, '\n')
    if args.save and args.load_path=='':
        createHyperparametersFile(path, args, model, command_line)
        

    train(model, optimizer, train_loader, test_loader, args.T1, args.T2, betas, device, args.epochs, criterion, alg=args.alg, 
                 random_sign=args.random_sign, check_thm=args.check_thm, save=args.save, path=path, checkpoint=checkpoint, 
                 thirdphase=args.thirdphase, scheduler=scheduler)


elif args.todo=='gducheck':

    if args.task != 'imagenet':
        dataiter = iter(train_loader)
        images, labels = dataiter.next()
        images, labels = images[0:10,:], labels[0:10]
        images, labels = images.to(device), labels.to(device)
    else:
        images = []
        all_files = glob.glob('imagenet_samples/*.JPEG')
        for filename in all_files:
            image = Image.open(filename)
            image = torchvision.transforms.functional.center_crop(image, 224)
            image = torchvision.transforms.functional.to_tensor(image)
            image.unsqueeze_(0)
            image = image.add_(-image.mean()).div_(image.std())
            images.append(image)
        labels = torch.randint(1000, (len(images),))
        images = torch.cat(images, dim=0)
        images, labels = images.to(device), labels.to(device)
        print(images.shape)

    BPTT, EP = check_gdu(model, images, labels, args.T1, args.T2, betas, criterion)
    if args.thirdphase:
        beta_1, beta_2 = args.betas
        _, EP_2 = check_gdu(model, images, labels, args.T1, args.T2, (beta_1, -beta_2), criterion)

    RMSE(BPTT, EP)
    if args.save:
        torch.save(BPTT, path+'/BPTT.pt')
        torch.save(EP, path+'/EP.pt')
        if args.thirdphase:
            torch.save(EP_2, path+'/EP_2.pt')
            plot_gdu(BPTT, EP, path, EP_2=EP_2)
        else:
            plot_gdu(BPTT, EP, path)


elif args.todo=='evaluate':

    training_acc = evaluate(model, train_loader, args.T1, device)
    training_acc /= len(train_loader.dataset)
    print('\nTrain accuracy :', round(training_acc,2), file=open(path+'/hyperparameters.txt', 'a'))
    test_acc = evaluate(model, test_loader, args.T1, device)
    test_acc /= len(test_loader.dataset)
    print('\nTest accuracy :', round(test_acc, 2), file=open(path+'/hyperparameters.txt', 'a'))








