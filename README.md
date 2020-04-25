# Equilibrium Propagation (EP)  

Reproducing some results of this __[paper](https://arxiv.org/pdf/1905.13633.pdf)__.  

## Summary table of the command lines arguments  

|Flag|Description|Example|
|-------|-----------|-------|
|`--model`|choose MLP or CNN and Vector field|`'MLP'`, `'VFMLP'`,`'CNN'`,`'VFCNN'`|
|`--task`|choose the task|`'MNIST'`, `'CIFAR10'`|
|`--data-aug`|enable data augmentation for CIFAR10||
|`--archi`|layers dimension for MLP|`784 512 10`|
|`--channels`|feature maps for CNN|`128 256 512`|
|`--pools`|layers wise poolings. `m` is maxpool, `a` is avgpool and `i` is no pooling|`'mmm'`|
|`--kernels`|kernel sizes for CNN|`3 3 3`|
|`--strides`|strides for CNN|`1 1 1`|
|`--fc`|linear classifier|`--fc 10` for one fc layer, `--fc 512 10`|
|`--act`|activation function for neurons|`'tanh'`,`'mysig'`,`'hard_sigmoid'`|
|`--todo`|train or check the theorem|`'train'`,`'gducheck'`|
|`--check-thm`|check the theorem while training||
|`--alg`|EP or BPTT|`'EP'`, `'BPTT'`|
|`--T1`,`--T2`|number of time steps for phase 1 and 2|`--T1 30 --T2 10`|
|`--betas`|beta values for EP|`--betas 0.0 0.1`|
|`--random-sign`|choose a random sign for beta_2||
|`--thirdphase`|two phases 2 are done with beta_2 and -beta_2||
|`--loss`|loss functions|`--loss 'mse'`,`--loss 'cel'`, `--loss 'cel' --softmax`|
|`--optim`|optimizer for training|`'sgd'`, `'adam'`|
|`--lrs`|layer wise learning rates|`--lrs 0.01 0.005`|
|`--epochs`|number of epochs|`200`|
|`--mbs`|minibatch size|`128`|
|`--device`|index of the gpu|`0`|
|`--save`|create a folder with results||
|`--save-nrn`|plots neurons activation in results folder||
|`load-path`|resume learning|`'results/2020-04-25/10-11-12'`|
|`--seed`|choose the seed|`0`|


## MLP on MNIST:

Checking gradient descent updates (GDU) property by setting `--todo 'gducheck'`:  
```
python main.py --model 'MLP' --task 'MNIST' --archi 784 512 10 --act 'tanh' --todo 'gducheck' --betas 0.0 0.01 --T1 30 --T2 10 --mbs 50 --device 0  
```

Training by setting `--todo 'train'`:  
```
python main.py --model 'MLP' --task 'MNIST' --archi 784 512 10 --lrs 0.08 0.04 --epochs 3 --act 'mysig' --todo 'train' --betas 0.0 0.1 --T1 30 --T2 10 --mbs 50 --device 0  
```
Note that even when training, the GDU theorem can be checked on the fly by setting `--check-thm` regardless of the architecture.  

## CNN on MNIST or CIFAR10:  

The training can be done with **Equilibrium Propagation (EP)**  by setting `--alg 'EP'` and three different loss functions:

+ Mean Square Error by setting `--loss 'mse'`:

```
python main.py --model 'CNN' --task 'CIFAR10' --channels 128 256 512 --kernels 3 3 3 --pools 'mmm' --strides 1 1 1 --fc 10 --optim 'adam' --lrs 5e-5 5e-5 1e-5 7e-6 --epochs 400 --act 'hard_sigmoid' --todo 'train' --betas 0.0 0.5 --T1 250 --T2 30 --mbs 128 --alg 'EP' --random-sign --loss 'mse' --save --device 0
```

+ Crossentropy Loss on output neurons by setting `--loss 'cel'`:

```
python main.py --model 'CNN' --task 'CIFAR10' --channels 128 256 512 --kernels 3 3 3 --pools 'mmm' --strides 1 1 1 --fc 10 --optim 'adam' --lrs 5e-5 5e-5 1e-5 7e-6 --epochs 400 --act 'hard_sigmoid' --todo 'train' --betas 0.0 0.5 --T1 250 --T2 30 --mbs 128 --alg 'EP' --random-sign --loss 'cel' --save --device 0
```

+ Crossentropy Loss on FC(penultimate) by setting `--loss 'cel' --softmax`:

```
python main.py --model 'CNN' --task 'CIFAR10' --channels 128 256 512 --kernels 3 3 3 --pools 'mmm' --strides 1 1 1 --fc 10 --optim 'adam' --lrs 5e-5 5e-5 1e-5 7e-6 --epochs 400 --act 'hard_sigmoid' --todo 'train' --betas 0.0 0.5 --T1 250 --T2 30 --mbs 128 --alg 'EP' --random-sign --loss 'cel' --softmax --save --device 0
```

The flag `--random-sign` provides a random sign for beta in the second phase, it removes a bias in the estimation of the derivative with respect to beta. It can be replaced by the flag `--thirdphase` where the estimation of the derivative of dPhi is done between -beta and +beta. For the CIFAR10 simulation to be stable it is **mandatory** to use either of the flags.

The training can also be done by **Back Propagation Through Time (BPTT)** via `--alg 'BPTT'` and the three same loss functions as above: 

```
python main.py --model 'CNN' --task 'CIFAR10' --channels 128 256 512 --kernels 3 3 3 --pools 'mmm' --strides 1 1 1 --fc 10 --optim 'adam' --lrs 5e-5 5e-5 1e-5 7e-6 --epochs 400 --act 'hard_sigmoid' --todo 'train' --T1 250 --T2 30 --mbs 128 --alg 'BPTT' --loss 'mse' --save --device 0
```

For Vector Field models use `--model 'VFMLP'` instead of `--model 'MLP'` and `--model 'VFCNN'` instead of `--model 'CNN'`.
