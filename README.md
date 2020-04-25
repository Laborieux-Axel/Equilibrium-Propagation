# Equilibrium Propagation (EP)  

Reproducing some results of this __[paper](https://arxiv.org/pdf/1905.13633.pdf)__ and scaling to CIFAR10.  
See the bottom of the page for a summary of all the arguments in the command lines.

## Examples of command lines  
### Multi Layer Perceptron on MNIST  

Checking gradient descent updates (GDU) property by setting `--todo 'gducheck'`:  
```
python main.py --model 'MLP' --task 'MNIST' --archi 784 512 10 --act 'tanh' --todo 'gducheck' --betas 0.0 0.01 --T1 30 --T2 10 --mbs 50 --device 0  
```

Training by setting `--todo 'train'`:  
```
python main.py --model 'MLP' --task 'MNIST' --archi 784 512 10 --lrs 0.08 0.04 --epochs 3 --act 'mysig' --todo 'train' --betas 0.0 0.1 --T1 30 --T2 10 --mbs 50 --device 0  
```
Note that during training, the GDU theorem quantifying the similarity between BPTT and EP updates can be checked and printed in the console by setting `--check-thm`.  

### Convolutional Neural Network on MNIST or CIFAR10    

The training can be done with **Equilibrium Propagation**  by setting `--alg 'EP'` and three different loss functions:

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


## Summary table of the command lines arguments  

|Arguments|Description|Examples|
|-------|---------|------|
|`model`|Choose MLP or CNN and Vector field.|`--model 'MLP'`, `--model 'VFMLP'`,`--model 'CNN'`,`--model 'VFCNN'`|
|`task`|Choose the task.|`--task 'MNIST'`, `--task 'CIFAR10'`|
|`data-aug`|Enable data augmentation for CIFAR10.|`--data-aug`|
|`archi`|Layers dimension for MLP.|`--archi 784 512 10`|
|`channels`|Feature maps for CNN.|`--channels 128 256 512`|
|`pools`|Layers wise poolings. `m` is maxpool, `a` is avgpool and `i` is no pooling.|`--pools 'mmm'`|
|`kernels`|Kernel sizes for CNN.|`--kernels 3 3 3`|
|`strides`|Strides for CNN.|`--strides 1 1 1`|
|`fc`|Linear classifier|`--fc 10` for one fc layer, `--fc 512 10`|
|`act`|Activation function for neurons|`--act 'tanh'`,`'mysig'`,`'hard_sigmoid'`|
|`todo`|Train or check the theorem|`--todo 'train'`,`--todo 'gducheck'`|
|`alg`|EqProp or BackProp Through Time.|`--alg 'EP'`, `--alg 'BPTT'`|
|`check-thm`|Check the theorem while training. (only if EP)|`--check-thm`|
|`T1`,`T2`|Number of time steps for phase 1 and 2.|`--T1 30 --T2 10`|
|`betas`|Beta values for EP.|`--betas 0.0 0.1`|
|`random-sign`|Choose a random sign for beta_2.|`--random-sign`|
|`thirdphase`|Two phases 2 are done with beta_2 and -beta_2.|`--thirdphase`|
|`loss`|Loss functions.|`--loss 'mse'`,`--loss 'cel'`, `--loss 'cel' --softmax`|
|`optim`|Optimizer for training.|`--optim 'sgd'`, `--optim 'adam'`|
|`lrs`|Layer wise learning rates.|`--lrs 0.01 0.005`|
|`epochs`|Number of epochs.|`--epochs 200`|
|`mbs`|Minibatch size|`--mbs 128`|
|`device`|Index of the gpu.|`--device 0`|
|`save`|Create a folder with results.|`--save`|
|`save-nrn`|Plots neurons activation in results folder.|`--save-nrn`|
|`load-path`|Resume the training of a saved simulations.|`--load-path 'results/2020-04-25/10-11-12'`|
|`seed`|Choose the seed.|`--seed 0`|
