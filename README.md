# Scaling Equilibrium Prop to Deep ConvNets

This repository contains the code producing the results of the paper "Scaling Equilibrium Prop to Deep ConvNets by Drastically Reducing its Gradient Estimator Bias".  
This implementation of EP makes extensive use of PyTorch automatic differentiation capability. We recommand the reader to look first at the class MLP of model_utils.py to understand how this implementation works, the convolutional architecture has the same structure but is naturally wordier. The dynamics of neurons as well as the EP update rule is obtained by automatic differentiation of the primitive function Phi.

## Setting up the environment

Run the following command lines to set the environment using conda:
```
conda create --name EP python=3.6
conda activate EP
conda install -c conda-forge matplotlib
conda install pytorch torchvision -c pytorch
```
## Training

When setting the flags `--todo 'train' --save`, a results folder will be created at results/(EP or BPTT)/loss/yyyy-mm-dd/hh-mm-ss with a plot of the train and test accuracy updated at each epoch, and an histogram of neural activations. The best performing model is saved at model.pt and the checkpoint for resuming training at checkpoint.tar. To resume training, simply rerun the same command line with the flag `--load-path 'results/.../hh-mm-ss'` and set the epoch argument to the remaining number of epochs. When the training is over, the final model and checkpoint are saved at final_model.pt and final_checkpoint.tar (they usually differ from the best model).

### Training a recurrent CNN on CIFAR-10 with symmetric connections

+ For the results on the MSE Loss function (relevant arguments `--loss 'mse'`):
```
# EP with one-sided gradient estimate
python main.py --model 'CNN' --task 'CIFAR10' --data-aug --channels 128 256 512 512 --kernels 3 3 3 3 --pools 'mmmm' --strides 1 1 1 1 --paddings 1 1 1 0 --fc 10 --optim 'sgd' --lrs 0.25 0.15 0.1 0.08 0.05 --wds 3e-4 3e-4 3e-4 3e-4 3e-4 --mmt 0.9 --lr-decay --epochs 120 --act 'my_hard_sig' --todo 'train' --T1 250 --T2 30 --mbs 128 --alg 'EP' --betas 0.0 0.5 --loss 'mse' --save --device 0 
```

```
# EP with random sign gradient estimate
python main.py --model 'CNN' --task 'CIFAR10' --data-aug --channels 128 256 512 512 --kernels 3 3 3 3 --pools 'mmmm' --strides 1 1 1 1 --paddings 1 1 1 0 --fc 10 --optim 'sgd' --lrs 0.25 0.15 0.1 0.08 0.05 --wds 3e-4 3e-4 3e-4 3e-4 3e-4 --mmt 0.9 --lr-decay --epochs 120 --act 'my_hard_sig' --todo 'train' --T1 250 --T2 30 --mbs 128 --alg 'EP' --random-sign --betas 0.0 0.5 --loss 'mse' --save --device 0 
```

```
# EP with symmetric gradient estimate
python main.py --model 'CNN' --task 'CIFAR10' --data-aug --channels 128 256 512 512 --kernels 3 3 3 3 --pools 'mmmm' --strides 1 1 1 1 --paddings 1 1 1 0 --fc 10 --optim 'sgd' --lrs 0.25 0.15 0.1 0.08 0.05 --wds 3e-4 3e-4 3e-4 3e-4 3e-4 --mmt 0.9 --lr-decay --epochs 120 --act 'my_hard_sig' --todo 'train' --T1 250 --T2 30 --mbs 128 --alg 'EP' --thirdphase --betas 0.0 0.5 --loss 'mse' --save --device 0 
```

```
# BPTT
python main.py --model 'CNN' --task 'CIFAR10' --data-aug --channels 128 256 512 512 --kernels 3 3 3 3 --pools 'mmmm' --strides 1 1 1 1 --paddings 1 1 1 0 --fc 10 --optim 'sgd' --lrs 0.25 0.15 0.1 0.08 0.05 --wds 3e-4 3e-4 3e-4 3e-4 3e-4 --mmt 0.9 --lr-decay --epochs 120 --act 'my_hard_sig' --todo 'train' --T1 250 --T2 30 --mbs 128 --alg 'BPTT' --loss 'mse' --save --device 0 
```

+ For the training using the Cross Entropy Loss function (relevant arguments `--loss 'cel' --softmax`):

```
# EP with symmetric gradient estimate
python main.py --model 'CNN' --task 'CIFAR10' --data-aug --channels 128 256 512 512 --kernels 3 3 3 3 --pools 'mmmm' --strides 1 1 1 1 --paddings 1 1 1 0 --fc 10 --optim 'sgd' --lrs 0.25 0.15 0.1 0.08 0.05 --wds 3e-4 3e-4 3e-4 3e-4 3e-4 --mmt 0.9 --lr-decay --epochs 120 --act 'my_hard_sig' --todo 'train' --T1 250 --T2 25 --mbs 128 --alg 'EP' --betas 0.0 1.0 --thirdphase --loss 'cel' --softmax --save --device 0 
```

```
# BPTT
python main.py --model 'CNN' --task 'CIFAR10' --data-aug --channels 128 256 512 512 --kernels 3 3 3 3 --pools 'mmmm' --strides 1 1 1 1 --paddings 1 1 1 0 --fc 10 --optim 'sgd' --lrs 0.25 0.15 0.1 0.08 0.05 --wds 3e-4 3e-4 3e-4 3e-4 3e-4 --mmt 0.9 --lr-decay --epochs 120 --act 'my_hard_sig' --todo 'train' --T1 250 --T2 25 --mbs 128 --alg 'BPTT' --loss 'cel' --softmax --save --device 0 
```

+ For the Crossentropy Loss training using dropout run :

```
# EP with symmetric gradient estimate and dropout
python main_dropout.py --model 'CNN' --task 'CIFAR10' --data-aug --channels 128 256 512 512 --kernels 3 3 3 3 --pools 'mmmm' --strides 1 1 1 1 --paddings 1 1 1 0 --fc 10 --optim 'sgd' --lrs 0.25 0.15 0.1 0.08 0.05 --dropouts 1.0 1.0 1.0 0.9 1.0 --wds 3e-4 3e-4 3e-4 3e-4 3e-4 --mmt 0.9 --lr-decay --epochs 120 --act 'my_hard_sig' --todo 'train' --T1 250 --T2 25 --mbs 128 --alg 'EP' --betas 0.0 1.0 --thirdphase --loss 'cel' --softmax --save --device 0 
```

To run BPTT with dropout a GPU with more than 10Gb RAM is required.
```
# BPTT dropout
python main_dropout.py --model 'CNN' --task 'CIFAR10' --data-aug --channels 128 256 512 512 --kernels 3 3 3 3 --pools 'mmmm' --strides 1 1 1 1 --paddings 1 1 1 0 --fc 10 --optim 'sgd' --lrs 0.25 0.15 0.1 0.08 0.05 --dropouts 1.0 1.0 1.0 0.9 1.0 --wds 3e-4 3e-4 3e-4 3e-4 3e-4 --mmt 0.9 --lr-decay --epochs 120 --act 'my_hard_sig' --todo 'train' --T1 250 --T2 25 --mbs 128 --alg 'BPTT' --loss 'cel' --softmax --save --device 0
```


### Training a recurrent CNN on CIFAR-10 with asymmetric connections

EP with different updates between forward and backward weights:

```
python main.py --model 'VFCNN' --task 'CIFAR10' --data-aug --channels 128 256 512 512 --kernels 3 3 3 3 --pools 'mmmm' --strides 1 1 1 1 --paddings 1 1 1 0 --fc 10 --optim 'sgd' --lrs 0.25 0.15 0.1 0.08 0.05 --wds 3e-4 3e-4 3e-4 3e-4 3e-4 --mmt 0.9 --lr-decay --epochs 120 --act 'my_hard_sig' --todo 'train' --T1 250 --T2 30 --mbs 128 --alg 'EP' --betas 0.0 1.0 --thirdphase --loss 'cel' --softmax --save --device 0
```

EP with same update between forward and backward weights:

```
python main.py --model 'VFCNN' --task 'CIFAR10' --data-aug --channels 128 256 512 512 --kernels 3 3 3 3 --pools 'mmmm' --strides 1 1 1 1 --paddings 1 1 1 0 --fc 10 --optim 'sgd' --lrs 0.25 0.15 0.1 0.08 0.05 --wds 3e-4 3e-4 3e-4 3e-4 3e-4 --mmt 0.9 --lr-decay --epochs 120 --act 'my_hard_sig' --todo 'train' --T1 250 --T2 30 --mbs 128 --alg 'EP' --betas 0.0 1.0 --thirdphase --same-update --loss 'cel' --softmax --save --device 0
```

BPTT

```
python main.py --model 'VFCNN' --task 'CIFAR10' --data-aug --channels 128 256 512 512 --kernels 3 3 3 3 --pools 'mmmm' --strides 1 1 1 1 --paddings 1 1 1 0 --fc 10 --optim 'sgd' --lrs 0.25 0.15 0.1 0.08 0.05 --wds 3e-4 3e-4 3e-4 3e-4 3e-4 --mmt 0.9 --lr-decay --epochs 120 --act 'my_hard_sig' --todo 'train' --T1 250 --T2 30 --mbs 128 --alg 'BPTT' --loss 'cel' --softmax --save --device 0
```

## Evaluating

To evaluate a model, simply change the flag `--todo` to  `--todo 'evaluate'` and specify the path to the folder the same way as for resuming training. Train and Test accuracy will be appended to the hyperparameters.txt file.

```
python main.py --model 'CNN' --task 'CIFAR10' --data-aug --todo 'evaluate' --T1 250 --mbs 200 --thirdphase --loss 'mse' --save --device 0 --load-path 'results/test'
```


## Comparing EP and BPTT

EP updates approximates ground truth gradients computed by BPTT. To check if the theorem is satisfied set the `--todo` flag to `--todo 'gducheck'`. With the flag `--save` enabled, plots comparing EP (dashed) and BPTT (solid) updates for each layers will be created in the results folder.

```
python main.py --model 'CNN' --task 'CIFAR10' --data-aug --todo 'gducheck' --T1 250 --T2 15 --mbs 128 --thirdphase --betas 0.0 0.1 --loss 'mse' --save --device 0 --load-path 'results/test'
```

## More command lines

More command line are available at in the check folder of this repository, including training MLP on MNIST.
See the bottom of the page for a summary of all the arguments in the command lines.

## Summary table of the command lines arguments  

|Arguments|Description|Examples|
|-------|------|------|
|`model`|Choose MLP or CNN and Vector field.|`--model 'MLP'`, `--model 'VFMLP'`,`--model 'CNN'`,`--model 'VFCNN'`|
|`task`|Choose the task.|`--task 'MNIST'`, `--task 'CIFAR10'`|
|`data-aug`|Enable data augmentation for CIFAR10.|`--data-aug`|
|`lr-decay`|Enable learning rate decay.|`--lr-decay`|
|`scale`|Multiplication factor for weight initialisation.|`--scale 0.2`|
|`archi`|Layers dimension for MLP.|`--archi 784 512 10`|
|`channels`|Feature maps for CNN.|`--channels 128 256 512`|
|`pools`|Layers wise poolings. `m` is maxpool, `a` is avgpool and `i` is no pooling. All are kernel size 2 and stride 2.|`--pools 'mmm'` for 3 conv layers.|
|`kernels`|Kernel sizes for CNN.|`--kernels 3 3 3`|
|`strides`|Strides for CNN.|`--strides 1 1 1`|
|`paddings`|Padding for conv layers.|`--paddings 1 1 1`|
|`fc`|Linear classifier|`--fc 10` for one fc layer, `--fc 512 10`|
|`act`|Activation function for neurons|`--act 'tanh'`,`'mysig'`,`'hard_sigmoid'`|
|`todo`|Train or check the theorem|`--todo 'train'`,`--todo 'gducheck'`|
|`alg`|EqProp or BackProp Through Time.|`--alg 'EP'`, `--alg 'BPTT'`|
|`check-thm`|Check the theorem while training. (only if EP)|`--check-thm`|
|`T1`,`T2`|Number of time steps for phase 1 and 2.|`--T1 30 --T2 10`|
|`betas`|Beta values beta1 and beta2 for EP phases 1 and 2.|`--betas 0.0 0.1`|
|`random-sign`|Choose a random sign for beta2.|`--random-sign`|
|`thirdphase`|Two phases 2 are done with beta2 and -beta2.|`--thirdphase`|
|`loss`|Loss functions.|`--loss 'mse'`,`--loss 'cel'`, `--loss 'cel' --softmax`|
|`optim`|Optimizer for training.|`--optim 'sgd'`, `--optim 'adam'`|
|`lrs`|Layer wise learning rates.|`--lrs 0.01 0.005`|
|`wds`|Layer wise weight decays. (`None` by default).|`--wds 1e-4 1e-4`|
|`mmt`|Global momentum. (if SGD).|`--mmt 0.9`|
|`epochs`|Number of epochs.|`--epochs 200`|
|`mbs`|Minibatch size|`--mbs 128`|
|`device`|Index of the gpu.|`--device 0`|
|`save`|Create a folder where the accuracys are plotted upon training and the best model is saved.|`--save`|
|`load-path`|Resume the training of a saved simulations.|`--load-path 'results/2020-04-25/10-11-12'`|
|`seed`|Choose the seed.|`--seed 0`|
