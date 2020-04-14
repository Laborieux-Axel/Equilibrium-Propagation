# Equilibrium Propagation

Reproducing some results of https://arxiv.org/pdf/1905.13633.pdf  

(test: I can edit the README.md)

Check GDU:  
> python main.py --model 'MLP' --task 'MNIST' --archi 784 512 10 --act 'tanh' --todo 'gducheck' --betas 0.0 0.01 --T1 30 --T2 10 --mbs 50 --device 0  

Train:  
> python main.py --model 'MLP' --task 'MNIST' --archi 784 512 10 --lrs 0.08 0.04 --epochs 3 --act 'mysig' --todo 'train' --betas 0.0 0.1 --T1 30 --T2 10 --mbs 50 --device 0  


Scaling to CIFAR10:  

Check GDU:  
> python main.py --model 'CNN' --task 'CIFAR10' --act 'hard_sigmoid' --pool 'avg' --todo 'gducheck' --betas 0.0 0.4 --T1 200 --T2 10 --device 0  

Train:  
> python main.py --model 'CNN' --task 'CIFAR10' --lrs 0.08 0.04 0.01 0.005 0.001 --epochs 1 --act 'hard_sigmoid' --pool 'avg'  --todo 'train' --betas 0.0 0.4 --T1 150 --T2 10 --mbs 100 --device 0  


