# tanh, beta 0.01
python main.py --model 'MLP' --task 'MNIST' --archi 784 512 10 --act 'tanh' --todo 'gducheck' --betas 0.0 0.01 --T1 30 --T2 10 --mbs 50 --loss 'mse' --save --device 7 --seed 0
# tanh, beta 0.1
python main.py --model 'MLP' --task 'MNIST' --archi 784 512 10 --act 'tanh' --todo 'gducheck' --betas 0.0 0.1 --T1 30 --T2 10 --mbs 50 --loss 'mse' --save --device 7 --seed 0
# tanh, beta 0.1, and beta -0.1
python main.py --model 'MLP' --task 'MNIST' --archi 784 512 10 --act 'tanh' --todo 'gducheck' --betas 0.0 0.1 --T1 30 --T2 10 --mbs 50 --loss 'mse' --save --thirdphase --device 7 --seed 0
