# mse, small beta
python main.py --model 'CNN' --task 'CIFAR10' --act 'my_hard_sig' --kernels 3 3 3 3 --channels 128 256 512 512 --pools 'mmmm' --strides 1 1 1 1 --paddings 1 1 1 0 --todo 'gducheck' --betas 0.0 0.01 --T1 250 --T2 30 --loss 'mse' --save --device 0
# mse, big beta
python main.py --model 'CNN' --task 'CIFAR10' --act 'my_hard_sig' --kernels 3 3 3 3 --channels 128 256 512 512 --pools 'mmmm' --strides 1 1 1 1 --paddings 1 1 1 0 --todo 'gducheck' --betas 0.0 0.1 --T1 250 --T2 30 --loss 'mse' --save --device 0
# mse, big beta symmetric
python main.py --model 'CNN' --task 'CIFAR10' --act 'my_hard_sig' --kernels 3 3 3 3 --channels 128 256 512 512 --pools 'mmmm' --strides 1 1 1 1 --paddings 1 1 1 0 --todo 'gducheck' --betas 0.0 0.1 --T1 250 --T2 30 --loss 'mse' --save --device 0 --thirdphase

