<h1 align="center"> Exploring Model Dynamics for Accumulative Poisoning Discovery </h1>

This repo contains the sample code of our proposed
```Memorization Discrepancy``` and the corresponding ```Discrepancy-awared Sample Correction (DSC)```. The code is developed based on [Accuulative Attack](https://github.com/ShawnXYang/AccumulativeAttack).



## Environment preliminaries

This project is tested under the following environment settings:
- GPU: Geforce 3090 or Tesla V100
- Cuda: 11.4
- Python: 3.6
- PyTorch: >= 1.9.1
- Torchvision: >= 0.10.1

## Running commands

### Burn-in phase
```python
python train_cifar.py
```

### Accumulative poisoning attacks in online learning cases
Below is the original running commands for accumulative phase + poisoned trigger(controlled by `--use_advtrigger`) + online poisoned trigger (controlled by `--use_online_advtrigger`):
```python
python online_accu_train_adv_relate.py \
                  --batch_size 100 --epoch 100 --test_batch_size 500 --log_name log_test_online.txt\
                  --resume checkpoints_base_bn --use_bn --model_name epoch40.pth \
                  --mode 'eval' --onlinemode 'train' --lr 1e-1 --momentum 0.9 \
                  --beta 1. --only_reg --threshold 0.18 --use_advtrigger
```

### ST for Accumulative poisoning attacks
```python
CUDA_VISIBLE_DEVICES='0' python online_accu_train_adv_relate.py \
                  --batch_size 100 --epoch 100 --test_batch_size 500 --log_name log_test_online_adv.txt\
                  --resume checkpoints_base_bn --use_bn --model_name epoch40.pth \
                  --mode 'eval' --onlinemode 'train' --lr 1e-1 --momentum 0.9 \
                  --beta 1. --only_reg --threshold 0.18 --use_advtrigger --med="ST"
```
### AT for Accumulative poisoning attacks
```python
CUDA_VISIBLE_DEVICES='0' python online_accu_train_adv_relate.py \
                  --batch_size 100 --epoch 100 --test_batch_size 500 --log_name log_test_online_adv.txt\
                  --resume checkpoints_base_bn --use_bn --model_name epoch40.pth \
                  --mode 'eval' --onlinemode 'train' --lr 1e-1 --momentum 0.9 \
                  --beta 1. --only_reg --threshold 0.18 --use_advtrigger --med="AT"
```
### DSC for Accumulative poisoning attacks
```python
CUDA_VISIBLE_DEVICES='0' python online_accu_train_adv_relate.py \
                  --batch_size 100 --epoch 100 --test_batch_size 500 --log_name log_test_online_adv.txt\
                  --resume checkpoints_base_bn --use_bn --model_name epoch40.pth \
                  --mode 'eval' --onlinemode 'train' --lr 1e-1 --momentum 0.9 \
                  --beta 1. --only_reg --threshold 0.18 --use_advtrigger --med="OURS"
```

## Reference Code
- Accumulative Attack: https://github.com/ShawnXYang/AccumulativeAttack

- TRADES: https://github.com/yaodongyu/TRADES


---
If you find our paper and repo useful, please cite our paper:
```bibtex
@inproceedings{zhu2023unleashing,
title       ={Exploring Model Dynamics for Accumulative Poisoning Discovery},
author      ={Jianing Zhu and Xiawei Guo and Jiangchao Yao and Chao Du and Li He and Shuai Yuan and Tongliang Liu and Liang Wang and Bo Han},
booktitle   ={International Conference on Machine Learning},
year        ={2023}
}
```
