CUDA_VISIBLE_DEVICES='0' python online_accu_train_adv.py \
                  --batch_size 100 --epoch 100 --test_batch_size 500 --log_name log_test_online_adv.txt\
                  --resume checkpoints_base_bn --use_bn --model_name epoch40.pth \
                  --mode 'eval' --onlinemode 'train' --lr 1e-1 --momentum 0.9 \
                  --beta 1. --only_reg --threshold 0.18 --use_advtrigger --med="OURS"