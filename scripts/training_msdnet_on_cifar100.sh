echo "dns-ratio is ${1} on GPU ${2}"
python ../src/train.py \
 --gpus ${2} --dns-ratio ${1} \
 --data-name cifar100 --data-root ~/.cache/datasets/cifar100 --autoaugment 0 \
 --model-name msdnet_cifar100 --model-desc dns-test-v1 --num-tasks 7 \
 --resume 0 --latest-checkpoint ../models/msdnet_cifar100-dns-test-v1-dns_ratio-0.5-on-cifar100-top1-74.82000160217285.pth \
 --epochs 300 --batch-size 500 --start-epoch 0 \
 --lr 0.1 --lr-decay-type multistep --lr-decay-steps 250-280-295 --lr-decay-rate 0.1 \
 --weight-decay 5e-4 \
 --useKD 0 -T 3.0 --gamma 0.9 \
 --useFD 0 --FD-loss-coefficient 0.03 \
 --model-saving-root ../models --log-root ./logs \
 --print-freq 50 --verbose 1
