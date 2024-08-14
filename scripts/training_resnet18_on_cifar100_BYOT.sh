echo "dns-ratio is ${1} on GPU ${2}"
# BYOT on cifar100
python ../src/train.py \
 --gpus ${2} --dns-ratio ${1} \
 --data-name cifar100 --data-root ~/.cache/datasets/cifar100 --autoaugment 1 \
 --model-name resnet18 --model-desc BYOT --num-tasks 4 \
 --resume 0 --latest-checkpoint ../models/resnet18-BYOT-on-cifar100/save_models/latest-training-state-dict.pth.tar \
 --epochs 300 --batch-size 500 \
 --lr 0.1 --lr-decay-type multistep --lr-decay-steps 250-280-295 --lr-decay-rate 0.1 \
 --weight-decay 5e-4 \
 --useKD 1 -T 3.0 --gamma 0.3 \
 --useFD 1 --FD-loss-coefficient 0.03 \
 --use-dns 0 --use-fr 0 \
 --model-saving-root ../models --log-root ./logs \
 --print-freq 50 --verbose 0
