echo "dns-ratio is ${1} on GPU ${2}"
python ../src/train.py \
 --gpus ${2} --dns-ratio ${1} \
 --data-name imagenet --data-root /open_datasets/imagenet --autoaugment 1 \
 --model-name msdnet_imagenet --model-desc dns-test-v1 --num-tasks 5 \
 --resume 0 --latest-checkpoint ../models/resnet18-dns-test-v1-dns_ratio-0.1-on-cifar100/save_models/latest-training-state-dict.pth.tar \
 --epochs 90 --batch-size 256 --num-workers 4 \
 --lr 0.1 --lr-decay-type multistep --lr-decay-steps 70-80-85 --lr-decay-rate 0.1 \
 --weight-decay 0 \
 --useKD 0 -T 3.0 --gamma 0.9 \
 --useFD 0 --FD-loss-coefficient 0.03 \
 --model-saving-root ../models --log-root ./logs \
 --print-freq 50 --verbose 1
