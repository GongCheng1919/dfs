# vgg7_64_cifar100 on cifar100 on CPU with dns_ratio 0.5
model_name=vgg7_64_cifar100
batch_size=500
dns_ratio=0.5
useKD=0
useFD=0
useDNS=0
useFR=0
iteration=11
GPU=\'\'
data_name=cifar100
data_root=~/datasets/cifar100

# 解析命令行参数
while getopts ":m:b:d:k:f:n:r:i:g:t:o:" opt; do
  case $opt in
    m) model_name="$OPTARG"
    ;;
    b) batch_size="$OPTARG"
    ;;
    d) dns_ratio="$OPTARG"
    ;;
    k) useKD="$OPTARG"
    ;;
    f) useFD="$OPTARG"
    ;;
    n) useDNS="$OPTARG"
    ;;
    r) useFR="$OPTARG"
    ;;
    i) iteration="$OPTARG"
    ;;
    g) GPU="$OPTARG"
    ;;
    t) data_name="$OPTARG"
    ;;
    o) data_root="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done

# 打印参数
echo "model_name=$model_name"
echo "batch_size=$batch_size"
echo "dns_ratio=$dns_ratio"
echo "useKD=$useKD"
echo "useFD=$useFD"
echo "useDNS=$useDNS"
echo "useFR=$useFR"
echo "iteration=$iteration"
echo "GPU=$GPU"
echo "data_name=$data_name"
echo "data_root=$data_root"

python ../src/evaluate_training_efficiency.py \
 --gpus ${GPU} --dns-ratio ${dns_ratio} \
 --data-name ${data_name} --data-root ${data_root} --autoaugment 1 \
 --model-name ${model_name} --model-desc mtl-training-efficiency-evaluation-v4 --num-tasks 6 \
 --resume 0 --latest-checkpoint ../models/resnet18-dns-test-v1-dns_ratio-0.1-on-cifar100/save_models/latest-training-state-dict.pth.tar \
 --epochs 100 --batch-size ${batch_size} --num-workers 8 \
 --lr 0.1 --lr-decay-type multistep --lr-decay-steps 70-85-90 --lr-decay-rate 0.1 \
 --weight-decay 1e-4 \
 --useKD ${useKD} -T 3.0 --gamma 0.3 \
 --useFD ${useFD} --FD-loss-coefficient 0.03 \
 --use-dns ${useDNS} --use-fr ${useFR} \
 --model-saving-root ../models --log-root ./logs \
 --print-freq 10 --verbose 1 \
 --iteration ${iteration}