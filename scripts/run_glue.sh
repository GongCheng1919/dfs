TASK_NAME=${1} # mrpc
USE_DNS=${2} # 1
# 将字符串转换为整数
# ((USE_DNS = ${2}))
DNS_RATIO=${3} # 0.5
# DNS_RATIO=$(echo "scale=2; ${3}" | bc)
NUM_TRAINING_EPOCHS=${4} # 5

# 设置为fp16，则batchsize和lr都要加倍才行
python run_glue_no_trainer.py \
  --model_name_or_path bert-base-cased \
  --task_name $TASK_NAME \
  --max_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs $NUM_TRAINING_EPOCHS \
  --output_dir ./tmp/$TASK_NAME/ \
  --mixed-precision fp16 \
  --use-dns $USE_DNS \
  --dns-ratio $DNS_RATIO \
  --gpus 1
