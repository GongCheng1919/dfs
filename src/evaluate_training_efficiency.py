
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import shutil
import torch
import random
import datasets
from torch import nn,optim
import torch.nn.functional as F
from utils.train_utils import AverageMeter
import logging
from pprint import pprint

# arg_parser.add_argument

formatter = logging.Formatter('%(asctime)s [%(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
# 创建一个日志记录器
logger = logging.getLogger('evaluate training efficiency')
logger.setLevel(logging.DEBUG)
# 创建一个文件处理器，并Formatter 添加到处理器
# file_handler = logging.FileHandler('experiments.log')
# file_handler.setFormatter(formatter)
# 创建一个控制台处理器，并将 Formatter 添加到处理器
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
# 将处理器添加到记录器
# logger.addHandler(file_handler)
logger.addHandler(console_handler)
# logging.basicConfig(
#     level=logging.DEBUG,  # 设置日志级别，可以选择 DEBUG、INFO、WARNING、ERROR、CRITICAL
#     format='%(asctime)s [%(levelname)s] %(message)s',  # 设置日志格式
#     handlers=[
#         logging.StreamHandler()  # 输出到控制台
#     ]
# )

from arguments import arg_parser
from models.dns_msdnet import msdnet_cifar100, msdnet_imagenet
from models.dns_resnet import resnet18
from models.sdn_resnet import resnet18 as sdn_mtl_resnet18
from models.dns_vgg import vgg64_7_plane_mtl,vgg_dsn,vgg_dns_wo_fr,vgg_dns_w_fr,vgg7_64_cifar100,vgg16_64_imagenet
from train import KDCrossEntropy,FDDistributionLoss,MTLLoss
from tqdm import tqdm

arg_parser.add_argument('--iteration', type=int, default=100,
                        help='number of iteration for evaluating training efficiency')

def stat_FB_time(model,optimizer,trainloader,loss_fn,args,verbose=1,iteration=10):
    f_time = 0
    b_time = 0
    data_time = 0
    other_time = 0
    start_iter = 1
    # iteration = 100
    iteration = min(iteration,len(trainloader))
    start_time = time.perf_counter()
    if args.device.startswith("cuda"):
        # 创建 CUDA events 用于计时
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
    # for i in range(iteration):
    for i, (inputs, targets) in tqdm(enumerate(trainloader), total=iteration, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{percentage:3.0f}%]'):
    # for i, (inputs, targets) in enumerate(trainloader):
        if i>=iteration:
            break
        inputs = inputs.to(args.device)
        targets = targets.to(args.device)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize() # 同步以确保计时准确
        if i>=start_iter:
            # targets[0,0,0,0].item() # 同步以确保计时准确
            data_time += (time.perf_counter()-start_time)
        
        start_time = time.perf_counter()
        if args.device.startswith("cuda"):
            # 前向传播计时
            start_event.record()
        outputs,middle_features = model(inputs)
        if args.device.startswith("cuda"):
            end_event.record()
            torch.cuda.synchronize() # 同步以确保计时准确，等待前向传播完成
            forward_time = start_event.elapsed_time(end_event)
            if i>=start_iter:
                f_time+=forward_time/1000 # ms->s
        elif i>=start_iter:
            # outputs[0,0,0,0].item() # 同步以确保计时准确
            f_time+=(time.perf_counter()-start_time)
            
        start_time = time.perf_counter()
        if args.device.startswith("cuda"):
            # 计时
            start_event.record()
        # loss = nn.functional.nll_loss(outputs, targets)
        loss = loss_fn(outputs[:-1],middle_features[:-1],
                outputs[-1],middle_features[-1],targets)
        optimizer.zero_grad()
        if args.device.startswith("cuda"):
            end_event.record()
            torch.cuda.synchronize() # 同步以确保计时准确，等待前向传播完成
            run_time = start_event.elapsed_time(end_event)
            if i>=start_iter:
                other_time+=run_time/1000 # ms->s
        elif i>=start_iter:
            # loss.item() # 同步以确保计时准确
            other_time += (time.perf_counter()-start_time)

        
        start_time = time.perf_counter()
        if args.device.startswith("cuda"):
            # 反向传播计时
            start_event.record()
        loss.backward()
        if args.device.startswith("cuda"):
            end_event.record()
            torch.cuda.synchronize() # 同步以确保计时准确，等待反向传播完成
            run_time = start_event.elapsed_time(end_event)
            if i>=start_iter:
                b_time+=run_time/1000 # ms->s
        elif i>=start_iter:
            # optimizer.param_groups[0]['params'][0].grad[[0,0,0,0]].item()# 同步以确保计时准确，
            b_time+=(time.perf_counter()-start_time)

        start_time = time.perf_counter()
        if args.device.startswith("cuda"):
            # 反向传播计时
            start_event.record()
        optimizer.step()
        if args.device.startswith("cuda"):
            end_event.record()
            torch.cuda.synchronize() # 同步以确保计时准确，等待前向传播完成
            run_time = start_event.elapsed_time(end_event)
            if i>=start_iter:
                other_time+=run_time/1000 # ms->s
        elif i>=start_iter:
            # optimizer.param_groups[0]['params'][0][0,0,0,0].item()# 同步以确保计时准确，
            other_time += (time.perf_counter()-start_time)

        start_time = time.perf_counter()
    if verbose:
        print(f"forward time:", f_time)
        print(f"backward time:", b_time)
        print(f"data time:", data_time)
        print(f"other time:", other_time)
    return f_time/(iteration-start_iter),b_time/(iteration-start_iter),data_time/(iteration-start_iter),other_time/(iteration-start_iter)

def test_forward_backward_time(model, trainloader, loss_fn, optimizer,
                               device, args, 
                               iteration=10,verbose=1):
    # # vgg_dns_w_fr(0.1)
    # model = vgg_dsn()
    # model = vgg_dns_wo_fr()
    # model = vgg_dns_w_fr(dns_ratio=0.3)
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = "cuda:0"# 
    # model = dns_vit
    # device = "cpu"
    # summary(model.to(device), (3, 32, 32))
    # 定义输入张量
    # batch_size = 500
    model = model.to(device)
    f_time,b_time,data_time,other_time = stat_FB_time(model,optimizer,trainloader,loss_fn=loss_fn,args=args,verbose=verbose,iteration=iteration)
    return f_time,b_time,data_time,other_time

def main(args):

    logger.info("\n# logging setting\n")
    os.makedirs(args.log_root,exist_ok=True)
    logging_file = os.path.join(args.log_root,
                    f"{args.model_name}-{args.model_desc}-dns_ratio-{args.dns_ratio}-evaluate-training-efficiency.log")
    # 配置日志
    # logging.basicConfig(
    #     level=logging.DEBUG,  # 设置日志级别，可以选择 DEBUG、INFO、WARNING、ERROR、CRITICAL
    #     format='%(asctime)s [%(levelname)s] %(message)s',  # 设置日志格式
    #     handlers=[
    #         logging.FileHandler(logging_file),  # 输出到文�?
    #         logging.StreamHandler()  # 输出到控制台
    #     ]
    # )
    file_handler = logging.FileHandler(logging_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    if not args.verbose:
        print(f"## Stop output to screen, please see the log in {logging_file}")
        logger.removeHandler(console_handler)
    logger.info("\n#######################Training Begin#######################\n")
    logger.info(f"## Saving log to {logging_file} for this experiments")

    logger.info("\n# Stage 1: Test enviroment\n")
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpus
    # 检GPU 是否可用  
    gpuinfo=""
    if torch.cuda.is_available():  
        logger.info("GPU is available")  
        # 获取 GPU 设备数量  
        num_gpus = torch.cuda.device_count()  
        logger.info(f"Having {num_gpus} GPU devices")
        # 获取 GPU 设备信息  
        
        for i in range(num_gpus):  
            logger.info(f"Using GPU{i}: {torch.cuda.get_device_properties(i)}")  
            gpuinfo+=f"{torch.cuda.get_device_properties(i)}"
    else:  
        logger.info("GPU is not available")  
        gpuinfo+="GPU is not available"
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    seed = args.seed
    # 设置随机种子  
    torch.manual_seed(seed)  
    random.seed(seed)
    # 设置随机种子以确�? GPU 上的计算具有确定性行�?  
    torch.backends.cudnn.deterministic = True  
    logger.info(f"## Employ {device} with seed {seed} for this experiments")

    logger.info("\n# Stage 2: processing model architecture\n")
    model_name = args.model_name
    dns_ratio = max(min(args.dns_ratio,1),0)
    model = eval(model_name)(use_dns=args.use_dns,
                             use_fr=args.use_fr,
                             dns_ratio=dns_ratio).to(device)
    model_desc = args.model_desc
    model_save_name = f"{model_name}-{model_desc}"
    nBlocks = args.num_tasks
    logger.info(f"## Employ {model_save_name} model with dns-raio={dns_ratio:.2f} for this experiments")

    logger.info("\n# Stage 3: preprocessing dataset\n")
    # torch_weights_path=f"{os.environ['HOME']}/models/torch/weights"
    data_root=args.data_root
    dataset_name = args.data_name
    batch_size = args.batch_size
    num_workers = args.num_workers
    autoaugment=args.autoaugment
    if dataset_name.startswith("cifar10"):
        trainloader,testloader,classes = datasets.cifar.get_cifar_dataloader(
            train_batch_size=batch_size,val_batch_size=batch_size,
            root=data_root,dataset=dataset_name,
            autoaugment=autoaugment)
    elif dataset_name.startswith("imagenet"):
        trainloader,testloader = datasets.imagenet.get_dataset(
            data_path=data_root,batch_size=batch_size, workers=num_workers, 
            parse_type="torch",prefetch=True)
    else:
        logger.error(f"unkown data name {dataset_name}")
        exit
    logger.info(f"## Employ {dataset_name} dataset for this experiments")

    logger.info("\n# Stage 4: preparing optimizer and loss function \n")
    useKD = args.useKD
    useFD = args.useFD
    loss_coefficient=args.gamma
    feature_loss_coefficient=args.FD_loss_coefficient
    epochs=args.epochs
    temperature=args.T
    batchsize=args.batch_size
    init_lr=args.lr
    lr_decay_method=args.lr_decay_type
    lr_decay_rate = args.lr_decay_rate
    lr_decay_steps = [int(v) for v in args.lr_decay_steps.split("-")]
    momentum = args.momentum
    weight_decay=args.weight_decay
    criterion = nn.CrossEntropyLoss()
    if args.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=init_lr, weight_decay=weight_decay, momentum=momentum)
    elif args.optimizer == "rmsprop":
        raise ValueError("Not Implemenetation for {args.optimizer} optimizer")
    elif args.optimizer == "adam":
        raise ValueError("Not Implemenetation for {args.optimizer} optimizer")
    # init = False
    
    loss_fn = MTLLoss(nn.CrossEntropyLoss().to(device),useKD,loss_coefficient,temperature,
                      useFD,feature_loss_coefficient)
    
    logger.info(f'## Employ {args.optimizer} optimizer with (init_lr={init_lr},weight_decay={weight_decay},momentum={momentum})'
                f'and MTLLoss with (useKD={useKD},temperature={temperature},loss_coefficient={loss_coefficient},useFD={useFD},feature_loss_coefficient={feature_loss_coefficient}) for this experiments')
    
    # evaluating process
    f_time,b_time,data_time,other_time = test_forward_backward_time(model,trainloader, loss_fn, optimizer, device, args, args.iteration,args.verbose)

    logger.info("\n# Stage 5: evaluating training efficiency\n")
    logger.info(f"## Model: {args.model_name}-use_dns-{args.use_dns}-{args.dns_ratio}-use_fr-{args.use_fr}-useKD-{args.useKD}-useFD-{args.useFD}")
    logger.info(f"## BatchSize: {args.batch_size} ")
    logger.info(f"## GPU: {gpuinfo} ")
    logger.info(f"## Iteartion: {args.iteration} ")
    logger.info(f"## Forward time: {f_time:.4f} s")
    logger.info(f"## Backward time: {b_time:.4f} s")
    logger.info(f"## Data time: {data_time:.4f} s")
    logger.info(f"## Other time: {other_time:.4f} s")
    logger.info(f"## Total time: {f_time+b_time+data_time+other_time:.4f} s")
    logger.info(f"## FPS: {args.batch_size/(f_time+b_time+data_time+other_time):.4f}")
    logger.info(f"Saving results to {logging_file} for this experiments")

    logger.info("\n#######################Evaluate Finished#######################\n")
    logger.removeHandler(file_handler)

if __name__ == '__main__':
    args = arg_parser.parse_args()
    pprint(vars(args))
    main(args)
