import torch
import torchvision
import torchvision.transforms as transforms
from .autoaugment import CIFAR10Policy
from .cutout import Cutout

cifar100_means = [129.3, 124.1, 112.4]
cifar100_stds = [68.2, 65.4, 70.4]
cifar10_means = [125.30691805, 122.95039414, 113.86538318]
cifar10_stds = [62.99321928, 62.08870764, 66.70489964]

def get_cifar_dataset(root='/workspace/datasets/cifar10',dataset = 'cifar10',autoaugment=True, use_cifar10_mean_std=True):

    __dataset_obj__={'cifar10':torchvision.datasets.CIFAR10,'cifar100':torchvision.datasets.CIFAR100}
    # 请注意，由于训练错误，DFS对cifar10和cifar100使用了相同的均值，但是别的模型可能不是，这可能导致不一致的精度。
    __mean__={'cifar10': [0.4914, 0.4822, 0.4465], "cifar100":[0.5071, 0.4867, 0.4408]}
    __std__={'cifar10': [0.2023, 0.1994, 0.2010], "cifar100":[0.2675, 0.2565, 0.2761]}

    dataset_obj = __dataset_obj__[dataset]
    if use_cifar10_mean_std:
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
    else:
        mean = __mean__[dataset]
        std = __std__[dataset]
    # Dataset
    if autoaugment:
        transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4, fill=128),
                             transforms.RandomHorizontalFlip(), CIFAR10Policy(), transforms.ToTensor(),
                             Cutout(n_holes=1, length=16),
                             transforms.Normalize(mean, std)])
    else:
        # transform_train = transforms.Compose([
        #         transforms.Pad(4, padding_mode='reflect'),
        #         transforms.RandomHorizontalFlip(),
        #         transforms.RandomCrop(32),
        #         transforms.ToTensor(),
        #         transforms.Normalize(mean=[x / 255.0 for x in cifar100_means],
        #                                      std=[x / 255.0 for x in cifar100_stds])
        #     ])
        transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4, fill=128),
                                          transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                                          transforms.Normalize(mean, std)])

    # transform_test = transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[x / 255.0 for x in cifar100_means],
    #                                      std=[x / 255.0 for x in cifar100_stds])])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    trainset = dataset_obj(root=root, 
                            train=True, 
                            download=True, 
                            transform=transform_train)
    testset = dataset_obj(root=root, 
                           train=False, 
                           download=True, 
                           transform=transform_test)
    return trainset, testset

# example:
#     trainset, testset = get_cifar10_dataset()

def get_cifar_dataloader(train_batch_size=500,val_batch_size=100, num_workers=4, autoaugment=True,
                         root='/workspace/datasets/cifar10',
                         dataset = 'cifar10', use_cifar10_mean_std=True):

    
    trainset, testset = get_cifar_dataset(root=root,dataset=dataset,autoaugment=autoaugment, use_cifar10_mean_std=use_cifar10_mean_std)
    
    trainloader = torch.utils.data.DataLoader(trainset, 
                                              batch_size=train_batch_size, 
                                              shuffle=True, 
                                              num_workers=num_workers) 
    testloader = torch.utils.data.DataLoader(testset, 
                                             batch_size=val_batch_size, 
                                             shuffle=False, 
                                             num_workers=num_workers)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    return trainloader,testloader,classes

# example:
#     trainloader,testloader,classes = get_cifar10_dataloader()