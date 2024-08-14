import argparse

arg_parser = argparse.ArgumentParser(
                description='Image classification for DNS')

# envaroment args, such as GPUs
env_group =arg_parser.add_argument_group('ENV','enviroment arguments')
env_group.add_argument('--seed', default=0, type=int,
                       help='random seed (default: 0)')
env_group.add_argument('--gpus',default="0", type=str,
                    help='GPU available.')
env_group.add_argument('--verbose',default=0,type=int,
        help="whether print logging to screen")


# model args, such as model architecture
model_names = ['resnet18','msdnet_cifar100','msdnet_imagenet',"sdn_mtl_resnet18",
               "vgg64_7_plane_mtl",
               "vgg_dsn","vgg_dns_wo_fr","vgg_dns_w_fr",
               "vgg7_64_cifar100","vgg16_64_imagenet"]
model_group =arg_parser.add_argument_group('MODEL','model arguments')
model_group.add_argument('--model-name', metavar='D', default='resnet18',
                        choices=model_names,
                        help='model to work on')
model_group.add_argument('--model-desc', default="v1",type=str,
                        help='draft description for this model (will change save name)',required=False)
model_group.add_argument('--dns-ratio', default=0.5, type=float, metavar='M',
                         help='dns_raiot for MTL model training (default:0.5)')
model_group.add_argument('--num-tasks', default=4, type=int,
                         metavar='N', help='the number of tasks (default: 4 for resnet18, 7 for msdnet cifar, and 5 for msdnet iamgenet)')
model_group.add_argument('--use-mtl',default=1,type=int,help="whether use mtl loss in MTL training")
model_group.add_argument('--use-dns',default=1,type=int,help="whether use dns in MTL training")
model_group.add_argument('--use-fr',default=1,type=int,help="whether use fr in MTL training")

# data args
data_names = ["cifar100","imagenet"]
data_group =arg_parser.add_argument_group('DATA','dataset arguments')
data_group.add_argument('--data-name', metavar='D', default='cifar10',
                        choices=data_names,
                        help='data to work on')
data_group.add_argument('--data-root', metavar='DIR', default='./data/cifar/',
                        help='path to dataset (default: ./data/cifar)')
data_group.add_argument('-b', '--batch-size', default=500, type=int,
                         metavar='N', help='mini-batch size (default: 64)')
data_group.add_argument('--autoaugment',default=1,type=int,
                        help="enable automatic data augument for cifar dataset")


# traninig args
training_group =arg_parser.add_argument_group('TRAINING','traninig arguments')

# KD details
training_group.add_argument('--useKD',default=0,type=int,
                            required=False,help="Use Knowledge Distillation in training")
training_group.add_argument('--gamma', default=0.9, type=float, metavar='M',
                         help='gamma for KD loss (default 0.3 for resnet18 and 0.9 for msdnet)')
training_group.add_argument('-T', default=3.0, type=float, metavar='M',
                         help='Temperature for KD (default 3.0)')
training_group.add_argument('--useFD',default=0,type=int,
                            required=False,help="Use Feature Distillation in training")
training_group.add_argument('--FD-loss-coefficient', default=0.03, type=float, metavar='M',
                         help='loss coefficient for FD (default 0.03)')


## traninig details
training_group.add_argument('--print-freq', '-p', default=100, type=int,
                       metavar='N', help='print frequency (default: 100)')
training_group.add_argument('-j', '--num-workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
training_group.add_argument('--epochs', default=200, type=int, metavar='N',
                         help='number of total epochs to run (default: 200)')
training_group.add_argument('--start-epoch', default=0, type=int, metavar='N',
                         help='manual epoch number (useful on restarts)')

## optimizer details
training_group.add_argument('--optimizer', default='sgd',
                         choices=['sgd', 'rmsprop', 'adam'], metavar='N',
                         help='optimizer (default=sgd)')
training_group.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                         metavar='LR',
                         help='initial learning rate (default: 0.1)')
training_group.add_argument('--lr-decay-type', default='multistep', type=str, metavar='T',
                        help='learning rate strategy (default: multistep)',
                        choices=['cosine', 'multistep'])
training_group.add_argument('--lr-decay-steps', default='150-180-190', type=str, metavar='T',
                        help='learning rate decay steps (default: 150-180-190)')
training_group.add_argument('--lr-decay-rate', default=0.1, type=float, metavar='N',
                         help='decay rate of learning rate (default: 0.1)')
training_group.add_argument('--momentum', default=0.9, type=float, metavar='M',
                         help='momentum (default=0.9)')
training_group.add_argument('--alpha', default=0.99, type=float, metavar='M',
                         help='alpha for ')
training_group.add_argument('--beta1', default=0.9, type=float, metavar='M',
                         help='beta1 for Adam (default: 0.9)')
training_group.add_argument('--beta2', default=0.999, type=float, metavar='M',
                         help='beta2 for Adam (default: 0.999)')
training_group.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                         metavar='W', help='weight decay (default: 1e-4)')

# saving and logging args
logging_group =arg_parser.add_argument_group('SAVING and LOGGING','SAVING arguments')
logging_group.add_argument('--log-root', metavar='DIR', default='./logs/',
                        help='path to logging (default: ./logs/)')
logging_group.add_argument('--model-saving-root', metavar='DIR', default='./models/',
                        help='path to logging (default: ./models/)')
logging_group.add_argument('--resume', default=0,type=int,
                        help='resuem from latest checkpoint (default: 0)')
logging_group.add_argument('--no-resume-opt', action='store_true', default=False,
                   help='prevent resume of optimizer state when resuming model')
logging_group.add_argument('--latest-checkpoint', default="",type=str,
                        help='path to latest checkpoint',required=False)

if __name__ == '__main__':
    args = arg_parser.parse_args()
    print(args)