import torch.nn as nn
import torch
import math
from .feature_reroute import feature_reroute_func

# ---- FeatureReroute ---- #

# class FeatureReroute(nn.Module):
#     '''
#     Split the feature into two parts with the ratio along channel dimension,
    
#     '''
#     def __init__(self, num_channel, ratio=0.5):
#         super().__init__()
#         self.ratio = max(min(ratio,1),0)
#         self.num_channel = num_channel
#         self.feature_num1 = round(num_channel*ratio)
#         self.feature_num2 = num_channel - self.feature_num1
        
#     def forward(self,x): # 默认为nchw维度，对c维度进行切分
#         # 特征分割
#         tmp1, tmp2 = torch.split(x,[self.feature_num1,self.feature_num2],dim=1)
#         # 特征重组
#         x_plus = torch.cat([tmp1,tmp2.detach()],1)
#         x_sub = torch.cat([tmp1.detach(),tmp2],1)
#         # print(f"Using dns ratio is {self.ratio}. The shared featrue number is {self.feature_num1} and unshared one is {self.feature_num2}")
#         return x_plus,x_sub  
    
# def feature_reroute_func(x, ratio=0.5):
#     return FeatureReroute(x.size(1), ratio)(x)

# class FeaturePartitioning(nn.Module):
#     '''
#     Split the feature into two parts with the ratio along channel dimension,
    
#     '''
#     def __init__(self, num_channel, ratio=0.5):
#         super().__init__()
#         self.ratio = max(min(ratio,1),0)
#         self.num_channel = num_channel
#         self.feature_num1 = round(num_channel*ratio)
#         self.feature_num2 = num_channel - self.feature_num1
        
#     def forward(self,x): # 默认为nchw维度，对c维度进行切分
#         # 特征分割
#         x_plus, x_sub = torch.split(x,[self.feature_num1,self.feature_num2],dim=1)
#         return x_plus,x_sub 

# class FeaturePartitioningFake(nn.Module):
#     '''
#     Split the feature into two parts with the ratio along channel dimension,
#     使用�?0的tensor来填充丢失的特征，以模拟更多的feature，保证模型结构不会被更改
#     '''
#     def __init__(self, num_channel, ratio=0.5):
#         super().__init__()
#         self.ratio = max(min(ratio,1),0)
#         self.num_channel = num_channel
#         self.feature_num1 = round(num_channel*ratio)
#         self.feature_num2 = num_channel - self.feature_num1
#         self.fake_tensor1 = torch.zeros()
#         self.fake_tensor2 = torch.zeros()
        
#     def forward(self,x): # 默认为nchw维度，对c维度进行切分
#         tmp1, tmp2 = torch.split(x,[self.feature_num1,self.feature_num2],dim=1)
#         x_plus = torch.cat([tmp1,torch.zeros_like(tmp2)],1)
#         x_sub = torch.cat([torch.zeros_like(tmp1),tmp2],1)
#         return x_plus,x_sub  
    
# class FeatureRerouteFunction(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input, ratio=0.5):
#         ctx.ratio = max(min(ratio,1),0)
#         ctx.num_channel = ctx.num_channel
#         ctx.feature_num1 = int(ctx.num_channel*ctx.ratio)
#         ctx.feature_num2 = ctx.num_channel - ctx.feature_num1
#         output1 = input
#         output2 = input
#         return output1, output2

#     # 各遮一半实现的话感觉，梯度计算量并不会降低，因为对grad_output1和grad_output2都会计算全部的梯度，可能还真不如detach版本�?
#     @staticmethod
#     def backward(ctx, grad_output1, grad_output2):
#         input = ctx.saved_tensors
#         grad_output1[:,0:ctx.feature_num1 ,:,:] = 0
#         grad_output2[:,ctx.feature_num1:ctx.num_channel,:,:] = 0
#         grad_input = grad_output1 + grad_output2
#         grad_ratio = None
#         return grad_input, grad_ratio
    
# feature_reroute = FeatureRerouteFunction.apply

# class FeaturePartitioningFunction(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input, ratio=0.5):
#         ctx.ratio = max(min(ratio,1),0)
#         ctx.num_channel = ctx.num_channel
#         ctx.feature_num1 = round(ctx.num_channel*ctx.ratio)
#         ctx.feature_num2 = ctx.num_channel - ctx.feature_num1
#         return torch.split(x,[self.feature_num1,self.feature_num2],dim=1)

#     # 各遮一半实现的话感觉，梯度计算量并不会降低，因为对grad_output1和grad_output2都会计算全部的梯度，可能还真不如detach版本�?
#     @staticmethod
#     def backward(ctx, grad_output1, grad_output2):
#         grad_ratio = None
#         return torch.cat([grad_output1,grad_output2],1), grad_ratio
    
# feature_partitioning = FeaturePartitioningFunction.apply

# # ---- END FeatureReroute ---- #

class ConvBasic(nn.Module):
    def __init__(self, nIn, nOut, kernel=3, stride=1,
                 padding=1):
        super(ConvBasic, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(nIn, nOut, kernel_size=kernel, stride=stride,
                      padding=padding, bias=False),
            nn.BatchNorm2d(nOut),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.net(x)


class ConvBN(nn.Module):
    def __init__(self, nIn, nOut, type: str, bottleneck,
                 bnWidth):
        """
        a basic conv in MSDNet, two type
        :param nIn:
        :param nOut:
        :param type: normal or down
        :param bottleneck: use bottlenet or not
        :param bnWidth: bottleneck factor
        """
        super(ConvBN, self).__init__()
        layer = []
        nInner = nIn
        if bottleneck is True:
            nInner = min(nInner, bnWidth * nOut)
            layer.append(nn.Conv2d(
                nIn, nInner, kernel_size=1, stride=1, padding=0, bias=False))
            layer.append(nn.BatchNorm2d(nInner))
            layer.append(nn.ReLU(True))

        if type == 'normal':
            layer.append(nn.Conv2d(nInner, nOut, kernel_size=3,
                                   stride=1, padding=1, bias=False))
        elif type == 'down':
            layer.append(nn.Conv2d(nInner, nOut, kernel_size=3,
                                   stride=2, padding=1, bias=False))
        else:
            raise ValueError

        layer.append(nn.BatchNorm2d(nOut))
        layer.append(nn.ReLU(True))

        self.net = nn.Sequential(*layer)

    def forward(self, x):
        # print(self.net, x.size())
        # pdb.set_trace()
        return self.net(x)


class ConvDownNormal(nn.Module):
    def __init__(self, nIn1, nIn2, nOut, bottleneck, bnWidth1, bnWidth2):
        super(ConvDownNormal, self).__init__()
        self.conv_down = ConvBN(nIn1, nOut // 2, 'down',
                                bottleneck, bnWidth1)
        self.conv_normal = ConvBN(nIn2, nOut // 2, 'normal',
                                   bottleneck, bnWidth2)

    def forward(self, x):
        # print(self.conv_down, self.conv_normal, '========')
        # pdb.set_trace()
        res = [x[1],
               self.conv_down(x[0]),
               self.conv_normal(x[1])]
        # print(res[0].size(), res[1].size(), res[2].size())
        return torch.cat(res, dim=1)


class ConvNormal(nn.Module):
    def __init__(self, nIn, nOut, bottleneck, bnWidth):
        super(ConvNormal, self).__init__()
        self.conv_normal = ConvBN(nIn, nOut, 'normal',
                                   bottleneck, bnWidth)

    def forward(self, x):
        if not isinstance(x, list):
            x = [x]
        res = [x[0],
               self.conv_normal(x[0])]

        return torch.cat(res, dim=1)


class MSDNFirstLayer(nn.Module):
    def __init__(self, nIn, nOut, args):
        super(MSDNFirstLayer, self).__init__()
        self.layers = nn.ModuleList()
        if args.data.startswith('cifar'):
            self.layers.append(ConvBasic(nIn, nOut * args.grFactor[0],
                                         kernel=3, stride=1, padding=1))
        elif args.data == 'ImageNet':
            conv = nn.Sequential(
                    nn.Conv2d(nIn, nOut * args.grFactor[0], 7, 2, 3),
                    nn.BatchNorm2d(nOut * args.grFactor[0]),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(3, 2, 1))
            self.layers.append(conv)

        nIn = nOut * args.grFactor[0]

        for i in range(1, args.nScales):
            self.layers.append(ConvBasic(nIn, nOut * args.grFactor[i],
                                         kernel=3, stride=2, padding=1))
            nIn = nOut * args.grFactor[i]

    def forward(self, x):
        res = []
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            res.append(x)

        return res


class MSDNLayer(nn.Module):
    def __init__(self, nIn, nOut, args, inScales=None, outScales=None):
        super(MSDNLayer, self).__init__()
        self.nIn = nIn
        self.nOut = nOut
        self.inScales = inScales if inScales is not None else args.nScales
        self.outScales = outScales if outScales is not None else args.nScales

        self.nScales = args.nScales
        self.discard = self.inScales - self.outScales

        self.offset = self.nScales - self.outScales
        self.layers = nn.ModuleList()

        if self.discard > 0:
            nIn1 = nIn * args.grFactor[self.offset - 1]
            nIn2 = nIn * args.grFactor[self.offset]
            _nOut = nOut * args.grFactor[self.offset]
            self.layers.append(ConvDownNormal(nIn1, nIn2, _nOut, args.bottleneck,
                                              args.bnFactor[self.offset - 1],
                                              args.bnFactor[self.offset]))
        else:
            self.layers.append(ConvNormal(nIn * args.grFactor[self.offset],
                                          nOut * args.grFactor[self.offset],
                                          args.bottleneck,
                                          args.bnFactor[self.offset]))

        for i in range(self.offset + 1, self.nScales):
            nIn1 = nIn * args.grFactor[i - 1]
            nIn2 = nIn * args.grFactor[i]
            _nOut = nOut * args.grFactor[i]
            self.layers.append(ConvDownNormal(nIn1, nIn2, _nOut, args.bottleneck,
                                              args.bnFactor[i - 1],
                                              args.bnFactor[i]))

    def forward(self, x):
        if self.discard > 0:
            inp = []
            for i in range(1, self.outScales + 1):
                inp.append([x[i - 1], x[i]])
        else:
            inp = [[x[0]]]
            for i in range(1, self.outScales):
                inp.append([x[i - 1], x[i]])

        res = []
        for i in range(self.outScales):
            res.append(self.layers[i](inp[i]))

        return res


class ParallelModule(nn.Module):
    """
    This module is similar to luatorch's Parallel Table
    input: N tensor
    network: N module
    output: N tensor
    """
    def __init__(self, parallel_modules):
        super(ParallelModule, self).__init__()
        self.m = nn.ModuleList(parallel_modules)

    def forward(self, x):
        res = []
        for i in range(len(x)):
            res.append(self.m[i](x[i]))

        return res


class ClassifierModule(nn.Module):
    def __init__(self, m, channel, num_classes):
        super(ClassifierModule, self).__init__()
        self.m = m
        # self.linear = nn.Linear(channel, num_classes)
        self.linear = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(channel, num_classes)
        )



    def forward(self, x):
        res = self.m(x[-1])
        res = res.view(res.size(0), -1)
        return self.linear(res), res

class MSDNet(nn.Module):
    def __init__(self, args):
        super(MSDNet, self).__init__()
        self.blocks = nn.ModuleList()
        self.classifier = nn.ModuleList()
        self.nBlocks = args.nBlocks
        self.steps = [args.base]
        self.args = args
        # todo: how many block?
        n_layers_all, n_layer_curr = args.base, 0
        for i in range(1, self.nBlocks):
            self.steps.append(args.step if args.stepmode == 'even'
                             else args.step * i + 1)
            n_layers_all += self.steps[-1]

        print("building network of steps: ")
        print(self.steps, n_layers_all)

        nIn = args.nChannels
        for i in range(self.nBlocks):
            print(' ********************** Block {} '
                  ' **********************'.format(i + 1))
            m, nIn = \
                self._build_block(nIn, args, self.steps[i],
                                  n_layers_all, n_layer_curr)
            self.blocks.append(m)
            n_layer_curr += self.steps[i]

            if args.data.startswith('cifar100'):
                self.classifier.append(
                    self._build_classifier_cifar(nIn * args.grFactor[-1], 100))
            elif args.data.startswith('cifar10'):
                self.classifier.append(
                    self._build_classifier_cifar(nIn * args.grFactor[-1], 10))
            elif args.data == 'ImageNet':
                self.classifier.append(
                    self._build_classifier_imagenet(nIn * args.grFactor[-1], 1000))
            else:
                raise NotImplementedError
                
        # adding initialization functions
        for m in self.blocks:
            if hasattr(m, '__iter__'):
                for _m in m:
                    self._init_weights(_m)
            else:
                self._init_weights(m)

        for m in self.classifier:
            if hasattr(m, '__iter__'):
                for _m in m:
                    self._init_weights(_m)
            else:
                self._init_weights(m)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.bias.data.zero_()

    def _build_block(self, nIn, args, step, n_layer_all, n_layer_curr):

        layers = [MSDNFirstLayer(3, nIn, args)] \
            if n_layer_curr == 0 else []
        for i in range(step):
            n_layer_curr += 1
            inScales = args.nScales
            outScales = args.nScales
            if args.prune == 'min':
                inScales = min(args.nScales, n_layer_all - n_layer_curr + 2)
                outScales = min(args.nScales, n_layer_all - n_layer_curr + 1)
            elif args.prune == 'max':
                interval = math.ceil(1.0 * n_layer_all / args.nScales)
                inScales = args.nScales - math.floor(1.0 * (max(0, n_layer_curr - 2)) / interval)
                outScales = args.nScales - math.floor(1.0 * (n_layer_curr - 1) / interval)
                # print(i, interval, inScales, outScales, n_layer_curr, n_layer_all)
            else:
                raise ValueError
            # print('|\t\tinScales {} outScales {}\t\t\t|'.format(inScales, outScales))

            layers.append(MSDNLayer(nIn, args.growthRate, args, inScales, outScales))
            print('|\t\tinScales {} outScales {} inChannels {} outChannels {}\t\t|'.format(inScales, outScales, nIn, args.growthRate))

            nIn += args.growthRate
            if args.prune == 'max' and inScales > outScales and \
                    args.reduction > 0:
                offset = args.nScales - outScales
                layers.append(
                    self._build_transition(nIn, math.floor(1.0 * args.reduction * nIn),
                                           outScales, offset, args))
                _t = nIn
                nIn = math.floor(1.0 * args.reduction * nIn)
                print('|\t\tTransition layer inserted! (max), inChannels {}, outChannels {}\t|'.format(_t, math.floor(1.0 * args.reduction * _t)))
            elif args.prune == 'min' and args.reduction > 0 and \
                    ((n_layer_curr == math.floor(1.0 * n_layer_all / 3)) or
                     n_layer_curr == math.floor(2.0 * n_layer_all / 3)):
                offset = args.nScales - outScales
                layers.append(self._build_transition(nIn, math.floor(1.0 * args.reduction * nIn),
                                                     outScales, offset, args))

                nIn = math.floor(1.0 * args.reduction * nIn)
                print('|\t\tTransition layer inserted! (min)\t|')
            print("")
            # print('|\t\tinScales {} outScales {} inChannels {} outChannels {}\t\t\t|'.format(inScales, outScales, in_channel, nIn))

        return nn.Sequential(*layers), nIn

    def _build_transition(self, nIn, nOut, outScales, offset, args):
        net = []
        for i in range(outScales):
            net.append(ConvBasic(nIn * args.grFactor[offset + i],
                                 nOut * args.grFactor[offset + i],
                                 kernel=1, stride=1, padding=0))
        return ParallelModule(net)

    def _build_classifier_cifar(self, nIn, num_classes):
        interChannels1, interChannels2 = 128, 128
        conv = nn.Sequential(
            ConvBasic(nIn, interChannels1, kernel=3, stride=2, padding=1),
            ConvBasic(interChannels1, interChannels2, kernel=3, stride=2, padding=1),
            nn.AvgPool2d(2),
        )
        return ClassifierModule(conv, interChannels2, num_classes)

    def _build_classifier_imagenet(self, nIn, num_classes):
        conv = nn.Sequential(
            ConvBasic(nIn, nIn, kernel=3, stride=2, padding=1),
            ConvBasic(nIn, nIn, kernel=3, stride=2, padding=1),
            nn.AvgPool2d(2)
        )
        return ClassifierModule(conv, nIn, num_classes)

    def forward(self, x):
        res = []
        feat = []
        for i in range(self.nBlocks):
            # print('!!!!! The {}-th block !!!!!'.format(i))
            x = self.blocks[i](x)
            # 进行特征分割和特征重组，最后一层不能分割重组
            if i < self.nBlocks - 1: 
                x_plus, x_sub = feature_reroute_func(x[-1], chanel_axis=1, 
                            ratio=self.args.dns_ratio, use_dns=self.args.use_dns, use_fr=self.args.use_fr)
                # classifier只对最后一个feature做分类，而前面的分类list只需要继续往后传即可
                pred, t = self.classifier[i]([x_sub])
                x[-1] = x_plus
            else: 
                pred, t = self.classifier[i](x)
            res.append(pred)
            feat.append(t)
            # res.append(self.classifier[i](x))
        return res, feat
        # return res

class WrappedModel(nn.Module):
    def __init__(self, module):
        super(WrappedModel, self).__init__()
        self.module = module
        
    def forward(self, x):
        return self.module(x)

def msdnet(args):
    model = MSDNet(args)
    if args.pretrained is not None:
        print('!!!!!! Load pretrained model !!!!!!')
        model = WrappedModel(model)
        checkpoint = torch.load(args.pretrained)
        model.load_state_dict(checkpoint['state_dict'])
    return model


# msdnet config
# arch_group.add_argument('--nBlocks', type=int, default=1)
# arch_group.add_argument('--nChannels', type=int, default=32)
# arch_group.add_argument('--base', type=int,default=4)
# arch_group.add_argument('--stepmode', type=str, choices=['even', 'lin_grow'])
# arch_group.add_argument('--step', type=int, default=1)
# arch_group.add_argument('--growthRate', type=int, default=6)
# arch_group.add_argument('--grFactor', default='1-2-4', type=str)
# arch_group.add_argument('--prune', default='max', choices=['min', 'max'])
# arch_group.add_argument('--bnFactor', default='1-2-4')
# arch_group.add_argument('--bottleneck', default=True, type=bool)
# arch_group.add_argument('--pretrained', default=None, type=str, metavar='PATH',
#                        help='path to load pretrained msdnet (default: none)')
# arch_group.add_argument('--priornet', default=None, type=str, metavar='PATH',
#                        help='path to load pretrained priornet (default: none)')

from easydict import EasyDict


def msdnet_cifar100(dns_ratio=0.5,use_dns=True,use_fr=True):
    # From Mate-GF "https://github.com/SYVAE/MetaGF.git" # gamma=0.9, gamma = 0.1, nChannels=16, growthRate=16, temprature=3.0
    args = EasyDict()
    args.data = 'cifar100'
    args.nBlocks = 7
    args.base = 4
    args.nChannels = 16
    args.step = 2 
    args.stepmode = 'even' # ['even', 'lin_grow']
    args.prune = 'max' # ['min', 'max']
    args.reduction = 0.5 # [0.5, 0.75]
    args.growthRate = 16 
    args.grFactor = [1,2,4] 
    args.nScales = len(args.grFactor)
    args.bottleneck = True
    args.bnFactor = [1,2,4]
    args.dns_ratio = max(min(dns_ratio,1),0) # [0,1]
    args.use_dns = use_dns
    args.use_fr = use_fr if use_dns else False
    args.pretrained = None
    args.priornet = None
    return msdnet(args)

def msdnet_imagenet(dns_ratio=0.5,use_dns=True,use_fr=True):
    # from GE "IMTA" gamma = 0.1/0.9, nChannels=32, growthRate=16, temprature=1.0/3.0
    args = EasyDict()
    args.data = 'ImageNet'
    args.nBlocks = 5
    args.base = 4
    args.nChannels = 32
    args.step = 4 
    args.stepmode = 'even' # ['even', 'lin_grow']
    args.prune = 'max' # ['min', 'max']
    args.reduction = 0.5 # [0.5, 0.75]
    args.growthRate = 16 
    args.grFactor = [1,2,4,4] 
    args.nScales = len(args.grFactor)
    args.bottleneck = True
    args.bnFactor = [1,2,4,4]
    args.dns_ratio = max(min(dns_ratio,1),0) # [0,1]
    args.use_dns = use_dns
    args.use_fr = use_fr if use_dns else False
    args.pretrained = None
    args.priornet = None
    return msdnet(args)