import torch
from torch import nn
import torch.nn.functional as F

# 就用VGG7 on cifar100来测试一下DNS到底能不能加速

def vgg_block(num_convs, in_channels, out_channels):
    net = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
           nn.ReLU(True),
           nn.BatchNorm2d(out_channels)]
    for i in range(num_convs - 1): # 定义后面的许多层
        net.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        net.append(nn.ReLU(True))
        net.append(nn.BatchNorm2d(out_channels))
    net.append(nn.MaxPool2d(2, 2)) # 定义池化层
    return nn.Sequential(*net)
 
# 下面我们定义一个函数对这个 vgg block 进行堆叠
def vgg_stack(num_convs, channels): # vgg_net = vgg_stack((1, 1, 2, 2, 2), ((3, 64), (64, 128), (128, 256), (256, 512), (512, 512)))
    net = []
    for n, c in zip(num_convs, channels):
        in_c = c[0]
        out_c = c[1]
        net.append(vgg_block(n, in_c, out_c))
    return nn.Sequential(*net)
 
#确定vgg的类型，是vgg11 还是vgg16还是vgg19
# vgg_net = vgg_stack((1, 1, 2, 2, 2), ((3, 64), (64, 128), (128, 256), (256, 512), (512, 512)))
#vgg类
# VGG7_64((2,2,2),((3,64),(64,128),(128,256)))
class VGG(nn.Module):
    def __init__(self,num_convs,channels,out_dim=16):
        super().__init__()
        self.feature = vgg_stack(num_convs,channels)
        self.fc = nn.Sequential(
            nn.Linear(out_dim*channels[-1][1], 1024),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1024, 10))
        
    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x
    
class VGG_plane(nn.Module):
    def __init__(self,num_convs,channels):
        super().__init__()
        self.feature = vgg_stack(num_convs,channels)
        self.avg_pool = nn.AvgPool2d(4)
        self.linear = nn.Linear(channels[-1][1], 10)
        
    def forward(self, x):
        x = self.feature(x)
        x = self.avg_pool(x)
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        return x

class MTL_VGG7_64_plane_base(nn.Module):
    def __init__(self,num_classes=10,dns_ratio=0.5):
        super().__init__()
        self.dns_ratio = dns_ratio
        # block 1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.act1 = nn.ReLU(True)
        self.bn1 = nn.BatchNorm2d(64)
        self.classifer1 = nn.Sequential(
                    nn.AvgPool2d(32),
                    nn.Flatten(),
                    nn.Linear(64, num_classes)
                )

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.act2 = nn.ReLU(True)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.classifer2 = nn.Sequential(
                    nn.AvgPool2d(16),
                    nn.Flatten(),
                    nn.Linear(64, num_classes)
                )

        # block 2
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.act3 = nn.ReLU(True)
        self.bn3 = nn.BatchNorm2d(128)
        self.classifer3 = nn.Sequential(
                    nn.AvgPool2d(16),
                    nn.Flatten(),
                    nn.Linear(128, num_classes)
                )
        
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.act4 = nn.ReLU(True)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.classifer4 = nn.Sequential(
                    nn.AvgPool2d(8),
                    nn.Flatten(),
                    nn.Linear(128, num_classes)
                )
        

        # block 3
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.act5 = nn.ReLU(True)
        self.bn5 = nn.BatchNorm2d(256)
        self.classifer5 = nn.Sequential(
                    nn.AvgPool2d(8),
                    nn.Flatten(),
                    nn.Linear(256, num_classes)
                )
        
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.act6 = nn.ReLU(True)
        self.bn6 = nn.BatchNorm2d(256)
        self.pool6 = nn.MaxPool2d(2, 2)
        # classifier
        self.classifer6 = nn.Sequential(
            nn.AvgPool2d(4),
            nn.Flatten(),
            # nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        f1 = self.bn1(self.act1(self.conv1(x)))
        c1 = self.classifer1(f1)
        f2 = self.pool2(self.bn2(self.act2(self.conv2(f1))))
        c2 = self.classifer2(f2)
        f3 = self.bn3(self.act3(self.conv3(f2)))
        c3 = self.classifer3(f3)
        f4 = self.pool4(self.bn4(self.act4(self.conv4(f3))))
        c4 = self.classifer4(f4)
        f5 = self.bn5(self.act5(self.conv5(f4)))
        c5 = self.classifer5(f5)
        f6 = self.pool6(self.bn6(self.act6(self.conv6(f5))))
        c6 = self.classifer6(f6)
        return [c1,c2,c3,c4,c5,c6],[f1,f2,f3,f4,f5,f6]

class STL_VGG7_64_plane(MTL_VGG7_64_plane_base):
    def forward(self, x):
        logits,features = super().forward(x)
        return logits[-1]

def vgg64_7_plane_mtl(dns_ratio=0.5):
    return MTL_VGG7_64_plane_base(num_classes=100,dns_ratio=dns_ratio)

    
def vgg7_M(M):
    return VGG((2,2,2),((3,M),(M,2*M),(2*M,4*M)))

def vgg7_M_plane(M):
    return VGG_plane((2,2,2),((3,M),(M,2*M),(2*M,4*M)))
    
def vgg7_64():
    return vgg7_M(64)
    
def vgg7_64_plane():
    return vgg7_M_plane(64)


class feature_split(nn.Module):
    def __init__(self,
                  plus_fea_num,
                  sub_fea_num,
                  dns_ratio=0.5,
                  chanel_axis=1):
        super().__init__()
        self.plus_fea_num=plus_fea_num
        self.sub_fea_num=sub_fea_num
        self.chanel_axis=chanel_axis
        self.dns_ratio=dns_ratio

    def __call__(self,f):
        # if not self.use_dns:
        #     return f,f
        # if self.dns_ratio==0:
        #     return None,f
        # elif self.dns_ratio == 1:
        #     return f,None
        # else:
        #     f_p, f_s = torch.split(f,[self.plus_fea_num,self.sub_fea_num],dim=self.chanel_axis)
        #     return f_p, f_s
        f_p, f_s = torch.split(f,[self.plus_fea_num,self.sub_fea_num],dim=self.chanel_axis)
        return f_p, f_s

class split_conv_relu_bn_impl(nn.Module):
    def __init__(self,plus_fea_num,sub_fea_num,output_fea_num,
                 pooling = False,
                 use_dns=True,
                 use_fr=True):
        super().__init__()
        self.plus_fea_num=plus_fea_num
        self.sub_fea_num=sub_fea_num
        self.output_fea_num=output_fea_num
        self.use_dns=use_dns
        self.use_fr=use_fr
        # if use_dns:
        #     self.conv_p = nn.Conv2d(plus_fea_num, output_fea_num, kernel_size=3, bias=None, padding=1)
        # else:
        #     self.conv_p = nn.Conv2d(plus_fea_num+sub_fea_num, output_fea_num, kernel_size=3, bias=None, padding=1)
        # if use_fr:
        #     self.conv_s = nn.Conv2d(sub_fea_num, output_fea_num, kernel_size=3, bias=None, padding=1)
        self.conv_p = nn.Conv2d(plus_fea_num, output_fea_num, kernel_size=3, bias=None, padding=1)
        self.conv_s = nn.Conv2d(sub_fea_num, output_fea_num, kernel_size=3, bias=None, padding=1)
        self.act = nn.ReLU(True)
        self.bn = nn.BatchNorm2d(output_fea_num)
        if pooling:
            self.pool = nn.MaxPool2d(2, 2)
        else:
            self.pool = nn.Identity()

    def forward(self, f_p,f_s):
        # 这里应该要有一个判断
        # 如果不用fr，就不加f_s的结果
        # 如果用fr，就加上f_s的结果，并且在f_s这里阻止反向传播
        # if self.use_fr:
        #     output = self.conv_p(f_p)+self.conv_s(f_s.detach())
        # else:
        #     output = self.conv_p(f_p)
        if self.use_dns:
            if self.use_fr:
                output = self.conv_p(f_p)+self.conv_s(f_s.detach())
            else:
                output = self.conv_p(f_p)
        else:
            output = self.conv_p(f_p)+self.conv_s(f_s)
        return self.pool(self.bn(self.act(output)))
    
class split_classifer_impl(nn.Module):
    def __init__(self,plus_fea_num,sub_fea_num,
                 output_size,
                 num_classes,
                 use_dns=True,
                 use_fr=True):
        super().__init__()
        self.plus_fea_num=plus_fea_num
        self.sub_fea_num=sub_fea_num
        self.output_size=output_size
        self.num_classes=num_classes
        self.use_dns=use_dns
        self.use_fr=use_fr
        self.avg_pool = nn.AvgPool2d(output_size)
        self.flatten = nn.Flatten()
        # if use_dns:
        #     self.fc_s = nn.Linear(sub_fea_num, num_classes,bias=None)
        # else:
        #     self.fc_s = nn.Linear(plus_fea_num+sub_fea_num, num_classes,bias=None)
        # if use_fr:
        #     self.fc_p = nn.Linear(plus_fea_num, num_classes,bias=None)
        self.fc_s = nn.Linear(sub_fea_num, num_classes,bias=None)
        self.fc_p = nn.Linear(plus_fea_num, num_classes,bias=None)

    def forward(self, f_p, f_s):
        f_s = self.flatten(self.avg_pool((f_s)))
        f_p = self.flatten(self.avg_pool(f_p))
        # if self.use_fr:
        #     f_p = self.flatten(self.avg_pool(f_p))
        #     output = self.fc_s(f_s)+ self.fc_p(f_p.detach())
        # else:
        #     output = self.fc_s(f_s)
        if self.use_dns:
            if self.use_fr:
                output = self.fc_s(f_s)+ self.fc_p(f_p.detach())
            else:
                output = self.fc_s(f_s)
        else:
            output = self.fc_s(f_s)+ self.fc_p(f_p)
        return output

class MTL_VGG7_64_plane_dns(nn.Module):
    def __init__(self,
                 channel_num=64,
                 num_classes=10,
                 use_dns=True,
                 use_fr=True,
                 dns_ratio=0.5):
        super().__init__()
        self.dns_ratio = max(min(dns_ratio,1),0)
        self.plus_fea_num = round(channel_num * dns_ratio)
        self.sub_fea_num = channel_num - self.plus_fea_num
        self.chanel_axis = 1
        self.use_dns = use_dns
        self.use_fr = use_fr if self.use_dns else False

        # block 1
        # feature_extractor 1 
        self.conv1 = nn.Conv2d(3, channel_num, kernel_size=3, padding=1)
        self.act1 = nn.ReLU(True)
        self.bn1 = nn.BatchNorm2d(channel_num)
        self.feature_splitor1 = feature_split(self.plus_fea_num,self.sub_fea_num,self.dns_ratio,self.chanel_axis)
        self.classifer1 = split_classifer_impl(self.plus_fea_num,self.sub_fea_num,32,
                                               num_classes, self.use_dns,self.use_fr)
        # feature_extractor 1 
        self.feature_extractor2 = split_conv_relu_bn_impl(self.plus_fea_num,self.sub_fea_num,channel_num, True, self.use_dns,self.use_fr)
        self.feature_splitor2 = feature_split(self.plus_fea_num,self.sub_fea_num,self.dns_ratio,self.chanel_axis)
        self.classifer2 = split_classifer_impl(self.plus_fea_num,self.sub_fea_num,16,
                                               num_classes, self.use_dns,self.use_fr)

        # block 2
        self.feature_extractor3 = split_conv_relu_bn_impl(self.plus_fea_num,self.sub_fea_num,2*channel_num, False, self.use_dns,self.use_fr)
        self.feature_splitor3 = feature_split(2*self.plus_fea_num,2*self.sub_fea_num,self.dns_ratio,self.chanel_axis)
        self.classifer3 = split_classifer_impl(2*self.plus_fea_num,2*self.sub_fea_num,16,
                                               num_classes, self.use_dns,self.use_fr)
        self.feature_extractor4 = split_conv_relu_bn_impl(2*self.plus_fea_num,2*self.sub_fea_num,2*channel_num, True, self.use_dns,self.use_fr)
        self.feature_splitor4 = feature_split(2*self.plus_fea_num,2*self.sub_fea_num,self.dns_ratio,self.chanel_axis)
        self.classifer4 = split_classifer_impl(2*self.plus_fea_num,2*self.sub_fea_num,8,
                                               num_classes, self.use_dns,self.use_fr)
        

        # block 3
        self.feature_extractor5 = split_conv_relu_bn_impl(2*self.plus_fea_num,2*self.sub_fea_num,4*channel_num,False,self.use_dns,self.use_fr)
        self.feature_splitor5 = feature_split(4*self.plus_fea_num,4*self.sub_fea_num,self.dns_ratio,self.chanel_axis)
        self.classifer5 = split_classifer_impl(4*self.plus_fea_num,4*self.sub_fea_num,8,
                                               num_classes, self.use_dns,self.use_fr)
        self.feature_extractor6 = split_conv_relu_bn_impl(4*self.plus_fea_num,4*self.sub_fea_num,4*channel_num,True,self.use_dns,self.use_fr)
        # self.feature_splitor6 = feature_split(4*self.plus_fea_num,4*self.sub_fea_num,self.chanel_axis,self.use_dns)
        # self.classifer6 = split_classifer_impl(4*self.plus_fea_num,4*self.sub_fea_num,4,
        #                                        num_classes,True, self.use_dns,self.use_fr)
        # # classifier
        self.classifer6 = nn.Sequential(
            nn.AvgPool2d(4),
            nn.Flatten(),
            # nn.Dropout(0.5),
            nn.Linear(4*channel_num, num_classes,bias=False)
        )

        
    def forward(self, x):
        f1 = self.bn1(self.act1(self.conv1(x)))
        f1_p, f1_s = self.feature_splitor1(f1)

        c1 = self.classifer1(f1_p, f1_s)
        f2 = self.feature_extractor2(f1_p,f1_s)
        f2_p, f2_s = self.feature_splitor2(f2)

        c2 = self.classifer2(f2_p, f2_s)
        f3 = self.feature_extractor3(f2_p,f2_s)
        f3_p, f3_s = self.feature_splitor3(f3)

        c3 = self.classifer3(f3_p, f3_s)
        f4 = self.feature_extractor4(f3_p,f3_s)
        f4_p, f4_s = self.feature_splitor4(f4)

        c4 = self.classifer4(f4_p, f4_s)
        f5 = self.feature_extractor5(f4_p,f4_s)
        f5_p, f5_s = self.feature_splitor5(f5)

        c5 = self.classifer5(f5_p, f5_s)
        f6 = self.feature_extractor6(f5_p,f5_s)
        c6 = self.classifer6(f6)

        return [c1,c2,c3,c4,c5,c6],[f1,f2,f3,f4,f5,f6]

def vgg_dsn(dns_ratio=0.5):
    return MTL_VGG7_64_plane_dns(channel_num=64,
                 num_classes=100,
                 use_dns=False,
                 use_fr=False,
                 dns_ratio=dns_ratio)
def vgg_dns_wo_fr(dns_ratio=0.5):
    return MTL_VGG7_64_plane_dns(channel_num=64,
                 num_classes=100,
                 use_dns=True,
                 use_fr=False,
                 dns_ratio=dns_ratio)
def vgg_dns_w_fr(dns_ratio=0.5):
    return MTL_VGG7_64_plane_dns(channel_num=64,
                 num_classes=100,
                 use_dns=True,
                 use_fr=True,
                 dns_ratio=dns_ratio)

class MTL_VGG_plane_dns(nn.Module):
    def __init__(self,
                 block_list, # block_list应该是一个列表，每一个元素为tuple，表示block内的卷积数量，输入输出通道数量，比如VGG参数是[(2,2,2),((3,64),(64,128),(128,256))]
                 input_size = 32, # for cifar10/100
                 num_classes=10,
                 use_dns=True,
                 use_fr=True,
                 dns_ratio=0.5,
                 pooling_feature_distill=False,
                 distill_feature=False,
                 verbose=False):
        super().__init__()
        self.dns_ratio = max(min(dns_ratio,1),0)
        self.chanel_axis = 1
        self.use_dns = use_dns
        self.use_fr = use_fr if self.use_dns else False
        self.num_convs, self.channels = block_list
        self.nBlocks = sum(self.num_convs)
        self.input_size = input_size
        self.feature_extractor_list = []
        self.feature_splitor_list = []
        self.classifer_list = []
        self.adaptation_list = [] # used to resize the feature size and adaptive the teature feature 
        self.distill_feature=distill_feature
        output_sizes = []
        output_channels = []

        for i, (n, c) in enumerate(zip(self.num_convs, self.channels)):
            if verbose:
                print(f"Construct pooling block: {i} {n} convs, {c[0]}->{c[1]}")
            for b in range(n):
                if verbose:
                    print(f"\t build conv: {b}, {c[0] if b==0 else c[1]}->{c[1]}")
                pooling = b==(n-1) # 最后一层才添加池化层
                channel_num = c[0] if b==0 else c[1] # 第一个卷积核升维
                plus_fea_num = round(channel_num * dns_ratio)
                sub_fea_num = channel_num - plus_fea_num
                if b==0 and c[0]==3: ## 第一个卷积层不能被分割
                    self.feature_extractor_list.append(
                        nn.Sequential(
                            nn.Conv2d(c[0], c[1], kernel_size=3, padding=1),
                            nn.ReLU(True),
                            nn.BatchNorm2d(c[1])
                        )
                    )
                else:
                    self.feature_extractor_list.append(
                            split_conv_relu_bn_impl(plus_fea_num,sub_fea_num,c[1], pooling, self.use_dns,self.use_fr)
                            )
                if pooling:
                    input_size = input_size//2
                output_sizes.append(input_size)
                output_channels.append(c[1])
                # 分割和提取都是基于输出channel_num，也就是c[1]
                plus_fea_num = round(c[1] * dns_ratio)
                sub_fea_num = c[1] - plus_fea_num
                self.feature_splitor_list.append(feature_split(plus_fea_num,sub_fea_num,self.dns_ratio,self.chanel_axis))
                self.classifer_list.append(split_classifer_impl(plus_fea_num,sub_fea_num,output_sizes[-1],
                                               num_classes, self.use_dns,self.use_fr))
        self.feature_extractors = nn.ModuleList(self.feature_extractor_list)
        self.feature_splitors = nn.ModuleList(self.feature_splitor_list)
        # 最后一层的classifer不用分割：
        self.classifer_list[-1].use_dns = False
        self.classifers = nn.ModuleList(self.classifer_list)
        # adaption 
        if self.distill_feature:
            for i in range(self.nBlocks-1):
                if not pooling_feature_distill: # 输出尺寸太大了，只能使用小的特征进行蒸馏，或者不适用linear层蒸馏
                    align_size = output_sizes[i]//output_sizes[-1]
                    flatten_size = output_sizes[-1]*output_sizes[-1]*output_channels[i]
                    target_size = output_sizes[-1]*output_sizes[-1]*output_channels[-1]
                else:
                    align_size = output_sizes[i]
                    flatten_size = output_channels[i]
                    target_size = output_channels[-1]
                if verbose:
                    print(f"Construct {i} adaptation layer: {flatten_size}->{target_size} {(flatten_size*target_size)/2**20}MB")
                if align_size>1:
                    self.adaptation_list.append(
                        nn.Sequential(
                            nn.AvgPool2d(align_size), # 将特征池化对齐
                            nn.Flatten(),
                            nn.Linear(flatten_size, target_size,bias=None) # 转换一次特征，用于知识蒸馏
                        )
                    )
                else:
                    self.adaptation_list.append(
                        nn.Sequential(
                            nn.Flatten(),
                            nn.Linear(flatten_size, target_size,bias=None) # 转换一次特征，用于知识蒸馏
                        )
                    )
            if not pooling_feature_distill:
                self.adaptation_list.append(nn.Flatten()) # 最后一层只做拍平，不拍平也行，反正都一样
            else:
                self.adaptation_list.append(
                    nn.Sequential(
                        nn.AvgPool2d(output_sizes[-1]),
                        nn.Flatten()
                    )
                )
            self.adaptation_layers = nn.ModuleList(self.adaptation_list)

    def forward(self,x):
        x_p,x_s = None,None
        cs = []
        fs = []
        for b in range(self.nBlocks):
            if b == 0:
                x = self.feature_extractors[0](x)
            else:
                x = self.feature_extractors[b](x_p,x_s)
            x_p, x_s = self.feature_splitors[b](x)
            cs.append(self.classifers[b](x_p,x_s))
            if self.distill_feature:
                fs.append(self.adaptation_layers[b](x))
            else:
                fs.append(x)
        return cs,fs

def vgg7_64_cifar100(use_dns=False,use_fr=False,dns_ratio=0.5,distill_feature=True):
    return MTL_VGG_plane_dns(block_list=[(2,2,2),((3,64),(64,128),(128,256))],
                                     input_size=32,num_classes=100,pooling_feature_distill=False,distill_feature=distill_feature,
                                     use_dns=use_dns,use_fr=use_fr,dns_ratio=dns_ratio) 
def vgg16_64_imagenet(use_dns=False,use_fr=False,dns_ratio=0.5,distill_feature=True):
    return MTL_VGG_plane_dns(block_list=[(2,2,3,3,3),((3,64),(64,128),(128,256),(256,512),(512,512))],pooling_feature_distill=True,distill_feature=distill_feature,
                                     input_size=224,num_classes=1000,use_dns=use_dns,use_fr=use_fr,dns_ratio=dns_ratio)
