import torch
import torch.nn as nn
import torch.nn.functional as F


# ---- FeatureReroute ---- #

class FeatureReroute(nn.Module):
    '''
    Split the feature into two parts with the ratio along channel dimension,
    
    '''
    def __init__(self, num_channel, ratio=0.5):
        super().__init__()
        self.ratio = max(min(ratio,1),0)
        self.num_channel = num_channel
        self.feature_num1 = round(num_channel*ratio)
        self.feature_num2 = num_channel - self.feature_num1
        
    def forward(self,x): # 默认为nchw维度，对c维度进行切分
        tmp1, tmp2 = torch.split(x,[self.feature_num1,self.feature_num2],dim=1)
        x_plus = torch.cat([tmp1,tmp2.detach()],1)
        x_sub = torch.cat([tmp1.detach(),tmp2],1)
        return x_plus,x_sub  

class FeaturePartitioning(nn.Module):
    '''
    Split the feature into two parts with the ratio along channel dimension,
    
    '''
    def __init__(self, num_channel, ratio=0.5):
        super().__init__()
        self.ratio = max(min(ratio,1),0)
        self.num_channel = num_channel
        self.feature_num1 = round(num_channel*ratio)
        self.feature_num2 = num_channel - self.feature_num1
        
    def forward(self,x): # 默认为nchw维度，对c维度进行切分
        x_plus, x_sub = torch.split(x,[self.feature_num1,self.feature_num2],dim=1)
        return x_plus,x_sub  

class FeaturePartitioningFake(nn.Module):
    '''
    Split the feature into two parts with the ratio along channel dimension,
    使用全0的tensor来填充丢失的特征，以模拟更多的feature，保证模型结构不会被更改
    '''
    def __init__(self, num_channel, ratio=0.5):
        super().__init__()
        self.ratio = max(min(ratio,1),0)
        self.num_channel = num_channel
        self.feature_num1 = round(num_channel*ratio)
        self.feature_num2 = num_channel - self.feature_num1
        self.fake_tensor1 = torch.zeros()
        self.fake_tensor2 = torch.zeros()
        
    def forward(self,x): # 默认为nchw维度，对c维度进行切分
        tmp1, tmp2 = torch.split(x,[self.feature_num1,self.feature_num2],dim=1)
        x_plus = torch.cat([tmp1,torch.zeros_like(tmp2)],1)
        x_sub = torch.cat([torch.zeros_like(tmp1),tmp2],1)
        return x_plus,x_sub  
    
class FeatureRerouteFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, ratio=0.5):
        ctx.ratio = max(min(ratio,1),0)
        ctx.num_channel = ctx.num_channel
        ctx.feature_num1 = int(ctx.num_channel*ctx.ratio)
        ctx.feature_num2 = ctx.num_channel - ctx.feature_num1
        output1 = input
        output2 = input
        return output1, output2

    # 各遮一半实现的话感觉，梯度计算量并不会降低，因为对grad_output1和grad_output2都会计算全部的梯度，可能还真不如detach版本呢
    @staticmethod
    def backward(ctx, grad_output1, grad_output2):
        input = ctx.saved_tensors
        grad_output1[:,0:ctx.feature_num1 ,:,:] = 0
        grad_output2[:,ctx.feature_num1:ctx.num_channel,:,:] = 0
        grad_input = grad_output1 + grad_output2
        grad_ratio = None
        return grad_input, grad_ratio
    
feature_reroute = FeatureRerouteFunction.apply

class FeaturePartitioningFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, ratio=0.5):
        ctx.ratio = max(min(ratio,1),0)
        ctx.num_channel = ctx.num_channel
        ctx.feature_num1 = round(ctx.num_channel*ctx.ratio)
        ctx.feature_num2 = ctx.num_channel - ctx.feature_num1
        return torch.split(x,[self.feature_num1,self.feature_num2],dim=1)

    # 各遮一半实现的话感觉，梯度计算量并不会降低，因为对grad_output1和grad_output2都会计算全部的梯度，可能还真不如detach版本呢
    @staticmethod
    def backward(ctx, grad_output1, grad_output2):
        grad_ratio = None
        return torch.cat([grad_output1,grad_output2],1), grad_ratio
    
feature_partitioning = FeaturePartitioningFunction.apply

# ---- END FeatureReroute ---- #

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

def ScalaNet(channel_in, channel_out, size):
    return nn.Sequential(
        nn.Conv2d(channel_in, 128, kernel_size=1, stride=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(128, 128, kernel_size=size, stride=size),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(128, channel_out, kernel_size=1, stride=1),
        nn.BatchNorm2d(channel_out),
        nn.ReLU(),
        nn.AvgPool2d(4, 4)
        )

class SepConv(nn.Module):

    def __init__(self, channel_in, channel_out, kernel_size=3, stride=2, padding=1, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(channel_in, channel_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=channel_in, bias=False),
            nn.Conv2d(channel_in, channel_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channel_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(channel_in, channel_in, kernel_size=kernel_size, stride=1, padding=padding, groups=channel_in, bias=False),
            nn.Conv2d(channel_in, channel_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channel_out, affine=affine),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.op(x)

def dowmsampleBottleneck(channel_in, channel_out, stride=2):
    return nn.Sequential(
        nn.Conv2d(channel_in, 128, kernel_size=1, stride=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(128, 128, kernel_size=3, stride=stride, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(128, channel_out, kernel_size=1, stride=1),
        nn.BatchNorm2d(channel_out),
        nn.ReLU(),
        )

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=100, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None,dns_ratio=0.5):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.scala1 = nn.Sequential(
            SepConv(
                channel_in=64 * block.expansion,
                channel_out=128 * block.expansion
            ),
            SepConv(
                channel_in=128 * block.expansion,
                channel_out=256 * block.expansion
            ),
            SepConv(
                channel_in=256 * block.expansion,
                channel_out=512 * block.expansion
            ),
            nn.AvgPool2d(4, 4)
        )

        self.scala2 = nn.Sequential(
            SepConv(
                channel_in=128 * block.expansion,
                channel_out=256 * block.expansion,
            ),
            SepConv(
                channel_in=256 * block.expansion,
                channel_out=512 * block.expansion,
            ),
            nn.AvgPool2d(4, 4)
        )
        self.scala3 = nn.Sequential(
            SepConv(
                channel_in=256 * block.expansion,
                channel_out=512 * block.expansion,
            ),
            nn.AvgPool2d(4, 4)
        )
        self.scala4 = nn.AvgPool2d(4, 4)

        self.attention1 = nn.Sequential(
            SepConv(
                channel_in=64 * block.expansion,
                channel_out=64 * block.expansion
            ),
            nn.BatchNorm2d(64 * block.expansion),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Sigmoid()
        )

        self.attention2 = nn.Sequential(
            SepConv(
                channel_in=128 * block.expansion,
                channel_out=128 * block.expansion
            ),
            nn.BatchNorm2d(128 * block.expansion),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Sigmoid()
        )

        self.attention3 = nn.Sequential(
            SepConv(
                channel_in=256 * block.expansion,
                channel_out=256 * block.expansion
            ),
            nn.BatchNorm2d(256 * block.expansion),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Sigmoid()
        )

        self.fc1 = nn.Linear(512 * block.expansion, num_classes)
        self.fc2 = nn.Linear(512 * block.expansion, num_classes)
        self.fc3 = nn.Linear(512 * block.expansion, num_classes)
        self.fc4 = nn.Linear(512 * block.expansion, num_classes)

        self.adaptation_layer1 = nn.Linear(512 * block.expansion,512 * block.expansion)
        self.adaptation_layer2 = nn.Linear(512 * block.expansion,512 * block.expansion)
        self.adaptation_layer3 = nn.Linear(512 * block.expansion,512 * block.expansion)

        self.dns1 = FeatureReroute(64,ratio=dns_ratio)
        self.dns2 = FeatureReroute(128,ratio=dns_ratio)
        self.dns3 = FeatureReroute(256,ratio=dns_ratio)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        feature_list = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x_plus, x_sub = self.dns1(x)

        fea1 = self.attention1(x_sub)
        fea1 = fea1 * x_sub
        feature_list.append(fea1)

        x = self.layer2(x_plus)
        x_plus, x_sub = self.dns2(x)

        fea2 = self.attention2(x_sub)
        fea2 = fea2 * x_sub
        feature_list.append(fea2)

        x = self.layer3(x_plus)
        x_plus, x_sub = self.dns3(x)

        fea3 = self.attention3(x_sub)
        fea3 = fea3 * x_sub
        feature_list.append(fea3)

        x = self.layer4(x_plus)
        feature_list.append(x)

        out1_feature = self.scala1(feature_list[0]).view(x.size(0), -1)
        out1 = self.fc1(out1_feature)
        out1_feature = self.adaptation_layer1(out1_feature)

        out2_feature = self.scala2(feature_list[1]).view(x.size(0), -1)
        out2 = self.fc2(out2_feature)
        out2_feature = self.adaptation_layer2(out2_feature)

        out3_feature = self.scala3(feature_list[2]).view(x.size(0), -1)
        out3 = self.fc3(out3_feature)
        out3_feature = self.adaptation_layer3(out3_feature)
        
        out4_feature = self.scala4(feature_list[3]).view(x.size(0), -1)
        out4 = self.fc4(out4_feature)

        return [out4, out3, out2, out1], [out4_feature, out3_feature, out2_feature, out1_feature]

class ResNetWithoutFR(nn.Module):

    def __init__(self, block, layers, num_classes=100, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None,dns_ratio=0.5):
        super(ResNetWithoutFR, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.dns_ratio = dns_ratio

        self.inplanes = 64 # int(64*dns_ratio)
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0],partitionied_intput=False)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0],partitionied_intput=True)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1],partitionied_intput=True)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2],partitionied_intput=True)
        self.scala1 = nn.Sequential(
            SepConv(
                channel_in= round(64 * block.expansion*(1-self.dns_ratio)), # 64 * block.expansion,
                channel_out=128 * block.expansion
            ),
            SepConv(
                channel_in=128 * block.expansion,
                channel_out=256 * block.expansion
            ),
            SepConv(
                channel_in=256 * block.expansion,
                channel_out=512 * block.expansion
            ),
            nn.AvgPool2d(4, 4)
        )

        self.scala2 = nn.Sequential(
            SepConv(
                channel_in=round(128 * block.expansion*(1-self.dns_ratio)),# 128 * block.expansion,
                channel_out=256 * block.expansion,
            ),
            SepConv(
                channel_in=256 * block.expansion,
                channel_out=512 * block.expansion,
            ),
            nn.AvgPool2d(4, 4)
        )
        self.scala3 = nn.Sequential(
            SepConv(
                channel_in=round(256 * block.expansion*(1-self.dns_ratio)),# 256 * block.expansion,
                channel_out=512 * block.expansion,
            ),
            nn.AvgPool2d(4, 4)
        )
        self.scala4 = nn.AvgPool2d(4, 4)

        self.attention1 = nn.Sequential(
            SepConv(
                channel_in=round(64 * block.expansion*(1-self.dns_ratio)), # 64 * block.expansion,
                channel_out=round(64 * block.expansion*(1-self.dns_ratio)), # 64 * block.expansion,
            ),
            nn.BatchNorm2d(round(64 * block.expansion*(1-self.dns_ratio))),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Sigmoid()
        )

        self.attention2 = nn.Sequential(
            SepConv(
                channel_in=round(128 * block.expansion*(1-self.dns_ratio)),
                channel_out=round(128 * block.expansion*(1-self.dns_ratio))
            ),
            nn.BatchNorm2d(round(128 * block.expansion*(1-self.dns_ratio))),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Sigmoid()
        )

        self.attention3 = nn.Sequential(
            SepConv(
                channel_in=round(256 * block.expansion*(1-self.dns_ratio)),
                channel_out=round(256 * block.expansion*(1-self.dns_ratio))
            ),
            nn.BatchNorm2d(round(256 * block.expansion*(1-self.dns_ratio))),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Sigmoid()
        )

        self.fc1 = nn.Linear(512 * block.expansion, num_classes)
        self.fc2 = nn.Linear(512 * block.expansion, num_classes)
        self.fc3 = nn.Linear(512 * block.expansion, num_classes)
        self.fc4 = nn.Linear(512 * block.expansion, num_classes)
        
        self.adaptation_layer1 = nn.Linear(512 * block.expansion,512 * block.expansion)
        self.adaptation_layer2 = nn.Linear(512 * block.expansion,512 * block.expansion)
        self.adaptation_layer3 = nn.Linear(512 * block.expansion,512 * block.expansion)

        self.fp1 = FeaturePartitioning(64,ratio=self.dns_ratio)
        self.fp2 = FeaturePartitioning(128,ratio=self.dns_ratio)
        self.fp3 = FeaturePartitioning(256,ratio=self.dns_ratio)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False,partitionied_intput=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if partitionied_intput:
            self.inplanes=round(self.inplanes*self.dns_ratio) # 如果要切分，那么输入通道就会减半，应该直接切掉的
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        # 只有这里是变形的，所以只需要对这里，还有上面的downsample变形就OK，其他地方不用变形
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        feature_list = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x_plus, x_sub = self.fp1(x)

        fea1 = self.attention1(x_sub)
        fea1 = fea1 * x_sub
        feature_list.append(fea1)

        x = self.layer2(x_plus)
        x_plus, x_sub = self.fp2(x)

        fea2 = self.attention2(x_sub)
        fea2 = fea2 * x_sub
        feature_list.append(fea2)

        x = self.layer3(x_plus)
        x_plus, x_sub = self.fp3(x)

        fea3 = self.attention3(x_sub)
        fea3 = fea3 * x_sub
        feature_list.append(fea3)

        x = self.layer4(x_plus)
        feature_list.append(x)

        out1_feature = self.scala1(feature_list[0]).view(x.size(0), -1)
        out1 = self.fc1(out1_feature)
        out1_feature = self.adaptation_layer1(out1_feature)

        out2_feature = self.scala2(feature_list[1]).view(x.size(0), -1)
        out2 = self.fc2(out2_feature)
        out2_feature = self.adaptation_layer2(out2_feature)

        out3_feature = self.scala3(feature_list[2]).view(x.size(0), -1)
        out3 = self.fc3(out3_feature)
        out3_feature = self.adaptation_layer3(out3_feature)

        out4_feature = self.scala4(feature_list[3]).view(x.size(0), -1)

        out4 = self.fc4(out4_feature)

        return [out4, out3, out2, out1], [out4_feature, out3_feature, out2_feature, out1_feature]

def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model

def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    # We use the stable dns_ratio=0.5 for resnet18
    kwargs.pop("dns_ratio")
    kwargs.pop("use_fr")
    kwargs.pop("use_dns")
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


def wide_resnet50_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def wide_resnet101_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


def _resnet_without_fr(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNetWithoutFR(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model

def resnet18_without_fr(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet_without_fr('resnet18_without_fr', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)