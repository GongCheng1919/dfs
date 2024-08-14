import torch
from torch import nn



class FeatureReroute(nn.Module):
    '''
    Split the feature into two parts with the ratio along channel dimension,
    '''
    def __init__(self, num_channel, chanel_axis = -1, ratio=0.5, use_dns=True, use_fr=True):
        super().__init__()
        self.ratio = max(min(ratio,1),0)
        self.use_fr = use_fr
        self.use_dns = use_dns
        self.num_channel = num_channel
        self.chanel_axis = chanel_axis
        self.feature_num1 = round(num_channel*ratio)
        self.feature_num2 = num_channel - self.feature_num1

    def forward(self,x): # 默认为nchw维度，对c维度进行切分
        if not self.use_dns:
            # print("Do not use DNS")
            return x, x
        else:
            if self.ratio >= 1.:
                # print("DNS is 1")
                if self.use_fr:
                    return x,x.detach()
                else:
                    return x,torch.zeros_like(x,requires_grad=False)
            elif self.ratio <= 0.:
                # print("DNS is 0")
                if self.use_fr:
                    return x.detach(),x
                else:
                    return torch.zeros_like(x,requires_grad=False),x
            # 其他情况下的结果
            # 特征分割
            tmp1, tmp2 = torch.split(x,[self.feature_num1,self.feature_num2],dim=self.chanel_axis)
            # 特征重组
            if self.use_fr:
                # print("use DNS and use fr") 
                x_plus = torch.cat([tmp1,tmp2.detach()],self.chanel_axis)
                # 对于ci的浅层分类器而言，其可以复用当前层的特征，
                # 只是略微增加了反向传播计算量，但是可以极大地增加参数量和模型容量，对于小模型来说非常有效
                # x_sub = torch.cat([tmp1.detach(),tmp2],self.chanel_axis)
                x_sub = x 
            else:
                # print("use DNS but do not use fr")
                x_plus = torch.cat([tmp1,torch.zeros_like(tmp2,requires_grad=False)],self.chanel_axis)
                x_sub = torch.cat([torch.zeros_like(tmp1,requires_grad=False),tmp2],self.chanel_axis)
            # print(f"Using dns ratio is {self.ratio}. The shared featrue number is {self.feature_num1} and unshared one is {self.feature_num2}")
            return x_plus,x_sub

def feature_reroute_func(x, chanel_axis = -1, ratio=0.5, use_dns=True, use_fr=True):
    return FeatureReroute(x.size(chanel_axis), chanel_axis, ratio, use_dns, use_fr)(x)

class dns_embbeding(object):
    def __init__(self,chanel_axis,dns_ratio, use_dns=True, use_fr=True):
        self.chanel_axis = chanel_axis
        self.dns_ratio = dns_ratio
        self.use_dns = use_dns
        self.use_fr = use_fr
        self.x_sub = None
        
    def dns_embbeding_hook(self):
        # print(module,len(input),input[0].shape)
        # y = torch.zeros(3,16*5*5)
        def dns_embbeding_(module,input,result):
            x_plus,self.x_sub = feature_reroute_func(result,self.chanel_axis,self.dns_ratio,self.use_dns,self.use_fr)
            return x_plus
            # return torch.ones_like(result)
        return dns_embbeding_