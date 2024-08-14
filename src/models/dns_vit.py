# vit_tiny_patch16_224.blocks[0]
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from torch import nn
from .feature_reroute import dns_embbeding
class DNSWapparedViT(nn.Module):
    def __init__(self,
                 backbone_vit_model,
                 dns_ratio=0.5,
                 use_dns=True,
                 use_fr=True,
                 start_layer=6 # 从第几个layer再开始分割吧，不然好像对精度影响有点大。
                ):
        super().__init__()
        self.dns_ratio = dns_ratio
        self.use_dns = use_dns
        self.use_fr = use_fr
        self.start_layer = start_layer
        if not use_dns:
            print("Do not use DNS for MTL")
        elif not use_fr:
            print(f"Use DNS (dns_ratio={dns_ratio}) but not FR for MTL")
        else:
            print(f"Use DNS (dns_ratio={dns_ratio}) and FR for MTL")
        
        self.backbone_model = backbone_vit_model
        self.num_classes = self.backbone_model.num_classes
        # 模型的block数量，通常等于exit的数量，但是有时候浅层的效果太差了，就需要丢弃浅层，这个参数由start_layer决定
        self.num_blocks = len(self.backbone_model.blocks)
        # nBlocks指的是exit的数量
        self.nBlocks = self.num_blocks - self.start_layer
        self.blocks = self.backbone_model.blocks
        self.global_pool = self.backbone_model.global_pool
        # 这里应该是nBlocks-1呀，最后一层不能分割，不然一半特征就没了，你搞忘了啊！！！难怪!!!
        self.dns_x_subs_objs = [dns_embbeding(-1,self.dns_ratio,self.use_dns,self.use_fr) for _ in range(self.nBlocks-1)]
        self.dns_x_subs_objs.append( # 最后一层的特征，不能使用dns进行分割，只需要单独拿出来，用于做FD监督用，即作为teacher feature。
            dns_embbeding(-1,1,False,False)
        )
        # self.backbone_model.dns_x_subs_hooks = []
        # 清空之前所有的用于DNS的hooks
        if hasattr(self.backbone_model,"dns_x_subs_hooks"):
            if len(self.backbone_model.dns_x_subs_hooks)>0:
                for handle in self.backbone_model.dns_x_subs_hooks:
                    handle.remove()
        # 重新构建hook函数
        self.backbone_model.dns_x_subs_hooks = []
        for i in range(self.nBlocks): 
            # if len(self.blocks[i]._forward_hooks)>0 \
            #     and list(self.blocks[i]._forward_hooks.values())[0].__name__=="dns_embbeding_":
            #     raise ValueError("hook function has been registered, please remove the old hook function and then regiester new one!")
            handle = self.blocks[self.start_layer+i].register_forward_hook(self.dns_x_subs_objs[i].dns_embbeding_hook()) # 注册hook function
            self.backbone_model.dns_x_subs_hooks.append(handle)
        # create classifier # n - 1个输出
        self.middle_norms = nn.ModuleList([nn.LayerNorm(self.backbone_model.embed_dim,eps=1e-6) for _ in range(self.nBlocks-1)])
        self.middle_heads = nn.ModuleList([nn.Linear(self.backbone_model.embed_dim, self.num_classes) for _ in range(self.nBlocks-1)])
        self.fc_norm = nn.Identity()
        self.head_drop = self.backbone_model.head_drop
        # self.classifer_modules=[
        #     for _ in range(self.nBlocks)
        # ]
    def middle_task(self,x,task_id):
        x = self.middle_norms[task_id](x)
        if self.global_pool:
            x = x[:, self.backbone_model.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return self.middle_heads[task_id](x)
    
    def forward(self,input):
        final_logits = self.backbone_model(input)
        # 这里拿出来的是nBlocks层的中间特征，其中最后一层没有做DNS分割，因为没有更深层需要传递了
        middle_features = [obj.x_sub for obj in self.dns_x_subs_objs]
        mtl_logits = []
        for i in range(self.nBlocks-1):
            middle_logits = self.middle_task(middle_features[i],i)
            mtl_logits.append(middle_logits)
        mtl_logits.append(final_logits)
        return mtl_logits,middle_features
    
# Demo:       
# dns_vit = DNSWapparedViT(vit_tiny_patch16_224,0.3,True,False)
# inputs = torch.ones(1,3,224,224)
# res = dns_vit(inputs)