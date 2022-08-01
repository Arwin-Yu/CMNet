import torch.nn.functional as F
import torch
import torch.nn as nn  

class Basiconv(nn.Module):
    def __init__(self, in_nc, out_nc, kernel_size, stride, padding, dilation=1):
        super().__init__()
        self.featurs = nn.Sequential(
            nn.Conv2d(in_nc, out_nc, kernel_size, stride, padding, dilation=dilation),
            nn.BatchNorm2d(out_nc),
            nn.GELU()
        )
    def forward(self,x):
        return self.featurs(x)
  
class sub_1st(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        # receptive_filed= 7.0 
        self.branch1 = nn.Sequential(
            Basiconv(in_nc=3, out_nc=dim//4, kernel_size=3, stride=2, padding=1), 
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # receptive_filed=15.0
        self.branch2 = nn.Sequential(
            Basiconv(in_nc=3, out_nc=dim//4, kernel_size=11, stride=2, padding=5), 
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # receptive_filed=55
        self.branch3 = nn.Sequential(
            Basiconv(in_nc=3, out_nc=dim//8, kernel_size=7, stride=2, padding=9, dilation=3),
            Basiconv(in_nc=dim//8, out_nc=dim//4, kernel_size=7, stride=2, padding=9, dilation=3), 
        )
        # receptive_filed=97
        self.branch4 = nn.Sequential(
            Basiconv(in_nc=3, out_nc=dim//8, kernel_size=9, stride=2, padding=16, dilation=4),
            Basiconv(in_nc=dim//8, out_nc=dim//4, kernel_size=9, stride=2, padding=16, dilation=4), 
        )  
        layer_scale_init_value = 1e-2 
        layer_scale_1 = layer_scale_init_value * torch.ones(((dim//4)), requires_grad=True).unsqueeze(-1).unsqueeze(-1).cuda()
        layer_scale_2 = layer_scale_init_value * torch.ones(((dim//4)), requires_grad=True).unsqueeze(-1).unsqueeze(-1).cuda()
        layer_scale_3 = layer_scale_init_value * torch.ones(((dim//4)), requires_grad=True).unsqueeze(-1).unsqueeze(-1).cuda()
        layer_scale_4 = layer_scale_init_value * torch.ones(((dim//4)), requires_grad=True).unsqueeze(-1).unsqueeze(-1).cuda()

        self.layer_scale_1 = nn.Parameter(layer_scale_1)
        self.layer_scale_2 = nn.Parameter(layer_scale_2)
        self.layer_scale_3 = nn.Parameter(layer_scale_3)
        self.layer_scale_4 = nn.Parameter(layer_scale_4)
        self.bn = nn.BatchNorm2d(dim)
 
    def forward(self, x):
        x1 = self.layer_scale_1 * self.branch1(x)
        x2 = self.layer_scale_2 * self.branch2(x)
        x3 = self.layer_scale_3 * self.branch3(x)
        x4 = self.layer_scale_4 * self.branch4(x) 
        value = torch.cat([x1,x2,x3,x4], dim=1)   
        return self.bn(value) 
 
class PatchMerging(nn.Module):  
    def __init__(self, dim, aux_train=False):
        super().__init__()  
        if aux_train:
            self.proj = nn.Identity()
        else:
            self.proj = nn.Linear(4* dim, 2*dim, bias=False) 
    def forward(self, x ):
        """
        x: B, H*W, C
        """
        _, _, H, W = x.shape  
        x = x.transpose(1,3)

        # padding 
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input: 
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # [B, H/2, W/2, C]
        x1 = x[:, 1::2, 0::2, :]  # [B, H/2, W/2, C]
        x2 = x[:, 0::2, 1::2, :]  # [B, H/2, W/2, C]
        x3 = x[:, 1::2, 1::2, :]  # [B, H/2, W/2, C]
        x = torch.cat([x0, x1, x2, x3], -1)  # [B, H/2, W/2, 4*C] 
        x = self.proj(x)
        return x.transpose(1,3)
 
class channel_fusion(nn.Module):
    def __init__(self, in_nc):
        super().__init__() 
        self.feature = nn.Sequential(
            nn.Conv2d(in_channels=in_nc, out_channels=in_nc , kernel_size=1), 
            nn.BatchNorm2d(in_nc ),
            nn.GELU()
        ) 
    def forward(self, x):   
        return self.feature(x)
 
class spatial_fusion(nn.Module):
    def __init__(self, in_nc):
        super().__init__() 
        self.feature = nn.Sequential(
            nn.Conv2d(in_channels=in_nc, out_channels=in_nc, kernel_size=3, padding=1, groups=in_nc),
            nn.BatchNorm2d(in_nc),
            nn.GELU()
        ) 
    def forward(self, x):   
        return self.feature(x)

class fusion(nn.Module):
    def __init__(self, in_dim ):
        super().__init__()
        self.channel_fusion = channel_fusion(in_nc=in_dim)
        self.spatial_fusion = spatial_fusion(in_nc=in_dim)  

        self.bn = nn.BatchNorm2d(in_dim) 
    def forward(self, x):
        x1 = self.channel_fusion(x)
        x2 = self.spatial_fusion(x)
        out = x1+x2
        out = self.bn(out)

        return out

class Block(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.attn_1 = fusion(in_dim)
        self.attn_2 = fusion(in_dim)
        self.act = nn.GELU()
    def forward(self, x):
        residual = x
        x = self.attn_1(x)
        x = self.attn_2(x)
        x = x + residual
        return self.act(x)

class   CMNet(nn.Module):
    def __init__(self, aux_train=False, in_channel = [3, 128, 256, 512, 1024], blocks_num=[2,2,6,2] , num_classes=1000 ):
        super().__init__() 
        self.aux_train = aux_train
        self.sub_1st = sub_1st(dim = in_channel[1])  
 
        self.stage1 = self._make_layer(block=Block, dim=in_channel[1] , block_num=blocks_num[0], sub=PatchMerging)  # [b, 192, 28, 28]
        self.stage2 = self._make_layer(block=Block, dim=in_channel[2] , block_num=blocks_num[1], sub=PatchMerging)  # [b, 384, 14, 14]
        self.stage3 = self._make_layer(block=Block, dim=in_channel[3] , block_num=blocks_num[2], sub=PatchMerging)  # [b, 768, 7, 7]
        self.stage4 = self._make_layer(block=Block, dim=in_channel[4] , block_num=blocks_num[3], sub=None)
 
        self.top = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # output size = (1, 1)
            nn.Flatten(start_dim=1),
            nn.Linear(in_channel[-1] , num_classes)
        )
        if aux_train :
            self.aux = nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  #[b, 96, 28, 28]
                PatchMerging(in_channel[1], aux_train=aux_train),             #[b, 384, 14, 14]
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  #[b, 384, 7, 7]  
                PatchMerging(in_channel[1]*4, aux_train=aux_train),            #[b, 1536, 4, 4] 
                nn.AdaptiveAvgPool2d((1, 1)),  # output size = (1, 1)
                nn.Flatten(start_dim=1),
                nn.Linear(in_channel[1]*16 , num_classes) 
            )

    def _make_layer(self, block, dim , block_num, sub): 
        layers = [] 
        for _ in range(block_num):
            layers.append(block(dim )) 
        if sub is not None:
            layers.append(sub(dim)) 
        return nn.Sequential(*layers)

  
    def forward(self, x ):
        x = self.sub_1st(x)  
        if self.aux_train:
            aux = self.aux(x)
        x = self.stage1(x) 
        x = self.stage2(x) 
        x = self.stage3(x) 
        x = self.stage4(x) 
        x = self.top(x)

        if self.aux_train:
            value = 0.3 * aux + 0.7 * x
        else:
            value = x
        return value

 

def CMNet_base(aux, in_channel=[3, 128, 256, 512, 1024], blocks_num=[2,2,6,2] , num_classes=1000 ):
    # 11,731,432
    return CMNet(aux, in_channel , blocks_num , num_classes )

 
if __name__ == "__main__":
    img = torch.ones([1, 3, 224, 224]).cuda()
    from torchsummary import summary
    model = CMNet_base(num_classes=1000)
    summary(model.to('cuda'), (3,224,224)) 
    y = model(img)
    print(y.shape)

 