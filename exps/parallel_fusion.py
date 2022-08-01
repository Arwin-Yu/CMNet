#  11,258,056 
import torch.nn.functional as F
import torch
import torch.nn as nn  
 
class sub_1st(nn.Module):
    def __init__(self, ):
        super().__init__()
        # receptive_filed= 7.0 
        self.conv1 = nn.Conv2d(3, 96, kernel_size=7, stride=2,  padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(96)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
 
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  
        return x
 
class PatchMerging(nn.Module):  
    def __init__(self, dim):
        super().__init__()  
        self.proj = nn.Linear(4* dim, 2*dim, bias=False) 
    def forward(self, x ):
        """
        x: B, H*W, C
        """
        _, _, H, W = x.shape  
        x = x.transpose(1,3)

        # padding
        # 如果输入feature map的H，W不是2的整数倍，需要进行padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            # to pad the last 3 dimensions, starting from the last dimension and moving forward.
            # (C_front, C_back, W_left, W_right, H_top, H_bottom)
            # 注意这里的Tensor通道是[B, H, W, C]，所以会和官方文档有些不同
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

class TNet(nn.Module):
    def __init__(self, in_channel=[3, 96, 192, 384, 768], blocks_num=[4,4,12,4] , num_classes=1000):
        super().__init__()
        
        self.sub_1st = sub_1st() # [b, 96, 56, 56]
 
        self.stage1 = self._make_layer(block=Block, dim=in_channel[1] , block_num=blocks_num[0], sub=PatchMerging)  # [b, 192, 28, 28]
        self.stage2 = self._make_layer(block=Block, dim=in_channel[2] , block_num=blocks_num[1], sub=PatchMerging)  # [b, 384, 14, 14]
        self.stage3 = self._make_layer(block=Block, dim=in_channel[3] , block_num=blocks_num[2], sub=PatchMerging)  # [b, 768, 7, 7]
        self.stage4 = self._make_layer(block=Block, dim=in_channel[4] , block_num=blocks_num[3], sub=None)

        self.top = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # output size = (1, 1)
            nn.Flatten(start_dim=1),
            nn.Linear(768 , num_classes)
        )
    def _make_layer(self, block, dim , block_num, sub):
 
        layers = [] 
        for _ in range(block_num):
            layers.append(block(dim ))
        
        if sub is not None:
            layers.append(sub(dim))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.sub_1st(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x) 
        x = self.top(x)
        return x


def TNet(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return TNet()


if __name__ == "__main__":
    img = torch.ones([1, 3, 224, 224])
    from torchsummary import summary
    model = TNet(num_classes=1000)
    summary(model.to('cuda'), (3,224,224)) 
 