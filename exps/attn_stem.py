# attn_stem

import torch.nn.functional as F
import torch
import torch.nn as nn 

class Basiconv(nn.Module):
    def __init__(self, in_nc, out_nc, kernel_size, stride, padding, dilation=1):
        super().__init__()
        self.featurs = nn.Sequential(
            nn.Conv2d(in_nc, out_nc, kernel_size, stride, padding, dilation=dilation),
            nn.BatchNorm2d(out_nc),
            nn.ReLU(True)
        )
    def forward(self,x):
        return self.featurs(x)
  
class sub_1st(nn.Module):
    def __init__(self, ):
        super().__init__()
        # receptive_filed= 7.0 
        self.branch1 = nn.Sequential(
            Basiconv(in_nc=3, out_nc=32, kernel_size=3, stride=2, padding=1), 
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # receptive_filed=15.0
        self.branch2 = nn.Sequential(
            Basiconv(in_nc=3, out_nc=32, kernel_size=11, stride=2, padding=5), 
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # receptive_filed=55
        self.branch3 = nn.Sequential(
            Basiconv(in_nc=3, out_nc=16, kernel_size=7, stride=2, padding=9, dilation=3),
            Basiconv(in_nc=16, out_nc=32, kernel_size=7, stride=2, padding=9, dilation=3), 
        )
        # receptive_filed=97
        self.branch4 = nn.Sequential(
            Basiconv(in_nc=3, out_nc=16, kernel_size=9, stride=2, padding=16, dilation=4),
            Basiconv(in_nc=16, out_nc=32, kernel_size=9, stride=2, padding=16, dilation=4), 
        )  

        self.proj_branch = nn.Sequential(
            nn.Linear(4, 16, False),
            nn.ReLU(),
            nn.Linear(16, 4, False) 
        ) 
        self.proj_channel = nn.Sequential(
            nn.Linear(32, 16, False),
            nn.ReLU(),
            nn.Linear(16, 32, False),
            nn.Sigmoid()
        ) 
        self.proj_last = nn.Sequential(
            nn.Conv2d(4*32, 64, kernel_size=1),
            nn.BatchNorm2d(64)
        ) 

    def forward(self, x): 
        B, _, H, W = x.shape
        branch_features_gap = []
        branch_features = []

        branch1 = self.branch1(x) 
        branch_features_gap.append(branch1.mean(-1).mean(-1))
        branch2 = self.branch2(x) 
        branch_features_gap.append(branch2.mean(-1).mean(-1))
        branch3 = self.branch3(x) 
        branch_features_gap.append(branch3.mean(-1).mean(-1))
        branch4 = self.branch4(x) 
        branch_features_gap.append(branch4.mean(-1).mean(-1))

        branch_features = [branch1,branch2,branch3,branch4]
        features=torch.stack(branch_features, dim=0) #[k, bs, c, h, w]  
        bf = torch.stack(branch_features_gap, dim=0).transpose(0,2) #[c, bs, k]

        branch_attn = F.softmax(self.proj_branch(bf), dim=-1).transpose(0,2) #[c, bs, k] --> proj+softmax --> [c, bs, k] --> trans --> [k, bs, c]
        channel_attn = self.proj_channel(branch_attn).unsqueeze(-1).unsqueeze(-1) #[k, bs, c, 1, 1] 
    
        outputs = features * channel_attn #[k, bs, c, h, w] 
 
        outputs = outputs.transpose(0,1).reshape(B, -1, 56, 56)  #[k, bs, c, h, w] -->  #[bs, k, c, h, w] -->  #[bs, k*c, h, w]
        outputs = self.proj_last(outputs)
        return outputs

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out

 
class TNet(nn.Module): 
    def __init__(self,
                 block,
                 blocks_num,
                 num_classes=1000,
                 include_top=True,
                 groups=1,
                 width_per_group=64):
        super().__init__()
        self.include_top = include_top
        self.in_channel = 64

        self.groups = groups
        self.width_per_group = width_per_group

        self.conv1 = sub_1st()
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x) 

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x 

def TNet(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return TNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, include_top=include_top)

if __name__ == "__main__":
    img = torch.ones([1, 3, 224, 224])
    from torchsummary import summary
    model = TNet(num_classes=1000)
    summary(model.to('cuda'), (3,224,224)) 


 