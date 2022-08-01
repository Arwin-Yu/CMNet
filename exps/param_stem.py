# param_stem
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
        layer_scale_init_value = 1e-2 
        layer_scale_1 = layer_scale_init_value * torch.ones(((32)), requires_grad=True).unsqueeze(-1).unsqueeze(-1).cuda()
        layer_scale_2 = layer_scale_init_value * torch.ones(((32)), requires_grad=True).unsqueeze(-1).unsqueeze(-1).cuda()
        layer_scale_3 = layer_scale_init_value * torch.ones(((32)), requires_grad=True).unsqueeze(-1).unsqueeze(-1).cuda()
        layer_scale_4 = layer_scale_init_value * torch.ones(((32)), requires_grad=True).unsqueeze(-1).unsqueeze(-1).cuda()

        self.layer_scale_1 = nn.Parameter(layer_scale_1)
        self.layer_scale_2 = nn.Parameter(layer_scale_2)
        self.layer_scale_3 = nn.Parameter(layer_scale_3)
        self.layer_scale_4 = nn.Parameter(layer_scale_4)
        self.bn = nn.BatchNorm2d(128)

        self.proj_last = nn.Sequential(
            nn.Conv2d(4*32, 64, kernel_size=1),
            nn.BatchNorm2d(64)
        ) 
        
    def forward(self, x):
        x1 = self.layer_scale_1 * self.branch1(x)
        x2 = self.layer_scale_2 * self.branch2(x)
        x3 = self.layer_scale_3 * self.branch3(x)
        x4 = self.layer_scale_4 * self.branch4(x) 
        value = torch.cat([x1,x2,x3,x4], dim=1)   
        value = self.bn(value)
        value = self.proj_last(value)
        return value

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
        super( ).__init__()
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

 