'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class HeightSpatialAttentionFC(nn.Module):
    def __init__(self, height):
        super(HeightSpatialAttentionFC, self).__init__()

        self.height = height
        self.hidden = max(self.height//16, 1)
        self.fc = nn.Sequential(
            nn.Linear(self.height, self.hidden, bias = False),
            nn.ReLU(),
            nn.Linear(self.hidden, self.height, bias = False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # get avg pooling or max pooling
        avg_out = torch.mean(x, dim=[1,3], keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        max_out, _ = torch.max(max_out, dim=3, keepdim=True)
        avg_out = self.fc(avg_out.transpose(2,3)).transpose(2,3)
        max_out = self.fc(max_out.transpose(2,3)).transpose(2,3)
        out = avg_out + max_out
        return self.sigmoid(out)

class HeightSpatialAttention(nn.Module):
    def __init__(self, kernel_size=7, height=0):
        super(HeightSpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 得到max和avg
        avg_out = torch.mean(x, dim=[1,3], keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        max_out, _ = torch.max(max_out, dim=3, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class BasicBlock_MTL(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, attention =True, share = True, input_split = False, height = 32, attn_type = 'conv'):
        super(BasicBlock_MTL, self).__init__()
        self.attention = attention
        self.share = share
        self.input_split = input_split
        self.height = height

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv1_dual = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1_dual = nn.BatchNorm2d(planes)
        self.conv2_dual = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2_dual = nn.BatchNorm2d(planes)

        #self.ca = ChannelAttention(planes)
        if self.attention == True:
            if attn_type == 'fc':
                self.sa = HeightSpatialAttentionFC(height=self.height)
                self.sa_dual = HeightSpatialAttentionFC(height = self.height)
            else:
                self.sa = HeightSpatialAttention(height=self.height)
                self.sa_dual = HeightSpatialAttention(height = self.height)

        self.shortcut = nn.Sequential()
        self.shortcut_dual = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
            self.shortcut_dual = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        if not self.input_split:
            x1 = x
            x2 = x
        else:
            x1 = x[0]
            x2 = x[1]
        out = F.relu(self.bn1(self.conv1(x1)))
        out = self.bn2(self.conv2(out))
        if self.attention:
            self.weight = self.sa(out)
            out = self.weight*out
        out += self.shortcut(x1)
        out = F.relu(out)
        if self.share:
            return out

        elif not self.share:
            out2 = F.relu(self.bn1_dual(self.conv1_dual(x2)))
            out2 = self.bn2_dual(self.conv2_dual(out2))
            if self.attention:
                self.weight2 = self.sa(out2)
                out2 = self.weight2*out2
            out2 += self.shortcut(x2)
            out2 = F.relu(out2)
            return out, out2

    def getattention(self):
        if not self.share:
            return self.weight, self.weight2
        else:
            return self.weight


class ResNet_t_MTL(nn.Module):
    def __init__(self, block, num_blocks, attention, large_img, share_layer_order, attn_type = 'conv'):
        super(ResNet_t_MTL, self).__init__()
        self.in_planes = 64
        self.large_img = large_img
        self.share_layer_order = share_layer_order
        self.attn_type = attn_type
        if large_img:
            self.height = 56
        else:
            self.height = 32

        if not self.large_img:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            if self.share_layer_order == -1:
                self.conv1_dual = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
                self.bn1_dual = nn.BatchNorm2d(64)
        elif self.large_img:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            if self.share_layer_order == -1:
                self.conv1_dual = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
                self.bn1_dual = nn.BatchNorm2d(64)
                self.relu_dual = nn.ReLU(inplace=True)
                self.maxpool_dual = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, attention = attention, \
                        share = share_layer_order>=1, input_split = share_layer_order<0, height=self.height)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, attention = attention, \
                        share = share_layer_order>=2, input_split = share_layer_order<1, height=self.height//2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, attention = attention, \
                        share = share_layer_order>=3, input_split = share_layer_order<2, height=self.height//4)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, attention = attention, \
                        share = share_layer_order>=4, input_split = share_layer_order<3, height=self.height//8)


        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if self.share_layer_order < 4:
            self.avgpool_dual = nn.AdaptiveAvgPool2d((1,1))

        self.attn_weights = []
        self.attn_weights_dual = []

    def _make_layer(self, block, planes, num_blocks, stride, attention, share, input_split, height):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, attention, share, input_split, height, self.attn_type))
            input_split = not share
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.large_img:
            out = self.maxpool(self.relu(self.bn1(self.conv1(x))))
            if self.share_layer_order < 0:
                out2 = self.maxpool_dual(self.relu_dual(self.bn1_dual(self.conv1_dual(x))))
                out = out,out2
        else:
            out = F.relu(self.bn1(self.conv1(x)))
            if self.share_layer_order < 0:
                out2 = F.relu(self.bn1_dual(self.conv1_dual(x)))
                out = out, out2
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        self.attn_weights = []
        self.attn_weights_dual = []
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            if layer[-1].attention == True:
                if layer[-1].share:
                    self.attn_weights.append(layer[-1].getattention())
                    self.attn_weights_dual.append(layer[-1].getattention())
                else:
                    self.attn_weights.append(layer[-1].getattention()[0])
                    self.attn_weights_dual.append(layer[-1].getattention()[1])
        if self.share_layer_order >=4:
            out = self.avgpool(out)
            out = out.view(out.size(0), -1)
            return out
        # out = self.linear(out)
        else:
            out1 = self.avgpool(out[0])
            out1 = out1.view(out1.size(0), -1)
            out2 = self.avgpool_dual(out[1])
            out2 = out2.view(out2.size(0), -1)
            return out1, out2

    def get_attention_weights(self):
        # This must be get after forward step
        return self.attn_weights,self.attn_weights_dual

def resnet18_mtl(attention = True, large_img = True, share_layer_order = 4, attn_type = 'conv'):
    return ResNet_t_MTL(BasicBlock_MTL, [2,2,2,2], attention, large_img, share_layer_order, attn_type)


def test():
    net = resnet18_mtl()
    y = net(torch.randn(64,3, 224,224))
    print(y.size())

if __name__ == '__main__':
    test()