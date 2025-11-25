#jk: WideResNet implementation for CIFAR-10
#based on "Wide Residual Networks" by Zagoruyko and Komodakis

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

__all__ = ['WideResNet', 'wideresnet28_10', 'wideresnet28_2', 'wideresnet40_2', 'wideresnet16_8']

def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropout_rate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, 
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, 
                               padding=1, bias=False)
        self.dropout_rate = dropout_rate
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
            )
    
    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        if self.dropout_rate > 0:
            out = F.dropout(out, p=self.dropout_rate, training=self.training)
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)
        return out

class WideResNet(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate=0.0, num_classes=10):
        super(WideResNet, self).__init__()
        self.in_planes = 16
        
        assert ((depth - 4) % 6 == 0), 'Wide-resnet depth should be 6n+4'
        n = (depth - 4) // 6
        k = widen_factor
        
        nStages = [16, 16*k, 32*k, 64*k]
        
        self.conv1 = nn.Conv2d(3, nStages[0], kernel_size=3, stride=1, 
                               padding=1, bias=False)
        self.layer1 = self._make_layer(BasicBlock, nStages[1], n, stride=1, 
                                       dropout_rate=dropout_rate)
        self.layer2 = self._make_layer(BasicBlock, nStages[2], n, stride=2, 
                                       dropout_rate=dropout_rate)
        self.layer3 = self._make_layer(BasicBlock, nStages[3], n, stride=2, 
                                       dropout_rate=dropout_rate)
        self.bn1 = nn.BatchNorm2d(nStages[3])
        self.linear = nn.Linear(nStages[3], num_classes)
        
        self.apply(_weights_init)
    
    def _make_layer(self, block, planes, num_blocks, stride, dropout_rate):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, dropout_rate))
            self.in_planes = planes
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def wideresnet28_10(dropout_rate=0.3):
    """WideResNet-28-10 - most popular for robustness research"""
    return WideResNet(depth=28, widen_factor=10, dropout_rate=dropout_rate)

def wideresnet28_2(dropout_rate=0.3):
    """WideResNet-28-2"""
    return WideResNet(depth=28, widen_factor=2, dropout_rate=dropout_rate)

def wideresnet40_2(dropout_rate=0.3):
    """WideResNet-40-2"""
    return WideResNet(depth=40, widen_factor=2, dropout_rate=dropout_rate)

def wideresnet16_8(dropout_rate=0.3):
    """WideResNet-16-8"""
    return WideResNet(depth=16, widen_factor=8, dropout_rate=dropout_rate)

def test(net):
    import numpy as np
    total_params = 0
    
    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.cpu().numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))

if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith('wideresnet'):
            print(net_name)
            test(globals()[net_name]())
            print()