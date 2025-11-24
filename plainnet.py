'''
Plain CNN architectures matching ResNet structure but WITHOUT skip connections.
Designed to have similar depth and parameter count as corresponding ResNets.

For fair comparison with ResNet architectures on CIFAR-10.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

__all__ = ['PlainNet', 'plainnet20', 'plainnet32', 'plainnet44', 'plainnet56', 'plainnet110']

def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class PlainBlock(nn.Module):
    """
    Plain convolutional block (no skip connections).
    Mirrors ResNet's BasicBlock structure but without the residual addition.
    """
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, dropout=0.0):
        super(PlainBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.dropout1 = nn.Dropout(p=dropout)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout1(out)
        out = self.bn2(self.conv2(out))
        out = self.dropout2(out)
        # NO skip connection here - this is the key difference from ResNet
        out = F.relu(out)
        return out


class PlainNet(nn.Module):
    """
    Plain CNN with same structure as ResNet but no skip connections.
    Uses the same number of layers and similar parameter count for fair comparison.
    """
    def __init__(self, block, num_blocks, num_classes=10, dropout=0.0):
        super(PlainNet, self).__init__()
        self.in_planes = 16
        self.dropout = dropout

        # Initial conv layer
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        
        # Three stages of plain blocks (matching ResNet architecture)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        
        # Final classification layer
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, dropout=self.dropout))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def plainnet20(num_classes=10, dropout=0.0):
    """Plain CNN with 20 layers (no skip connections)"""
    return PlainNet(PlainBlock, [3, 3, 3], num_classes=num_classes, dropout=dropout)


def plainnet32(num_classes=10, dropout=0.0):
    """Plain CNN with 32 layers (no skip connections)"""
    return PlainNet(PlainBlock, [5, 5, 5], num_classes=num_classes, dropout=dropout)


def plainnet44(num_classes=10, dropout=0.0):
    """Plain CNN with 44 layers (no skip connections)"""
    return PlainNet(PlainBlock, [7, 7, 7], num_classes=num_classes, dropout=dropout)


def plainnet56(num_classes=10, dropout=0.0):
    """Plain CNN with 56 layers (no skip connections)"""
    return PlainNet(PlainBlock, [9, 9, 9], num_classes=num_classes, dropout=dropout)


def plainnet110(num_classes=10, dropout=0.0):
    """Plain CNN with 110 layers (no skip connections)"""
    return PlainNet(PlainBlock, [18, 18, 18], num_classes=num_classes, dropout=dropout)


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))


if __name__ == "__main__":
    print("Testing Plain CNNs (no skip connections):")
    print("="*60)
    
    for net_name in __all__:
        if net_name.startswith('plainnet'):
            print(f"\n{net_name}:")
            model = globals()[net_name]()
            test(model)
            
            # Test forward pass
            x = torch.randn(2, 3, 32, 32)
            y = model(x)
            print(f"Output shape: {y.shape}")
    
    print("\n" + "="*60)
    print("Comparison: These PlainNets have the SAME architecture as ResNets")
    print("but WITHOUT skip connections. This lets us test the effect of")
    print("residual connections on adversarial robustness!")
    print("="*60)
