import torch


class Bottleneck(torch.nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = torch.nn.Conv1d(
            in_channels, out_channels, kernel_size=1, bias=False
        )
        self.bn1 = torch.nn.BatchNorm1d(out_channels)
        self.conv2 = torch.nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = torch.nn.BatchNorm1d(out_channels)
        self.conv3 = torch.nn.Conv1d(
            out_channels, self.expansion * out_channels, kernel_size=1, bias=False
        )
        self.bn3 = torch.nn.BatchNorm1d(self.expansion * out_channels)

        self.shortcut = torch.nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv1d(
                    in_channels,
                    self.expansion * out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                torch.nn.BatchNorm1d(self.expansion * out_channels),
            )

    def forward(self, x):
        out = torch.nn.functional.relu(self.bn1(self.conv1(x)))
        out = torch.nn.functional.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = torch.nn.functional.relu(out)
        return out


class ResNet(torch.nn.Module):
    def __init__(self, block, num_blocks, num_classes, dropout=0.2):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = torch.nn.Conv1d(
            1000, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = torch.nn.BatchNorm1d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear = torch.nn.Linear(512 * block.expansion, num_classes)
        self.softmax = torch.nn.Softmax(1)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        out = torch.nn.functional.relu(self.bn1(self.conv1(x)))
        out = torch.nn.functional.max_pool1d(out, kernel_size=3, stride=2, padding=1)
        out = self.dropout(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = torch.nn.functional.avg_pool1d(out, out.size()[2])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return self.softmax(out)


def ResNet152(num_classes):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes)


def ResNet101(num_classes):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes)
