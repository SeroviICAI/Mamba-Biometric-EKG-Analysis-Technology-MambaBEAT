import torch
import torch.nn as nn
import torch.nn.init as init


class Block(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, stride: int) -> None:
        super(Block, self).__init__()

        self.net = nn.Sequential(
            nn.Conv1d(input_channels, output_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(
                output_channels,
                output_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv1d(output_channels, output_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(output_channels),  # Adding batch normalization
        )

        # Initialize weights using He initialization
        for m in self.net.modules():
            if isinstance(m, nn.Conv1d):
                init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


class YModel(nn.Module):
    def __init__(
        self,
        hidden_sizes: tuple[int, ...],
        input_channels: int = 1000,
        output_channels: int = 5,
    ) -> None:
        super(YModel, self).__init__()

        self.features = nn.Sequential(
            nn.Conv1d(
                input_channels, hidden_sizes[0], kernel_size=7, padding=3, stride=2
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(hidden_sizes[0]),
            *[
                Block(hidden_sizes[i - 1], hidden_sizes[i], stride=1)
                for i in range(1, len(hidden_sizes))
            ]
        )

        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Linear(hidden_sizes[-1], output_channels)
        self.softmax = nn.Softmax(1)

        # Initialize weights using He initialization
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.classifier(x)
        return self.softmax(x)
