from torch import nn


class LSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_dim: int = 5,
        bidirectional: bool = False,
    ):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        expand = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_size * expand, output_dim)
        self.softmax = nn.Softmax(1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return self.softmax(out)
