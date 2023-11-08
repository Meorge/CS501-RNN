import torch.nn as nn
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PrimaryUserPresenceNetwork(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.num_neighbors = kwargs["num_neighbors"]

        self.self_module = SingleSecondaryUserModule(kwargs)
        self.neighbor_modules = [
            SingleSecondaryUserModule(kwargs) for _ in range(self.num_neighbors)
        ]

        self.final_linear = nn.Linear(in_features=0, out_features=2)

    def forward(self, input):
        ...

class SingleSecondaryUserModule(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.num_history = kwargs["num_history"]
        self.lstm_hidden_size = kwargs["lstm_hidden_size"]
        self.lstm_num_layers = kwargs["lstm_num_layers"]
        self.out_size = kwargs["out_size"]

        self.lstm = nn.LSTM(
            input_size=self.num_history,
            hidden_size=self.lstm_hidden_size,
            num_layers=self.lstm_num_layers,
            batch_first=True,
        )

        self.linear = nn.Linear(
            in_features=self.lstm_hidden_size + 1, out_features=self.out_size
        )

    def forward(self, input):
        hidden = torch.zeros(self.lstm_num_layers, self.lstm_hidden_size).to(device)
        cells = torch.zeros(self.lstm_num_layers, self.lstm_hidden_size).to(device)

        output, _ = self.lstm(input, (hidden, cells))
        prediction = self.linear(output[-1, :])
        return prediction
