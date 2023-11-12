import torch.nn as nn
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PrimaryUserPresenceNetwork(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.num_neighbors = kwargs["num_neighbors"]

        self.self_module = SingleSecondaryUserModule(**kwargs)
        self.user_modules = [
            SingleSecondaryUserModule(**kwargs) for _ in range(self.num_neighbors)
        ]

        self.user_modules.insert(0, self.self_module)

        self.final_linear = nn.Linear(
            in_features=len(self.user_modules), out_features=1
        )

    def forward(self, in_recent_values, in_reputation_history):
        out = np.zeros(len(self.user_modules))
        for i, module in enumerate(self.user_modules):
            out[i] = module(in_recent_values[i], in_reputation_history[i])
        
        out = self.final_linear(out)
        return out
        


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

    def forward(self, current_reading_input, history_input):
        hidden = torch.zeros(self.lstm_num_layers, self.lstm_hidden_size).to(device)
        cells = torch.zeros(self.lstm_num_layers, self.lstm_hidden_size).to(device)

        output, _ = self.lstm(history_input, (hidden, cells))

        linear_in = np.concatenate(output[-1, :], current_reading_input)
        prediction = self.linear(linear_in)
        return prediction


def train(
    n_epochs: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.modules.loss._Loss,
    inputs: tuple[torch.Tensor],
    expected_output: torch.Tensor,
):
    for epoch in range(n_epochs):
        print(f"EPOCH {epoch + 1}")
        model.train()
        inputs = inputs.to(device)
        expected_output = expected_output.to(device)

        output = model(inputs)
        print(f"Training desired: {expected_output}, predicted: {output}")

        loss = criterion(output, expected_output)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Loss: {loss.item()}")

        model.eval()
        with torch.no_grad():
            output = model(inputs)

        print()  # add a line between epochs


def main():
    """
    INPUT: (float, float)
    - First number is the value that the neighbor reported
    - Second number is the ground truth of whether the PU
        is there (0 for no, 1 for yes)
    OUTPUT: float
    - Probability that the neighbor is malicious
    """
    from json import load
    from os.path import join
    from feature_select import make_features
    
    with open(join("outfile")) as f:
        raw_data = load(f)["data"]
        inputs = make_features(raw_data)
        expected_output = np.array([1.0 if raw_data[-1]["pu_present"] else 0.0])

    n_epochs = 100
    n_hidden = 128
    learning_rate = 0.005
    model = PrimaryUserPresenceNetwork(hidden_size=n_hidden, num_neighbors=len(raw_data[0]["neighbor_measurements"]), num_layers=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss().to(device)
    train(n_epochs, model, optimizer, criterion, inputs, expected_output)


if __name__ == "__main__":
    main()
