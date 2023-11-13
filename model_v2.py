import torch.nn as nn
import torch
import numpy as np

try:
    from rich import print
except ImportError:
    print("rich not installed, using builtin print")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PrimaryUserPresenceNetwork(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.num_neighbors = kwargs["num_neighbors"]

        self.self_module = SingleSecondaryUserModule()
        self.user_modules = [
            SingleSecondaryUserModule() for _ in range(self.num_neighbors)
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
    def __init__(self, hidden_size=128, num_layers=1, out_size=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.out_size = out_size

        # Input is tensor of shape (L, H_in) for unbatched input
        # L = sequence length
        # H_in = input_size (1)
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
        )

        self.linear = nn.Linear(
            in_features=self.hidden_size + 1, out_features=self.out_size
        )

    def forward(self, current_reading_input: torch.Tensor, history_input: torch.Tensor):
        hidden = torch.zeros(self.num_layers, self.hidden_size).to(device)
        cells = torch.zeros(self.num_layers, self.hidden_size).to(device)

        history_input = history_input.unsqueeze(1)
        output, _ = self.lstm(history_input, (hidden, cells))
        
        output: torch.Tensor
        
        output = output.flatten()
        
        current_reading_input = current_reading_input.unsqueeze(0)
        print(output, current_reading_input)
        linear_in = np.concatenate((output, current_reading_input))
        print(linear_in)
        
        prediction = self.linear(linear_in)
        return prediction


def train(
    n_epochs: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.modules.loss._Loss,
    input_most_recent_values: torch.Tensor,
    input_rep_history_for_users: torch.Tensor,
    expected_output: torch.Tensor,
):
    for epoch in range(n_epochs):
        print(f"EPOCH {epoch + 1}")
        model.train()

        input_most_recent_values = input_most_recent_values.to(device)
        input_rep_history_for_users = input_rep_history_for_users.to(device)

        expected_output = expected_output.to(device)

        output = model(input_most_recent_values, input_rep_history_for_users)
        print(f"Training desired: {expected_output}, predicted: {output}")

        loss = criterion(output, expected_output)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Loss: {loss.item()}")

        model.eval()
        with torch.no_grad():
            output = model(input_most_recent_values, input_rep_history_for_users)

        print()  # add a line between epochs


def main():
    from json import load
    from os.path import join
    from feature_select import make_features

    with open(join("outfile")) as f:
        raw_data = load(f)["data"]
        input_most_recent_values, input_rep_history_for_users = make_features(raw_data)
        expected_output = np.array([1.0 if raw_data[-1]["pu_present"] else 0.0])

    input_most_recent_values = torch.tensor(
        input_most_recent_values, dtype=torch.float32
    ).to(device)
    input_rep_history_for_users = torch.tensor(
        input_rep_history_for_users, dtype=torch.float32
    ).to(device)
    expected_output = torch.tensor(expected_output, dtype=torch.float32).to(device)

    n_epochs = 100
    n_hidden = 128
    learning_rate = 0.005
    model = PrimaryUserPresenceNetwork(
        hidden_size=n_hidden, num_neighbors=len(raw_data[0]["neighbor_measurements"])
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss().to(device)
    train(
        n_epochs,
        model,
        optimizer,
        criterion,
        input_most_recent_values,
        input_rep_history_for_users,
        expected_output,
    )


if __name__ == "__main__":
    main()
