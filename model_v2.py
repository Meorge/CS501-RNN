from json import dump
from time import time
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
        self.hidden_size = kwargs["hidden_size"]

        self.self_module = SingleSecondaryUserModule(hidden_size=self.hidden_size)
        self.user_modules = [
            SingleSecondaryUserModule(hidden_size=self.hidden_size) for _ in range(self.num_neighbors)
        ]

        self.user_modules.insert(0, self.self_module)

        self.final_linear = nn.Linear(
            in_features=len(self.user_modules), out_features=1
        )

    def forward(self, in_recent_values, in_reputation_history):
        out = torch.zeros(len(self.user_modules))
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
        output = output[-1]  # potential issue: only using last hidden cell?
        output = output.flatten()

        current_reading_input = current_reading_input.unsqueeze(0)
        linear_in = torch.cat((output, current_reading_input))
        prediction = self.linear(linear_in)
        return prediction


def train(
    n_epochs: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.modules.loss._Loss,
    training_data: list[tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]],
):
    final_epoch_losses = []
    for epoch in range(n_epochs):
        last_loss = None
        epoch_start_time = time()
        for i, (
            (input_most_recent_values, input_rep_history_for_users),
            expected_output,
        ) in enumerate(training_data):
            model.train()

            input_most_recent_values = input_most_recent_values.to(device)
            input_rep_history_for_users = input_rep_history_for_users.to(device)

            expected_output = expected_output.to(device)

            output = model(input_most_recent_values, input_rep_history_for_users)

            loss = criterion(output, expected_output)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            last_loss = loss.item()

        epoch_end_time = time()

        print(
            f"EPOCH {epoch + 1}/{n_epochs}, Duration = {epoch_end_time - epoch_start_time:.2f}, Loss = {last_loss}"
        )

        final_epoch_losses.append((epoch_end_time - epoch_start_time, last_loss))

    return final_epoch_losses

    # model.eval()
    # with torch.no_grad():
    #     output = model(input_most_recent_values, input_rep_history_for_users)


def main():
    from load_training_data import get_all_training_data

    # Save the time that this training session started at
    time_string = f"{int(time())}"

    training_data = [
        (
            (
                torch.tensor(in_most_recent, dtype=torch.float32).to(device),
                torch.tensor(in_rep_hist, dtype=torch.float32).to(device),
            ),
            torch.tensor(out_expected, dtype=torch.float32).to(device),
        )
        for ((in_most_recent, in_rep_hist), out_expected) in get_all_training_data()
    ]

    # Breakdown of what this means:
    # - training_data is a list of pairs of input data and an output.
    # - training_data[0] is then a single pair of input data and an output,
    #   in the form ((in_most_recent, in_rep_hist), out_expected).
    # - training_data[0][0] is the first item in this tuple, which is itself
    #   a tuple containing the two sets of input data, (in_most_recent, in_rep_hist).
    # - training_data[0][0][0] is just in_most_recent, which is a tensor
    #   containing the last 100 values each node has measured.
    # - len(training_data[0][0][0]) is the length of this tensor, which is the
    #   total number of nodes in this network (minus the primary user).
    # - len(training_data[0][0][0]) - 1 is the number of neighbors that this
    #   node has, since it is not considered a neighbor of itself.
    n_neighbors = len(training_data[0][0][0]) - 1

    n_epochs = 50
    n_hidden_options = [8, 16, 32, 64, 128]
    lr_options = [1e-1, 1e-2, 1e-3, 5e-2, 5e-3]

    for n_hidden in n_hidden_options:
        for lr in lr_options:
            print(f"Training with {n_hidden} hidden size and LR {lr}")
            model = PrimaryUserPresenceNetwork(
                hidden_size=n_hidden,
                num_neighbors=n_neighbors,
            ).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            criterion = nn.MSELoss().to(device)

            start_time = time()
            final_epoch_losses = train(
                n_epochs,
                model,
                optimizer,
                criterion,
                training_data,
            )

            finish_time = time()

            info_string = f"hidden-{n_hidden}-lr-{lr}-time-{time_string}"

            with open(f"test_results_fixed_power/losses-{info_string}.json", "w") as f:
                dump(
                    {
                        "meta": {
                            "n_epochs": n_epochs,
                            "n_hidden": n_hidden,
                            "lr": lr,
                            "n_neighbors": n_neighbors,
                        },
                        "time": int(time()),
                        "duration": finish_time - start_time,
                        "epoch_losses": final_epoch_losses,
                    },
                    f,
                )

            torch.save(model.state_dict(), f"test_results_fixed_power/model-{info_string}.ckpt")


if __name__ == "__main__":
    main()
