import torch.nn as nn
import torch
from os.path import join
from json import load

"""
Documentation on nn.RNN module:
    https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
    
"From scratch" tutorial on RNNs in PyTorch:
    https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html

Tutorial on RNNs that's a bit more practical:
    https://blog.floydhub.com/a-beginners-guide-on-recurrent-neural-networks-with-pytorch/

Tutorial code on one-to-many RNN/LSTM:
    https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/recurrent_neural_network/main.py


According to this: https://stackoverflow.com/questions/48484985/why-is-the-loss-of-my-recurrent-neural-network-zero?rq=3
it sounds possible that we need to make our input batched - right now, it's training
perfectly on the unbatched input (hence a loss of 0) but not generalizing.
"""

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class NeighborClassificationNetwork(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int = 1):
        super().__init__()

        # Input size is 2; first feature is the value the neighbor reported,
        # and second feature is the actual presence of the PU (0 or 1)
        self.input_size = 2
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Input should be (L, H_in) for unbatched input
        # L = sequence length
        # H_in = input_size (2)
        # So input should be of shape (sequence_length, 2)
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True
        )

        # Output of linear layer is just 1 feature; we want to classify
        # the neighbor as either non-malicious (0) or malicious (1)
        self.linear = nn.Linear(
            in_features=self.hidden_size,
            out_features=1
        )

    def forward(self, input):
        hidden = torch.zeros(self.num_layers, self.hidden_size).to(device)
        cells = torch.zeros(self.num_layers, self.hidden_size).to(device)

        output, _ = self.lstm(input, (hidden, cells))
        prediction = self.linear(output)[-1]
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
        inputs = inputs.to(device)
        expected_output = expected_output.to(device)

        # Possibility: all of the training data only has one output,
        # which means that it instantly learns to always output a single,
        # simple value - thus instant loss of 0.0?
        output = model(inputs)
        loss = criterion(output, expected_output)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Loss at epoch {epoch + 1}: {loss.item()}")


def main():
    """
    INPUT: (float, float)
    - First number is the value that the neighbor reported
    - Second number is the ground truth of whether the PU
        is there (0 for no, 1 for yes)
    OUTPUT: float
    - Probability that the neighbor is malicious
    """
    with open(join("training_data", "train-0.5.json")) as f:
        raw_data = load(f)
        inputs = torch.tensor(raw_data["inputs"])
        expected_output = torch.tensor([raw_data["output"]])

    n_epochs = 100
    n_hidden = 128
    learning_rate = 0.005
    model = NeighborClassificationNetwork(hidden_size=n_hidden, num_layers=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss().to(device)
    train(n_epochs, model, optimizer, criterion, inputs, expected_output)


if __name__ == "__main__":
    main()
