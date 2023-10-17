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
    UPDATE: This might not be the case. As discussed below, the output of the model doesn't
    match what's expected. So it doesn't seem to just be learning "output 0.5 no matter what"
    or something similar.
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
            batch_first=True,
        )

        # Output of linear layer is just 1 feature; we want to classify
        # the neighbor as either non-malicious (0) or malicious (1)
        self.linear = nn.Linear(in_features=self.hidden_size, out_features=1)

    def forward(self, input):
        hidden = torch.zeros(self.num_layers, self.hidden_size).to(device)
        cells = torch.zeros(self.num_layers, self.hidden_size).to(device)

        output, _ = self.lstm(input, (hidden, cells))
        prediction = self.linear(output[-1, :])
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

        # Note: In the working RNN example, I can see that the expected output
        #  tensor (from variable `labels`) is an index (like 1, 2, 3, etc).
        #  However, the prediction appears to be a one-hot vector. For example,
        #  I think the `labels` value of `1` would map to a prediction value
        #  of `tensor([0, 1, 0, 0, 0, 0, 0, 0, 0, 0])`. So the output and
        #  expected output shouldn't be in the same format...?
        #  
        #  HAHA! I think I found the issue! The loss function we were using,
        #  `CrossEntropyLoss`, is best for discrete classes (such as object
        #  types, in the sample code we were using). But in our case, what we
        #  wanted was just a probability. Using `L1Loss` seems to have fixed
        #  the issue!
        loss = criterion(output, expected_output)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Loss: {loss.item()}")

        model.eval()
        with torch.no_grad():
            """
            Try testing the model to see what its output is, given the
            inputs. If this was an overfitting issue, I'd expect it to give
            the perfect output value.

            HOWEVER, it's not (output should be 0.5 and it's -0.0339 in a trial
            I just did). So the issue must be elsewhere...?
            """
            output = model(inputs)
            # print(f"Testing desired: {expected_output[0]}, predicted: {output[0]}")

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
    with open(join("train-0.5.json")) as f:
        raw_data = load(f)
        inputs = torch.tensor(raw_data["inputs"])
        expected_output = torch.tensor([raw_data["output"]])

    n_epochs = 100
    n_hidden = 128
    learning_rate = 0.005
    model = NeighborClassificationNetwork(hidden_size=n_hidden, num_layers=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss().to(device)
    train(n_epochs, model, optimizer, criterion, inputs, expected_output)


if __name__ == "__main__":
    main()
