import torch.nn as nn
import torch

# Some data I generated.
# First item in tuple is inputs - item[0] is reported value, item[1] is presence of PU
# Second item is probability that neighbor is malicious
test_data = [
    ((80.0, 0), 0.4),
    ((80.0, 0), 0.4),
    ((-111.0, 0), 0.4),
    ((-111.0, 0), 0.4),
    ((80.0, 0), 0.4),
    ((-111.0, 0), 0.4),
    ((-111.0, 0), 0.4),
    ((80.0, 0), 0.4),
]



class NeighborClassificationNetwork(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int = 1):
        super().__init__()

        # Input size is 2; first feature is the value the neighbor reported,
        # and second feature is the actual presence of the PU (0 or 1)
        self.input_size = 2
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.RNN(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
        )

        # Output of linear layer is just 1 feature; we want to classify
        # the neighbor as either non-malicious (0) or malicious (1)
        self.linear = nn.Linear(in_features=self.hidden_size, out_features=1)

    def forward(self, input, hidden):
        output, hidden = self.rnn(input, hidden)
        prediction = self.linear(hidden)
        return prediction


n_hidden = 128
learning_rate = 0.005
classifier = NeighborClassificationNetwork(hidden_size=n_hidden, num_layers=1)

criterion = nn.CrossEntropyLoss()
"""
INPUT: (float, float)
  - First number is the value that the neighbor reported
  - Second number is the ground truth of whether the PU
    is there (1 for yes, 2 for no)
OUTPUT: float
  - Probability that the neighbor is malicious
"""
data_point = torch.tensor([list(i) for i, _ in test_data])
true_classification = torch.tensor([[test_data[0][1]]])


def train():
    hidden = torch.zeros(1, n_hidden)

    classifier.zero_grad()

    # documentation on RNN module: https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
    # tutorial on RNNs: https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html

    # Input should be (L, H_in) for unbatched input
    # L = sequence length
    # H_in = input_size (2)
    # So input should be of shape (sequence_length, 2)
    output = classifier(data_point, hidden)

    loss = criterion(output, true_classification)
    loss.backward()

    for p in classifier.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item()

def main():
    current_loss = 0.0
    n_iters = 100
    for i in range(n_iters):
        output, loss = train()
        current_loss += loss
        print(output)

    print(f"Loss after {n_iters} iterations: {current_loss}")

    
main()


