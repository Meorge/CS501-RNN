import torch.nn as nn
import torch

class NeighborClassificationNetwork(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int = 1):
        super().__init__()

        # Input size should be 1, I think, since the input at each
        # time step is just the value reported by the neighbor.
        self.input_size = 1
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.RNN(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers
        )

        self.linear = nn.Linear(
            in_features=self.hidden_size,
            out_features=1  # we want classification of malicious or not - just one number?
        )

    def forward(self, input, hidden):
        output, hidden = self.rnn(input, hidden)
        prediction = self.linear(hidden)
        return prediction
    
    
n_hidden = 128
learning_rate = 0.005
classifier = NeighborClassificationNetwork(
    hidden_size=n_hidden,
    num_layers=1
)

criterion = nn.NLLLoss()

data_point = torch.tensor([[-70.0], [-70.0], [-70.0]])
true_classification = torch.tensor([[1.0], [1.0], [1.0]])

def train():
    hidden = torch.zeros(1, n_hidden)

    classifier.zero_grad()

    # TODO: feed each data point in sequence (t=0, then t=1, t=2, etc)
    for i in range(10):
        output, hidden = classifier(data_point, hidden)

    loss = criterion(output, true_classification)
    loss.backward()

    for p in classifier.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item()

train()