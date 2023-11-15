from torch import nn

# Define model
class AslSignModel(nn.Module):
    def __init__(self, lexicon_size, label_size, layer_a = 32, layer_b = 32):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(lexicon_size, layer_a),
            nn.ReLU(),
            nn.Linear(layer_a, layer_b),
            nn.ReLU(),
            nn.Linear(layer_b, label_size)
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits