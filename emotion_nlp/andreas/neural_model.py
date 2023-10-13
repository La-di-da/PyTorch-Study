from torch import nn

# Define model
class EmotionClassifer(nn.Module):
    def __init__(self, lexicon_size, label_size):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(lexicon_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, label_size)
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits