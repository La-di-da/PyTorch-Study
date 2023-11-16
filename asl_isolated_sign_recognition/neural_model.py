from torch import nn
from torch.nn import functional

# Define model
class AslSignModel(nn.Module):
    def __init__(self, frame_dim, hidden_dim, tagset_size):
        super().__init__()
        self.hidden_dim = hidden_dim

        # The LSTM takes frame vectors as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(frame_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, frame_tensor):
        lstm_out, _ = self.lstm(frame_tensor)
        tag_space = self.hidden2tag(lstm_out)
        tag_scores = functional.log_softmax(tag_space, dim=1)
        return tag_scores