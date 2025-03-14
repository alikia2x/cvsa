import torch.nn as nn

class CompactPredictor(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(input_size),
            nn.Linear(input_size, 256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Tanh(),  # Use Tanh to limit the output range
            nn.Linear(64, 1)
        )
        # Initialize the last layer to values close to zero
        nn.init.uniform_(self.net[-1].weight, -0.01, 0.01)
        nn.init.constant_(self.net[-1].bias, 0.0)

    def forward(self, x):
        return self.net(x)
