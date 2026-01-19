from torch import nn
from torch.nn import functional as f


class EuroNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(3, 1, 5)
        self.linear = nn.Linear(60 * 60, 10)

    def forward(self, x):
        a1 = self.conv(x)
        z1 = f.relu(a1)
        a2 = self.linear(z1.reshape(-1, 60 * 60))
        z2 = f.relu(a2)
        return z2
