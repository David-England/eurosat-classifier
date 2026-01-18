import torch
import torchvision
from torch import nn, optim
from torch.nn import functional as f
from torch.utils.data import DataLoader, random_split


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


torch.manual_seed(4)

# Load data
dset = torchvision.datasets.EuroSAT(
    "/eurosatd", download=True, transform=torchvision.transforms.ToTensor()
)
dset_train, dset_test = random_split(dset, [0.8, 0.2])

loader_train = DataLoader(dset_train, batch_size=15, shuffle=True)
loader_test = DataLoader(dset_test, batch_size=15, shuffle=True)

size_train = len(loader_train)
size_test = len(loader_test)

# Create NN
net = EuroNN()
loss_fn = nn.CrossEntropyLoss()
optimiser = optim.SGD(net.parameters())

# Train
for i, batch in enumerate(loader_train):
    x, t = batch

    optimiser.zero_grad()

    y = net(x)

    loss = loss_fn(y, t)
    loss.backward()
    optimiser.step()

    if i % 1000 == 0:
        print(f"PROGRESS: {i+1}/{size_train}")

# Test
print("EVALUATING")
correct, total = 0, 0

with torch.no_grad():
    for batch in loader_test:
        x, t = batch

        y = net(x)
        _, preds = torch.max(y, 1)
        correct += (preds == t).sum().item()
        total += preds.shape[0]

print(f"ACCURACY: {correct}/{total}; {correct/total*100.:.2f}%")
