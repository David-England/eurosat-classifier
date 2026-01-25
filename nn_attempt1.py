import nets
import torch
import torchvision
from torch import nn, optim
from torch.utils.data import DataLoader, random_split


def train(
    net: nets.EuroNN, loss_fn: nn.Module, optimiser: optim.Optimizer, loader: DataLoader
):
    for i, batch in enumerate(loader):
        x, t = batch

        optimiser.zero_grad()

        y = net(x)

        loss = loss_fn(y, t)
        loss.backward()
        optimiser.step()

        if i % 1000 == 0:
            print(f"PROGRESS: {i+1}/{size_train}")


def test(net: nets.EuroNN, loader: DataLoader):
    print("EVALUATING")
    correct, total = 0, 0

    with torch.no_grad():
        for batch in loader:
            x, t = batch

            y = net(x)
            _, preds = torch.max(y, 1)
            correct += (preds == t).sum().item()
            total += preds.shape[0]

    print(f"ACCURACY: {correct}/{total}; {correct/total*100.:.2f}%")


torch.manual_seed(4)

dset = torchvision.datasets.EuroSAT(
    "/eurosatd", download=True, transform=torchvision.transforms.ToTensor()
)
dset_train, dset_test = random_split(dset, [0.8, 0.2])

loader_train = DataLoader(dset_train, batch_size=15, shuffle=True)
loader_test = DataLoader(dset_test, batch_size=15, shuffle=True)

size_train = len(loader_train)
size_test = len(loader_test)

net = nets.EuroNN()
print(net)

train(net, nn.CrossEntropyLoss(), optim.SGD(net.parameters()), loader_train)
test(net, loader_test)
