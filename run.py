import nets
import torch
import torchvision
from torch import nn, optim
from torch.utils.data import DataLoader, random_split


def run(settings: dict):
    torch.manual_seed(4)

    dvc = get_device()

    dset = torchvision.datasets.EuroSAT(
        "/eurosatd", download=True, transform=torchvision.transforms.ToTensor()
    )
    dset_train, dset_test = random_split(dset, [0.8, 0.2])

    loader_train = DataLoader(
        dset_train, batch_size=settings["batch_size"], shuffle=True
    )
    loader_test = DataLoader(dset_test, batch_size=settings["batch_size"], shuffle=True)

    net = nets.EuroNN(settings).to(dvc)
    print(net)

    train(
        net,
        nn.CrossEntropyLoss(),
        optim.SGD(net.parameters()),
        loader_train,
        settings["epochs"],
        dvc,
    )
    test(net, loader_test, dvc)


def train(
    net: nets.EuroNN,
    loss_fn: nn.Module,
    optimiser: optim.Optimizer,
    loader: DataLoader,
    epochs: int,
    dvc: torch.device,
):
    length = len(loader)

    for epoch in range(epochs):
        for i, batch in enumerate(loader):
            x, t = [g.to(dvc) for g in batch]

            optimiser.zero_grad()

            y = net(x)

            loss = loss_fn(y, t)
            loss.backward()
            optimiser.step()

            batch_no = i + epoch * length

            if batch_no % 1000 == 0:
                print(f"PROGRESS: {batch_no + 1}/{epochs * length}")


def test(net: nets.EuroNN, loader: DataLoader, dvc: torch.device):
    print("EVALUATING")
    correct, total = 0, 0

    with torch.no_grad():
        for batch in loader:
            x, t = [g.to(dvc) for g in batch]

            y = net(x)
            _, preds = torch.max(y, 1)
            correct += (preds == t).sum().item()
            total += preds.shape[0]

    print(f"ACCURACY: {correct}/{total}; {correct/total*100.:.2f}%")


def get_device():
    if torch.accelerator.is_available():
        return torch.accelerator.current_accelerator()
    else:
        return torch.device("cpu")
