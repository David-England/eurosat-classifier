import torchvision
from torch.utils.data import DataLoader

dset_train = torchvision.datasets.EuroSAT(
    "/eurosatd", download=True, transform=torchvision.transforms.ToTensor()
)

loader_train = DataLoader(dset_train, batch_size=4, shuffle=True)

# Initial data
x, target = next(iter(loader_train))
print(x)
print(target)

print(x.shape)
