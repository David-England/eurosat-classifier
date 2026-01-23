from torch import nn
from torch.nn import functional as f

IMG_WIDTH, IMG_HEIGHT = 64, 64
KERNEL_SIZE = 5


class EuroNN(nn.Module):
    def __init__(self, num_conv_layers=1):
        self.__validate(num_conv_layers)

        super().__init__()

        self.__setup_conv_layers(num_conv_layers)
        self.__linear_input_size = (IMG_WIDTH - num_conv_layers * (KERNEL_SIZE - 1)) * (
            IMG_HEIGHT - num_conv_layers * (KERNEL_SIZE - 1)
        )
        self.linear = nn.Linear(self.__linear_input_size, 10)

    def forward(self, x):
        z = x

        for conv in self.convs:
            a = conv(z)
            z = f.relu(a)

        a2 = self.linear(z.reshape(-1, self.__linear_input_size))
        z2 = f.relu(a2)
        return z2

    def __validate(self, num_conv_layers: int):
        if num_conv_layers < 1:
            raise ValueError(
                f"number of conv. layers must be >= 1, but was {num_conv_layers}"
            )

    def __setup_conv_layers(self, num_conv_layers: int):
        convs = [nn.Conv2d(3, 1, KERNEL_SIZE)] + [
            nn.Conv2d(1, 1, KERNEL_SIZE) for _ in range(num_conv_layers - 1)
        ]
        self.convs = nn.ModuleList(convs)
