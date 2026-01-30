from torch import nn
from torch.nn import functional as f

IMG_WIDTH, IMG_HEIGHT = 64, 64


class EuroNN(nn.Module):
    def __init__(self, settings: dict):
        self.__validate(settings)

        super().__init__()

        self.__setup_conv_layers(
            settings["num_conv_layers"],
            settings["num_conv_features"],
            settings["kernel_size"],
        )
        self.__linear_input_size = (
            (IMG_WIDTH - settings["num_conv_layers"] * (settings["kernel_size"] - 1))
            * (IMG_HEIGHT - settings["num_conv_layers"] * (settings["kernel_size"] - 1))
            * settings["num_conv_features"]
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

    def __validate(self, settings: dict):
        # Validate positive integer settings:
        for k in ["num_conv_layers", "num_conv_features", "kernel_size"]:
            if k not in settings:
                raise ValueError(f"no setting {k}")
            if not isinstance(settings[k], int) or settings[k] < 1:
                raise ValueError(
                    f"setting {k} must be an integer >= 1, but was {settings[k]}"
                )

    def __setup_conv_layers(
        self, num_conv_layers: int, num_conv_features: int, kernel_size: int
    ):
        convs = [nn.Conv2d(3, num_conv_features, kernel_size)] + [
            nn.Conv2d(num_conv_features, num_conv_features, kernel_size)
            for _ in range(num_conv_layers - 1)
        ]
        self.convs = nn.ModuleList(convs)
