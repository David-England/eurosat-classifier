import json
from run import run

with open("default_settings.json", "r") as f:
    defaults = json.load(f)

    for batch_size in [10, 12, 15, 16, 20, 24, 32]:
        print(f"\n>> BATCH SIZE {batch_size}")
        settings = defaults.copy()
        settings["batch_size"] = batch_size
        run(settings)

    for epochs in [1, 2, 4, 5, 8, 16]:
        print(f"\n>> EPOCHS {epochs}")
        settings = defaults.copy()
        settings["epochs"] = epochs
        run(settings)

    for kernel_size in [3, 4, 5, 8, 16]:
        print(f"\n>> KERNEL SIZE {kernel_size}")
        settings = defaults.copy()
        settings["kernel_size"] = kernel_size
        run(settings)

    for num_conv_features in [1, 3, 5, 8, 16]:
        print(f"\n>> NUM. CONV FEATURES {num_conv_features}")
        settings = defaults.copy()
        settings["num_conv_features"] = num_conv_features
        run(settings)

    for num_conv_layers in [1, 2, 3, 5]:
        print(f"\n>> NUM. CONV LAYERS {num_conv_layers}")
        settings = defaults.copy()
        settings["num_conv_layers"] = num_conv_layers
        run(settings)
