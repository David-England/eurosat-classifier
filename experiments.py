import json
from run import run

with open("default_settings.json", "r") as f:
    defaults = json.load(f)

    for batch_size in [10, 12, 15, 16, 20, 24, 32]:
        print(f"\n>> BATCH SIZE {batch_size}")
        settings = defaults.copy()
        settings["batch_size"] = batch_size
        run(settings)
