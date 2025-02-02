import json
import os

data_dir = "data"

rename_list = []
delte_list = [
    "true_K",
    "true_m",
    "true_Sigma",
]

for root, dirs, files in os.walk(data_dir):
    for file in files:
        if file == "config.json":
            file_path = os.path.join(root, file)
            with open(file_path, "r") as f:
                config = json.load(f)
            print(file_path)
            print(config)

            for old_name, new_name in rename_list:
                if old_name in config:
                    config[new_name] = config.pop(old_name)
            for name in delte_list:
                if name in config:
                    del config[name]

            # Modify the config as needed
            print(config)

            with open(file_path, "w") as f:
                json.dump(config, f, indent=4)
