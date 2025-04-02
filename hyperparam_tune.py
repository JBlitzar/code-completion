from train import train_model
import os
from tqdm import tqdm

EXPERIMENT_DIRECTORY = "runs/code-decoder-v22-bigset-tuner"
EPOCHS = 10

hyperparam_sets = [
    {"name": "tiny", "heads": 2, "dim": 128, "layers": 2},
    {"name": "medium", "heads": 4, "dim": 256, "layers": 4},
    {"name": "more_heads", "heads": 8, "dim": 256, "layers": 4},
    {"name": "smalldim", "heads": 4, "dim": 128, "layers": 4},
    {"name": "deep_smalldim", "heads": 4, "dim": 128, "layers": 8},
    {"name": "bigdim", "heads": 4, "dim": 512, "layers": 4},
    {"name": "deeper", "heads": 4, "dim": 256, "layers": 8},
    {"name": "big_deeper", "heads": 4, "dim": 512, "layers": 8},
    {"name": "medium_drop", "heads": 4, "dim": 256, "layers": 4, "drop": 0.3},
    {"name": "bigdim_drop", "heads": 4, "dim": 512, "layers": 4, "drop": 0.3},
]


for config in (pbar := tqdm(hyperparam_sets, dynamic_ncols=True)):
    pbar.set_description(f"Config {config['name']}")

    # The dictionary comprehension is real
    cleaned_config = {k: v for k, v in config.items() if k != "name"}

    train_model(
        os.path.join(EXPERIMENT_DIRECTORY, f"CONFIG_{config['name']}"),
        EPOCHS,
        cleaned_config,
    )

    os.system("bash safe_cleanup.sh")
