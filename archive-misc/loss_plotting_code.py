import json
import argparse
import matplotlib.pyplot as plt
import os

def plot_loss_curves(json_path, output_path=None):
    """
    Reads a JSON file containing 'loss' and 'val_loss' lists,
    then plots them over epochs with publication‐quality styling.
    """
    # Load data
    with open(json_path, 'r') as f:
        s = f.read()
        s = "{" + "\"values\":" + s + "}"
        data = json.loads(s)["values"]



    # Apply publication‐style settings
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 12,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'legend.fontsize': 12,
        'figure.dpi': 300,
        'figure.figsize': (6, 4),
        'lines.linewidth': 2,
        'axes.grid': True,
        'grid.linestyle': '--',
        'grid.alpha': 0.5,
    })

    # Create plot
    epochs = [i[1] for i in data]
    values = [i[2] for i in data]
    fig, ax = plt.subplots()
    basename = os.basename(json_path)
    parts = basename.replace(".json", "").replace("_tensorboard", "").split("_")[-1].split("-")
    metric = parts[-1]
    experiment = "".join(parts[:-1])
    ax.plot(epochs, values,   label=json_path.spli,   marker='o')

    # Labels & title
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training vs. {metric} Loss over Epochs')

    # Legend
    ax.legend(loc='upper right')

    # Tight layout for clean margins
    fig.tight_layout()

    # Save or show
    if output_path:
        fig.savefig(output_path, format=output_path.split('.')[-1], bbox_inches='tight')
    else:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Plot training & validation loss curves from a JSON file."
    )
    parser.add_argument(
        '-i', '--input', required=True,
        help="Path to the JSON file (must contain 'loss' and 'val_loss' arrays)."
    )
    parser.add_argument(
        '-o', '--output', default=None,
        help="Path to save the figure (e.g. figure.pdf, figure.png). If omitted, displays interactively."
    )
    args = parser.parse_args()

    plot_loss_curves(args.input, args.output)
