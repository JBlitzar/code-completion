import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

def extract_run_name(filename):
    """Extract the run name from the filename."""
    basename = os.path.basename(filename)
    # Extract the part between '_' and '_tensorboard.csv'
    match = re.search(r'_([^_]+)(?:-loss)?_tensorboard\.csv$', basename)
    if match:
        return match.group(1)
    return basename.split('_')[1].split('-')[0]  # Fallback extraction

def setup_plot_style():
    """Apply publication-quality styling to plots."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'legend.fontsize': 10,
        'figure.dpi': 300,
        'figure.figsize': (10, 6),
        'lines.linewidth': 2.5,
        'axes.grid': True,
        'grid.linestyle': '--',
        'grid.alpha': 0.6,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

def get_metric_label(metric_name):
    """Return a human-readable label for the metric."""
    labels = {
        'loss_epoch': 'Loss',
        'perplexityval_epoch': 'Validation Perplexity',
        'topkacc_epoch': 'Top-K Accuracy',
        'acc_trainstep': 'Training Accuracy'
    }
    return labels.get(metric_name, metric_name.replace('_', ' ').title())

def get_color_mapping(run_names):
    """Create a consistent color mapping for all runs."""
    # Define a color palette with distinct colors
    # colors = [
    #     '#1f77b4',  # Blue
    #     '#ff7f0e',  # Orange
    #     '#2ca02c',  # Green
    #     '#d62728',  # Red
    #     '#9467bd',  # Purple
    #     '#8c564b',  # Brown
    #     '#e377c2',  # Pink
    #     '#7f7f7f',  # Gray
    #     '#bcbd22',  # Yellow-green
    #     '#17becf',  # Cyan
    # ]
#     colors = """#091717

# #13B3B9

# #265E5A

# #20808D

# #25E5A5

# #20808D

# #FBFAF4

# #E4E3D4

# #FFD2A6

# #A84B2F

# #944454""".lower().split("\n\n")
    colors = [
        "#e6194b",  # Red
        "#f58231",  # Orange
        "#ffe119",  # Yellow
        "#bfef45",  # Lime
        "#3cb44b",  # Green
        "#42d4f4",  # Cyan
        "#4363d8",  # Blue
        "#911eb4",  # Purple
        "#f032e6",  # Magenta
        "#a9a9a9"   # Grey
    ]
    
    # Create a mapping of run names to colors
    return {name: colors[i % len(colors)] for i, name in enumerate(sorted(run_names))}

def plot_metric(metric_dir, color_mapping, output_dir):
    """Plot all runs for a specific metric."""
    metric_name = os.path.basename(metric_dir)
    csv_files = glob.glob(os.path.join(metric_dir, '*.csv'))
    
    if not csv_files:
        print(f"No CSV files found in {metric_dir}")
        return
    
    plt.figure(figsize=(12, 7))
    
    for csv_file in sorted(csv_files):
        try:
            # Read the CSV file
            df = pd.read_csv(csv_file)
            
            # Extract run name from filename
            run_name = extract_run_name(csv_file)
            
            # Plot the data using step as x-axis
            color = color_mapping.get(run_name, 'gray')
            plt.plot(df['Step'], df['Value'], label=run_name, color=color, alpha=0.9)
            #plt.plot(df['Step'], df['Value'], label=run_name, color=color, marker='o', markersize=6, alpha=0.8)
            
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
    
    # Set labels and title
    plt.xlabel('Step')
    plt.ylabel(get_metric_label(metric_name))

    comparison = "Epoch" if "epoch" in metric_name else "Step"
    plt.title(f'{get_metric_label(metric_name)} vs. {comparison}', fontweight='bold')
    
    # Add legend with good positioning
    plt.legend(loc='best', frameon=True, fancybox=True, framealpha=0.9, 
               shadow=True, borderpad=1, ncol=2 if len(csv_files) > 5 else 1)
    
    # Add grid for better readability
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Tight layout for clean margins
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{metric_name}_plot.png')
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Saved plot to {output_path}")
    
    # Close the figure to free memory
    plt.close()

def main():
    # Base directory containing the metric directories
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs_jsons')
    
    # Output directory for plots
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup plot style
    setup_plot_style()
    
    # Get all metric directories
    metric_dirs = [d for d in glob.glob(os.path.join(base_dir, '*')) if os.path.isdir(d)]
    
    # Collect all run names across all metrics for consistent coloring
    all_run_names = set()
    for metric_dir in metric_dirs:
        csv_files = glob.glob(os.path.join(metric_dir, '*.csv'))
        for csv_file in csv_files:
            run_name = extract_run_name(csv_file)
            all_run_names.add(run_name)
    
    # Create color mapping
    color_mapping = get_color_mapping(all_run_names)
    
    # Plot each metric
    for metric_dir in metric_dirs:
        plot_metric(metric_dir, color_mapping, output_dir)
    
    print(f"All plots have been generated in {output_dir}")

if __name__ == '__main__':
    main()
