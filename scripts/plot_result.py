"""
    Plotting results
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_result(result_fname):

    # Load and preprocess data
    df = pd.read_csv(result_fname)
    df = df[df['success'] == True]
    df['num_agents'] = df['num_agents'].astype(int)
    df['num_samples'] = df['num_samples'].astype(int)

    # Set Seaborn style
    sns.set_theme(style="whitegrid", palette="colorblind")

    # Create plots
    metrics = ['runtime', 'solution_cost', 'solution_timestep']
    map_types = df['map_type'].unique()

    for map_type in map_types:
        map_df = df[df['map_type'] == map_type]

        # Create figure with subplots
        fig, axes = plt.subplots(3, 1, figsize=(9, 12))
        fig.suptitle(f'Performance Analysis: {map_type} Map', fontsize=16, y=1.02)

        for i, metric in enumerate(metrics):
            ax = axes[i]
            sns.lineplot(
                data=map_df,
                x='num_agents',
                y=metric,
                hue='solver',
                style='num_samples',
                markers=True,
                dashes=True,
                ax=ax,
                palette="Set1",
                markersize=8,
                errorbar=None
            )

            # Formatting
            ax.set_xscale('log')
            ax.set_xlabel('Number of Agents (log scale)', fontsize=12)
            ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
            ax.set_title(f'{metric.replace("_", " ").title()} Comparison', fontsize=14)
            ax.legend(title='Solver & Samples', bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)

            # Add value labels for LPSolver points (fewer data points)
            lpsolver_data = map_df[map_df['solver'] == 'LPSolver']
            for _, row in lpsolver_data.iterrows():
                ax.text(row['num_agents'], row[metric], 
                        f"{row[metric]:.1f}",
                        ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        basename = os.path.dirname(result_fname)
        fig_path = os.path.join(basename, f'{map_type}_performance.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.show()