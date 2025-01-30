import pandas as pd
from matplotlib import pyplot as plt, ticker

from util.color import generate_diverse_colors


def plot_model_accuracy(
    accuracy_dfs,
    ax=None,
    title="Overall Performance by Model",
    x_column='top_k', 
    x_axis_title=None,
    y_axis_title="Accuracy (%)",
    top_k=None,
):
    accuracy_dfs = accuracy_dfs if isinstance(accuracy_dfs, list) else [accuracy_dfs]
    for df in accuracy_dfs:
        if 'trainer_name' not in df.columns:
            df['trainer_name'] = 'clip'

    df = pd.concat(accuracy_dfs, axis=0, ignore_index=True, sort=False).copy()

    if top_k:
        df = df[df[x_column] <= top_k]

    df['accuracy_percentage'] = df['accuracy'] * 100
    if x_axis_title is None:
        x_axis_title = x_column

    markers = ['o', '*', 's', 'x', '^', 'D']

    unique_trainer_names = df['trainer_name'].unique()
    unique_trainer_names = df['trainer_name'].unique()
    num_groups = len(unique_trainer_names)
    
    colors = generate_diverse_colors(num_groups - 2, color_format='hex')
    colors = ['#4169e1', '#cc0000'] + colors
    color_map = dict(zip(unique_trainer_names, colors))
    
    marker_map = {trainer_name: markers[i % len(markers)] for i, trainer_name in enumerate(unique_trainer_names)}
    
    if 'size' not in df.columns:
        df['size'] = 8

    if 'marker' not in df.columns:
        df['marker'] = df['trainer_name'].map(marker_map)
        
    if 'color' not in df.columns:
        df['color'] = df['trainer_name'].map(color_map)

    if ax is None:
        fig, ax = plt.subplots(figsize=(20, 8))

    unique_x_values = sorted(df[x_column].unique())
    x_ticks = range(len(unique_x_values))
    x_map = {val: i for i, val in enumerate(unique_x_values)}

    for trainer_name in unique_trainer_names:
        trainer_data = df[df['trainer_name'] == trainer_name]
        uniform_x = [x_map[x] for x in trainer_data[x_column]]

        ax.plot(
            uniform_x,
            trainer_data['accuracy_percentage'],
            label=trainer_name,
            color=trainer_data['color'].iloc[0],
            marker=trainer_data['marker'].iloc[0],
            linestyle='-', 
            linewidth=2,
            markersize=trainer_data['size'].iloc[0],
        )
        ax.scatter(
            uniform_x,
            trainer_data['accuracy_percentage'],
            color=trainer_data['color'],
            s=trainer_data['size']*10,
        )
      
        for x, y in zip(uniform_x, trainer_data['accuracy_percentage']):
            ax.annotate(f"{y:.2f}%", (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=12)
  
    df['accuracy_percentage'].min()
    df['accuracy_percentage'].max()

    ax.set_xlabel(x_axis_title, fontsize=14, labelpad=15)
    ax.set_ylabel(y_axis_title, fontsize=14, labelpad=12)
    ax.set_title(title, fontsize=20)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f"{k}" for k in unique_x_values], fontsize=12)
    ax.set_yticklabels([f"{int(x)}%" for x in ax.get_yticks()], fontsize=12)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter())
    ax.grid(color='gray', linestyle='--', linewidth=0.5)
    ax.legend(fontsize=20)
    plt.tight_layout()

    return ax
