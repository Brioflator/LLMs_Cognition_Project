import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings

# ==================================================...
# VISUAL STYLING CONFIGURATION
# ==================================================...

# Default color palette for benchmarks (pastel/light colors)
BENCHMARK_COLORS = {
    'ARC': '#8DD3C7',           # Light teal (reasoning)
    'GSM8K': '#FFFFB3',         # Light yellow (math)
    'HellaSwag': '#BEBADA',     # Light purple (semantic)
    'Lambada': '#FB8072',       # Light coral (prediction)
    'LAMBADA': '#FB8072',       # Alternative capitalization
    'Winogrande': '#80B1D3',    # Light blue (common sense)
    'TruthfulQA': '#FDB462',    # Light orange
    'MMLU': '#B3DE69',          # Light green
    'BoolQ': '#FCCDE5',         # Light pink
}

# Colors for manipulation methods
METHOD_COLORS = {
    'Skip': '#1f77b4',                  # Blue
    'Switch': '#2ca02c',                # Green
    'Middle Repeat': '#ff7f0e',         # Orange
    'Parallel Layer': '#9467bd',        # Purple
    'Looped Parallel 3X': '#8c564b',    # Brown
    'Random Layer Order': '#e377c2',    # Pink
    'Reversed Layer Order': '#7f7f7f',  # Gray
    'Early Exit': '#bcbd22',            # Olive
    'Layer Swap': '#17becf',            # Cyan
}

# Plot styling defaults
STYLE_CONFIG = {
    'title_fontsize': 13,
    'title_fontweight': 'bold',
    'label_fontsize': 11,
    'tick_fontsize': 10,
    'legend_fontsize': 9,
    'line_width': 1.8,
    'median_line_width': 2.8,
    'baseline_line_width': 1.5,
    'benchmark_alpha': 0.7,          # Semi-transparent for benchmark lines
    'grid_alpha': 0.3,
    'grid_color': 'lightgray',
    'spine_color': 'black',
    'spine_width': 0.8,
    'legend_frame_alpha': 0.9,
}

# ==================================================...
# HELPER FUNCTIONS
# ==================================================...

def _setup_style():
    """Configure matplotlib style for consistent visualization."""
    # Try different style names for compatibility
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except OSError:
        try:
            plt.style.use('seaborn-whitegrid')
        except OSError:
            try:
                plt.style.use('ggplot')
            except OSError:
                pass  # Use default style

    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.edgecolor': STYLE_CONFIG['spine_color'],
        'axes.linewidth': STYLE_CONFIG['spine_width'],
        'grid.alpha': STYLE_CONFIG['grid_alpha'],
        'grid.color': STYLE_CONFIG['grid_color'],
        'font.family': 'sans-serif',
    })


def _get_benchmark_color(benchmark_name: str) -> str:
    """Get color for a benchmark, with fallback for unknown benchmarks."""
    if benchmark_name in BENCHMARK_COLORS:
        return BENCHMARK_COLORS[benchmark_name]
    # Generate a consistent color for unknown benchmarks
    hash_val = hash(benchmark_name) % 10
    fallback_colors = plt.cm.tab10.colors
    return fallback_colors[hash_val]


def _get_method_color(method_name: str) -> str:
    """Get color for a method, with fallback for unknown methods."""
    if method_name in METHOD_COLORS:
        return METHOD_COLORS[method_name]
    # Generate a consistent color for unknown methods
    hash_val = hash(method_name) % 10
    fallback_colors = plt.cm.Set2.colors
    return fallback_colors[hash_val]


def _normalize_data(
    data: np.ndarray,
    full_model_value: float,
    random_baseline: float = 0.0
) -> np.ndarray:
    """
    Normalize data to 0-1 scale where:
    - 0.0 = random/worst performance
    - 1.0 = full model performance
    """
    range_val = full_model_value - random_baseline
    if range_val == 0:
        return np.zeros_like(data)
    return (np.array(data) - random_baseline) / range_val


def _add_legend(
    ax: plt.Axes,
    loc: str = 'upper right',
    outside: bool = False,
    **kwargs
):
    """Add a styled legend to the axes."""
    legend_kwargs = {
        'fontsize': STYLE_CONFIG['legend_fontsize'],
        'frameon': True,
        'fancybox': True,
        'framealpha': STYLE_CONFIG['legend_frame_alpha'],
        'edgecolor': 'lightgray',
    }
    legend_kwargs.update(kwargs)

    if outside:
        ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), **legend_kwargs)
    else:
        ax.legend(loc=loc, **legend_kwargs)


def _configure_axes(
    ax: plt.Axes,
    xlabel: str,
    ylabel: str,
    title: str = None,
    xlim: Tuple = None,
    ylim: Tuple = None
):
    """Configure axes with consistent styling."""
    ax.set_xlabel(xlabel, fontsize=STYLE_CONFIG['label_fontsize'])
    ax.set_ylabel(ylabel, fontsize=STYLE_CONFIG['label_fontsize'])

    if title:
        ax.set_title(title, fontsize=STYLE_CONFIG['title_fontsize'],
                     fontweight=STYLE_CONFIG['title_fontweight'], pad=10)

    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)

    ax.tick_params(axis='both', labelsize=STYLE_CONFIG['tick_fontsize'])
    ax.grid(True, alpha=STYLE_CONFIG['grid_alpha'], color=STYLE_CONFIG['grid_color'])

    # Ensure all spines are visible
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(STYLE_CONFIG['spine_color'])
        spine.set_linewidth(STYLE_CONFIG['spine_width'])


# ==================================================...
# PLOT TYPE 1: LAYER-WISE OPERATION IMPACT PLOTS
# ==================================================...

def plot_layer_skip_comparison(
    data_dict: Dict[str, Dict[str, Any]],
    model_sizes: List[str],
    benchmark_name: str = 'LAMBADA',
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (16, 5)
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Create a 3-panel horizontal layout showing Skip vs Switch operations
    across different model sizes.

    Parameters
    ----------
    data_dict : Dict[str, Dict[str, Any]]
        Nested dictionary with structure:
        {
            'llama2-7b': {
                'skip': [accuracy_layer_0, accuracy_layer_1, ...],
                'switch': [accuracy_layer_0, accuracy_layer_1, ...],
                'baseline': float,  # Full model accuracy
                'num_layers': int
            },
            'llama2-13b': {...},
            'llama2-70b': {...}
        }
    model_sizes : List[str]
        List of model size keys to plot (e.g., ['llama2-7b', 'llama2-13b', 'llama2-70b'])
    benchmark_name : str
        Name of the benchmark for y-axis label
    save_path : Optional[str]
        Path to save the figure
    figsize : Tuple[float, float]
        Figure size (width, height)

    Returns
    -------
    Tuple[plt.Figure, np.ndarray]
        Figure and array of axes
    """
    _setup_style()

    n_models = len(model_sizes)
    fig, axes = plt.subplots(1, n_models, figsize=figsize, sharey=True)

    if n_models == 1:
        axes = [axes]

    for idx, model_name in enumerate(model_sizes):
        ax = axes[idx]
        model_data = data_dict.get(model_name, {})

        skip_data = model_data.get('skip', [])
        switch_data = model_data.get('switch', [])
        baseline = model_data.get('baseline', 100)
        num_layers = model_data.get('num_layers', len(skip_data))

        layer_indices = np.arange(num_layers)

        # Plot Skip line
        if len(skip_data) > 0:
            ax.plot(layer_indices, skip_data,
                   color='#1f77b4', linewidth=STYLE_CONFIG['line_width'],
                   label='Skip', marker='o', markersize=3, alpha=0.9)

        # Plot Switch line
        if len(switch_data) > 0:
            ax.plot(layer_indices, switch_data,
                   color='#2ca02c', linewidth=STYLE_CONFIG['line_width'],
                   label='Switch', marker='s', markersize=3, alpha=0.9)

        # Plot baseline
        ax.axhline(y=baseline, color='red', linestyle='--',
                   linewidth=STYLE_CONFIG['baseline_line_width'],
                   label='Full Model Baseline')

        # Format title nicely
        display_name = model_name.replace('-', ' ').replace('_', ' ').title()
        display_name = display_name.replace('Llama2', 'Llama 2')

        _configure_axes(
            ax,
            xlabel='Layer Index',
            ylabel=f'Accuracy (%) on {benchmark_name}' if idx == 0 else '',
            title=display_name,
            xlim=(0, num_layers - 1),
        )

        if idx == 0:
            _add_legend(ax, loc='upper right')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved layer skip comparison plot to: {save_path}")

    return fig, axes


# ==================================================...
# PLOT TYPE 2: MULTI-BENCHMARK DEGRADATION PLOTS
# ==================================================...

def plot_degradation_curve(
    data_dict: Dict[str, List[float]],
    method_name: str,
    model_name: str,
    full_model_values: Dict[str, float] = None,
    random_baseline_values: Dict[str, float] = None,
    normalize: bool = True,
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 7)
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create a single plot showing how all benchmarks degrade as more layers are affected.

    Parameters
    ----------
    data_dict : Dict[str, List[float]]
        Dictionary with benchmark names as keys and performance arrays as values:
        {
            'ARC': [score_0_layers, score_1_layer, ..., score_n_layers],
            'GSM8K': [...],
            'HellaSwag': [...],
            'Lambada': [...],
            'Winogrande': [...]
        }
    method_name : str
        Name of the manipulation method (e.g., 'Skip', 'Middle Repeat')
    model_name : str
        Name of the model (e.g., 'llama2-7b')
    full_model_values : Dict[str, float]
        Full model performance for each benchmark (for normalization)
    random_baseline_values : Dict[str, float]
        Random baseline performance for each benchmark (for normalization)
    normalize : bool
        Whether to normalize values to 0-1 scale
    save_path : Optional[str]
        Path to save the figure
    figsize : Tuple[float, float]
        Figure size

    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        Figure and axes
    """
    _setup_style()

    fig, ax = plt.subplots(figsize=figsize)

    # Default baselines if not provided
    if full_model_values is None:
        full_model_values = {k: max(v) if len(v) > 0 else 1.0
                           for k, v in data_dict.items()}
    if random_baseline_values is None:
        random_baseline_values = {k: 0.0 for k in data_dict.keys()}

    all_normalized_data = []

    # Plot each benchmark
    for benchmark_name, scores in data_dict.items():
        if len(scores) == 0:
            continue

        x_values = np.arange(len(scores))

        if normalize:
            full_val = full_model_values.get(benchmark_name, max(scores))
            random_val = random_baseline_values.get(benchmark_name, 0.0)
            y_values = _normalize_data(scores, full_val, random_val)
        else:
            y_values = np.array(scores)

        all_normalized_data.append(y_values)

        color = _get_benchmark_color(benchmark_name)
        ax.plot(x_values, y_values,
               color=color, linewidth=STYLE_CONFIG['line_width'],
               alpha=STYLE_CONFIG['benchmark_alpha'],
               label=benchmark_name)

    # Calculate and plot median line (bold black)
    if len(all_normalized_data) > 0:
        # Ensure all arrays have same length for median calculation
        min_len = min(len(d) for d in all_normalized_data)
        aligned_data = [d[:min_len] for d in all_normalized_data]
        median_values = np.median(aligned_data, axis=0)
        x_median = np.arange(len(median_values))

        ax.plot(x_median, median_values,
               color='black', linewidth=STYLE_CONFIG['median_line_width'],
               alpha=1.0, label='Median', zorder=10)

    # Plot baselines
    ax.axhline(y=1.0 if normalize else max(full_model_values.values()),
              color='red', linestyle='--',
              linewidth=STYLE_CONFIG['baseline_line_width'],
              label='Full Model', zorder=5)

    ax.axhline(y=0.0 if normalize else 0.0,
              color='gray', linestyle='--',
              linewidth=STYLE_CONFIG['baseline_line_width'] * 0.8,
              alpha=0.7, label='Random Baseline', zorder=5)

    # Configure axes
    ylabel = 'Normalized Benchmark Value' if normalize else 'Benchmark Score'
    _configure_axes(
        ax,
        xlabel='Number of Affected Layers',
        ylabel=ylabel,
        title=f'{method_name}: {model_name}',
        ylim=(-0.05, 1.1) if normalize else None
    )

    # Add legend on the right side
    _add_legend(ax, outside=True)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved degradation curve plot to: {save_path}")

    return fig, ax


# ==================================================...
# PLOT TYPE 3: METHOD COMPARISON PLOTS (2x1 GRID)
# ==================================================...

def plot_method_comparison_grid(
    data_dict_1: Dict[str, List[float]],
    data_dict_2: Dict[str, List[float]],
    method_names: List[str],
    model_name: str,
    full_model_values: Dict[str, float] = None,
    random_baseline_values: Dict[str, float] = None,
    normalize: bool = True,
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 12)
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Create a 2x1 subplot layout comparing two manipulation methods.

    Parameters
    ----------
    data_dict_1 : Dict[str, List[float]]
        Data for first method (same format as plot_degradation_curve)
    data_dict_2 : Dict[str, List[float]]
        Data for second method
    method_names : List[str]
        Names of the two methods (e.g., ['Skip', 'Middle Repeat'])
    model_name : str
        Name of the model
    full_model_values : Dict[str, float]
        Full model performance for normalization
    random_baseline_values : Dict[str, float]
        Random baseline values for normalization
    normalize : bool
        Whether to normalize values
    save_path : Optional[str]
        Path to save the figure
    figsize : Tuple[float, float]
        Figure size

    Returns
    -------
    Tuple[plt.Figure, np.ndarray]
        Figure and array of axes
    """
    _setup_style()

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    data_dicts = [data_dict_1, data_dict_2]

    # Collect all benchmark names for consistent legend
    all_benchmarks = set()
    for dd in data_dicts:
        all_benchmarks.update(dd.keys())

    # Default baselines
    if full_model_values is None:
        full_model_values = {}
        for dd in data_dicts:
            for k, v in dd.items():
                if k not in full_model_values and len(v) > 0:
                    full_model_values[k] = max(v)

    if random_baseline_values is None:
        random_baseline_values = {k: 0.0 for k in all_benchmarks}

    for ax_idx, (ax, data_dict, method_name) in enumerate(zip(axes, data_dicts, method_names)):
        all_normalized_data = []

        # Plot each benchmark
        for benchmark_name, scores in data_dict.items():
            if len(scores) == 0:
                continue

            x_values = np.arange(len(scores))

            if normalize:
                full_val = full_model_values.get(benchmark_name, max(scores))
                random_val = random_baseline_values.get(benchmark_name, 0.0)
                y_values = _normalize_data(scores, full_val, random_val)
            else:
                y_values = np.array(scores)

            all_normalized_data.append(y_values)

            color = _get_benchmark_color(benchmark_name)
            ax.plot(x_values, y_values,
                   color=color, linewidth=STYLE_CONFIG['line_width'],
                   alpha=STYLE_CONFIG['benchmark_alpha'],
                   label=benchmark_name)

        # Calculate and plot median
        if len(all_normalized_data) > 0:
            min_len = min(len(d) for d in all_normalized_data)
            aligned_data = [d[:min_len] for d in all_normalized_data]
            median_values = np.median(aligned_data, axis=0)
            x_median = np.arange(len(median_values))

            ax.plot(x_median, median_values,
                   color='black', linewidth=STYLE_CONFIG['median_line_width'],
                   alpha=1.0, label='Median', zorder=10)

        # Baselines
        ax.axhline(y=1.0 if normalize else max(full_model_values.values()),
                  color='red', linestyle='--',
                  linewidth=STYLE_CONFIG['baseline_line_width'],
                  label='Full Model', zorder=5)

        ax.axhline(y=0.0,
                  color='gray', linestyle='--',
                  linewidth=STYLE_CONFIG['baseline_line_width'] * 0.8,
                  alpha=0.7, label='Random Baseline', zorder=5)

        # Configure
        ylabel = 'Normalized Benchmark Value' if normalize else 'Benchmark Score'
        xlabel = 'Number of Affected Layers' if ax_idx == 1 else ''

        _configure_axes(
            ax,
            xlabel=xlabel,
            ylabel=ylabel,
            title=f'{method_name}: {model_name}',
            ylim=(-0.05, 1.1) if normalize else None
        )

        # Legend on right side
        _add_legend(ax, outside=True)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved method comparison grid to: {save_path}")

    return fig, axes


# ==================================================...
# PLOT TYPE 4: COMPREHENSIVE METHOD COMPARISON
# ==================================================...

def plot_all_methods_comparison(
    methods_dict: Dict[str, List[float]],
    model_name: str,
    full_model_value: float = 1.0,
    random_baseline: float = 0.0,
    normalize: bool = True,
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 8)
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create a single plot showing ALL manipulation methods overlaid.
    Shows which method degrades performance least.

    Parameters
    ----------
    methods_dict : Dict[str, List[float]]
        Dictionary with method names as keys and median performance arrays as values:
        {
            'Skip': [median_0_layers, median_1_layer, ...],
            'Middle Repeat': [...],
            'Parallel Layer': [...],
            'Looped Parallel 3X': [...],
            'Random Layer Order': [...],
            'Reversed Layer Order': [...]
        }
    model_name : str
        Name of the model
    full_model_value : float
        Full model performance value (for normalization)
    random_baseline : float
        Random baseline value (for normalization)
    normalize : bool
        Whether to normalize values
    save_path : Optional[str]
        Path to save the figure
    figsize : Tuple[float, float]
        Figure size

    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        Figure and axes
    """
    _setup_style()

    fig, ax = plt.subplots(figsize=figsize)

    # Plot each method
    for method_name, scores in methods_dict.items():
        if len(scores) == 0:
            continue

        x_values = np.arange(len(scores))

        if normalize:
            y_values = _normalize_data(scores, full_model_value, random_baseline)
        else:
            y_values = np.array(scores)

        color = _get_method_color(method_name)
        ax.plot(x_values, y_values,
               color=color, linewidth=STYLE_CONFIG['line_width'],
               alpha=0.9, label=method_name)

    # Plot baselines
    ax.axhline(y=1.0 if normalize else full_model_value,
              color='red', linestyle='--',
              linewidth=STYLE_CONFIG['baseline_line_width'],
              label='Full Model', zorder=5)

    ax.axhline(y=0.0 if normalize else random_baseline,
              color='gray', linestyle='--',
              linewidth=STYLE_CONFIG['baseline_line_width'] * 0.8,
              alpha=0.7, label='Random Baseline', zorder=5)

    # Configure axes
    ylabel = 'Normalized Benchmark Value' if normalize else 'Benchmark Score'
    _configure_axes(
        ax,
        xlabel='Number of Affected Layers',
        ylabel=ylabel,
        title=f'All Methods: {model_name}',
        ylim=(-0.05, 1.1) if normalize else None
    )

    # Legend on right side
    _add_legend(ax, outside=True)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved all methods comparison to: {save_path}")

    return fig, ax


# ==================================================...
# PLOT TYPE 5: SCALING COMPARISON PLOTS
# ==================================================...

def plot_scaling_comparison(
    data_dict: Dict[str, Dict[str, List[float]]],
    model_sizes: List[str],
    method_name: str,
    full_model_values: Dict[str, Dict[str, float]] = None,
    random_baseline_values: Dict[str, Dict[str, float]] = None,
    normalize: bool = True,
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (18, 5)
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Create 3 horizontal subplots showing how the same operation affects
    different model sizes.

    Parameters
    ----------
    data_dict : Dict[str, Dict[str, List[float]]]
        Nested dictionary:
        {
            'llama2-7b': {
                'ARC': [scores...],
                'GSM8K': [scores...],
                ...
            },
            'llama2-13b': {...},
            'llama2-70b': {...}
        }
    model_sizes : List[str]
        List of model sizes to plot
    method_name : str
        Name of the manipulation method
    full_model_values : Dict[str, Dict[str, float]]
        Full model values per model per benchmark
    random_baseline_values : Dict[str, Dict[str, float]]
        Random baseline values per model per benchmark
    normalize : bool
        Whether to normalize values
    save_path : Optional[str]
        Path to save the figure
    figsize : Tuple[float, float]
        Figure size

    Returns
    -------
    Tuple[plt.Figure, np.ndarray]
        Figure and array of axes
    """
    _setup_style()

    n_models = len(model_sizes)
    fig, axes = plt.subplots(1, n_models, figsize=figsize, sharey=True)

    if n_models == 1:
        axes = [axes]

    for idx, model_name in enumerate(model_sizes):
        ax = axes[idx]
        model_data = data_dict.get(model_name, {})

        # Get baselines for this model
        full_vals = full_model_values.get(model_name, {}) if full_model_values else {}
        random_vals = random_baseline_values.get(model_name, {}) if random_baseline_values else {}

        all_normalized_data = []

        # Plot each benchmark
        for benchmark_name, scores in model_data.items():
            if len(scores) == 0:
                continue

            x_values = np.arange(len(scores))

            if normalize:
                full_val = full_vals.get(benchmark_name, max(scores) if len(scores) > 0 else 1.0)
                random_val = random_vals.get(benchmark_name, 0.0)
                y_values = _normalize_data(scores, full_val, random_val)
            else:
                y_values = np.array(scores)

            all_normalized_data.append(y_values)

            color = _get_benchmark_color(benchmark_name)
            ax.plot(x_values, y_values,
                   color=color, linewidth=STYLE_CONFIG['line_width'],
                   alpha=STYLE_CONFIG['benchmark_alpha'],
                   label=benchmark_name if idx == n_models - 1 else None)

        # Calculate and plot median
        if len(all_normalized_data) > 0:
            min_len = min(len(d) for d in all_normalized_data)
            aligned_data = [d[:min_len] for d in all_normalized_data]
            median_values = np.median(aligned_data, axis=0)
            x_median = np.arange(len(median_values))

            ax.plot(x_median, median_values,
                   color='black', linewidth=STYLE_CONFIG['median_line_width'],
                   alpha=1.0, label='Median' if idx == n_models - 1 else None,
                   zorder=10)

        # Baselines
        ax.axhline(y=1.0 if normalize else 1.0,
                  color='red', linestyle='--',
                  linewidth=STYLE_CONFIG['baseline_line_width'],
                  label='Full Model' if idx == n_models - 1 else None,
                  zorder=5)

        ax.axhline(y=0.0,
                  color='gray', linestyle='--',
                  linewidth=STYLE_CONFIG['baseline_line_width'] * 0.8,
                  alpha=0.7,
                  label='Random Baseline' if idx == n_models - 1 else None,
                  zorder=5)

        # Format title
        display_name = model_name.replace('-', ' ').replace('_', ' ').title()
        display_name = display_name.replace('Llama2', 'Llama 2')

        ylabel = 'Normalized Benchmark Value' if (normalize and idx == 0) else ('' if normalize else 'Benchmark Score')
        _configure_axes(
            ax,
            xlabel='Number of Affected Layers',
            ylabel=ylabel,
            title=f'{method_name}: {display_name}',
            ylim=(-0.05, 1.1) if normalize else None
        )

    # Add legend to the last subplot
    _add_legend(axes[-1], outside=True)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved scaling comparison plot to: {save_path}")

    return fig, axes


# ==================================================...
# CONVENIENCE FUNCTIONS FOR DATA GENERATION (DEMO/TESTING)
# ==================================================...

def generate_demo_skip_switch_data(
    model_sizes: List[str] = ['llama2-7b', 'llama2-13b', 'llama2-70b'],
    num_layers_per_model: List[int] = [32, 40, 80]
) -> Dict[str, Dict[str, Any]]:
    """
    Generate demonstration data for skip/switch comparison plots.

    Returns a dictionary suitable for plot_layer_skip_comparison().
    """
    np.random.seed(42)

    data_dict = {}

    for model, n_layers in zip(model_sizes, num_layers_per_model):
        baseline = 70 + np.random.uniform(-5, 5)

        # Skip operation - typically shows more degradation...
        skip_pattern = np.sin(np.linspace(0, np.pi, n_layers)) * 30 + np.random.normal(0, 3, n_layers)
        skip_data = baseline - skip_pattern
        skip_data = np.clip(skip_data, 20, baseline)

        # Switch operation - typically shows less severe impact
        switch_pattern = np.sin(np.linspace(0, np.pi, n_layers)) * 15 + np.random.normal(0, 2, n_layers)
        switch_data = baseline - switch_pattern
        switch_data = np.clip(switch_data, 40, baseline)

        data_dict[model] = {
            'skip': skip_data.tolist(),
            'switch': switch_data.tolist(),
            'baseline': baseline,
            'num_layers': n_layers
        }

    return data_dict


def generate_demo_degradation_data(
    benchmarks: List[str] = ['ARC', 'GSM8K', 'HellaSwag', 'Lambada', 'Winogrande'],
    max_affected_layers: int = 30
) -> Dict[str, List[float]]:
    """
    Generate demonstration data for degradation curve plots.

    Returns a dictionary suitable for plot_degradation_curve().
    """
    np.random.seed(42)

    data_dict = {}

    for benchmark in benchmarks:
        # Start at full model performance (1.0 normalized) and degrade
        # Different benchmarks degrade at different rates
        decay_rate = np.random.uniform(0.03, 0.08)
        noise = np.random.normal(0, 0.02, max_affected_layers)

        x = np.arange(max_affected_layers)
        y = np.exp(-decay_rate * x) + noise
        y[0] = 1.0  # First point is always full model
        y = np.clip(y, 0, 1)

        data_dict[benchmark] = y.tolist()

    return data_dict


def generate_demo_methods_data(
    methods: List[str] = ['Skip', 'Middle Repeat', 'Parallel Layer',
                          'Looped Parallel 3X', 'Random Layer Order', 'Reversed Layer Order'],
    max_affected_layers: int = 30
) -> Dict[str, List[float]]:
    """
    Generate demonstration data for all-methods comparison plots.

    Returns a dictionary suitable for plot_all_methods_comparison().
    """
    np.random.seed(42)

    data_dict = {}

    # Different methods have different degradation characteristics
    decay_rates = {
        'Skip': 0.05,
        'Middle Repeat': 0.03,
        'Parallel Layer': 0.04,
        'Looped Parallel 3X': 0.02,
        'Random Layer Order': 0.08,
        'Reversed Layer Order': 0.06,
    }

    for method in methods:
        decay_rate = decay_rates.get(method, 0.05)
        noise = np.random.normal(0, 0.015, max_affected_layers)

        x = np.arange(max_affected_layers)
        y = np.exp(-decay_rate * x) + noise
        y[0] = 1.0
        y = np.clip(y, 0, 1)

        data_dict[method] = y.tolist()

    return data_dict