## imports =========================================================
import os
import sys
import time
import json
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm
import dill
from scipy.stats import kurtosis, skew
from axon_velocity.axon_velocity import GraphAxonTracking, get_default_graph_velocity_params
import matplotlib.cm as cm
from typing import Tuple, List, Dict

# plotting functions ==================================================
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def plot_all_reconstructions(gtr_list, unit_ids=None, colors=None, locations=None,
                              ax=None, fig=None, show_unit_ids=False, cmap_name="tab10",
                              alpha=0.8, linewidth=1.5, figsize=(10, 10), marker_size=4):
    """
    Plots all cleaned axon reconstructions on a shared figure, with fixed chip dimensions and unit color legend.

    Args:
        gtr_list (list): List of GraphAxonTracking objects.
        unit_ids (list): Optional list of unit IDs.
        colors (list or str): Optional list of colors or colormap name.
        locations (np.ndarray): Optional electrode layout for background.
        ax (matplotlib Axes): Optional axis to plot into.
        fig (matplotlib Figure): Optional figure.
        show_unit_ids (bool): Annotate each neuron.
        cmap_name (str): Name of the colormap to use.
        alpha (float): Transparency of traces.
        linewidth (float): Line width for branches.
        figsize (tuple): Size of the figure.
        marker_size (int): Size of branch point markers.

    Returns:
        fig, ax: Matplotlib figure and axis with the combined plot.
    """
    if ax is None or fig is None:
        fig, ax = plt.subplots(figsize=figsize)

    num_units = len(gtr_list)
    cmap = cm.get_cmap(cmap_name, num_units) if isinstance(colors, str) or colors is None else None

    legend_entries = []

    for i, gtr in enumerate(gtr_list):
        unit_color = colors[i] if isinstance(colors, list) else cmap(i)
        for branch in gtr.branches:
            ch_locs = gtr.locations[branch['channels']]
            ax.plot(ch_locs[:, 0], ch_locs[:, 1], '-', color=unit_color,
                    linewidth=linewidth, alpha=alpha)
            ax.plot(ch_locs[:, 0], ch_locs[:, 1], 'o', color=unit_color,
                    markersize=marker_size, alpha=alpha)

        if show_unit_ids and unit_ids:
            init_loc = gtr.locations[gtr.init_channel]
            ax.text(init_loc[0], init_loc[1], str(unit_ids[i]),
                    fontsize=10, ha='center', va='center', color=unit_color,
                    weight='bold', alpha=0.9)

        if unit_ids:
            legend_entries.append(plt.Line2D([0], [0], color=unit_color, lw=2, label=f"Unit {unit_ids[i]}"))

    # Optional: background probe layout
    if locations is not None:
        ax.plot(locations[:, 0], locations[:, 1], '.', color='lightgray', alpha=0.5, zorder=0)

    # Set fixed chip area box
    ax.set_xlim(0, 2000)
    ax.set_ylim(0, 4000)
    ax.set_aspect('equal')
    ax.axis('off')

    # Legend with unit color mapping
    if legend_entries:
        ax.legend(handles=legend_entries, fontsize=10, loc='lower left', frameon=False)

    return fig, ax

def plot_all_reconstructions_dep1(gtr_list, unit_ids=None, colors=None, locations=None,
                              ax=None, fig=None, title="All Reconstructed Axons",
                              show_unit_ids=False, cmap_name="tab20", alpha=0.8, linewidth=1.5,
                              figsize=(10, 10), marker_size=4):
    """
    Plots all cleaned axon reconstructions on a shared figure.

    Args:
        gtr_list (list): List of GraphAxonTracking objects.
        unit_ids (list): Optional list of unit IDs.
        colors (list or str): Optional list of colors or colormap name.
        locations (np.ndarray): Optional electrode layout for background.
        ax (matplotlib Axes): Optional axis to plot into.
        fig (matplotlib Figure): Optional figure.
        title (str): Title of the figure.
        show_unit_ids (bool): Annotate each neuron.
        cmap_name (str): Name of the colormap to use.
        alpha (float): Transparency of traces.
        linewidth (float): Line width for branches.
        figsize (tuple): Size of the figure.
        marker_size (int): Size of branch point markers.

    Returns:
        fig, ax: Matplotlib figure and axis with the combined plot.
    """
    if ax is None or fig is None:
        fig, ax = plt.subplots(figsize=figsize)

    num_units = len(gtr_list)
    cmap = cm.get_cmap(cmap_name, num_units) if isinstance(colors, str) or colors is None else None

    for i, gtr in enumerate(gtr_list):
        unit_color = colors[i] if isinstance(colors, list) else cmap(i)
        for branch in gtr.branches:
            ch_locs = gtr.locations[branch['channels']]
            ax.plot(ch_locs[:, 0], ch_locs[:, 1], '-', color=unit_color,
                    linewidth=linewidth, alpha=alpha)
            ax.plot(ch_locs[:, 0], ch_locs[:, 1], 'o', color=unit_color,
                    markersize=marker_size, alpha=alpha)

        if show_unit_ids and unit_ids:
            init_loc = gtr.locations[gtr.init_channel]
            ax.text(init_loc[0], init_loc[1], str(unit_ids[i]),
                    fontsize=10, ha='center', va='center', color=unit_color,
                    weight='bold', alpha=0.9)

    # Optional: background probe layout
    if locations is not None:
        ax.plot(locations[:, 0], locations[:, 1], '.', color='lightgray', alpha=0.5, zorder=0)

    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=18)

    return fig, ax

def get_branch_colors_from_gtr(gtr, cmap="rainbow"):
    """Returns a list of branch colors matching the GTR internal color logic."""
    cm_map = plt.get_cmap(cmap)
    raw_paths = gtr._paths_raw
    color_lookup = {i: cm_map(i / len(raw_paths)) for i in range(len(raw_paths))}
    return [color_lookup[branch['raw_path_idx']] for branch in gtr.branches]

def plot_template_propagation(template, locations, selected_channels, sort_templates=False,
                              color='C0', color_marker='r', fig=None, ax=None,
                              line_thickness=1.0, gtr=None, label_branches=False,
                              label_channels=False, font_size=20, gain=1.0,
                              marker_size=8, marker='o', markerfacecolor='none',
                              add_scale_bar=True, sampling_frequency=10000, cmap="rainbow"):
    """
    Draws a propagation-style plot of the waveform template, per-channel, colored by branch.

    Args:
        template (np.ndarray): Waveform array (channels x time).
        locations (list): List of (x, y) coordinates per channel.
        selected_channels (list): Channels to highlight.
        sort_templates (bool): Whether to sort by peak timing.
        color, color_marker (str): Default color options.
        fig, ax: Matplotlib objects to draw into.
        gtr (GraphAxonTracking): Required. Provides branch mapping.
        label_branches, label_channels (bool): Toggle annotations.
        font_size (int): Font size for labels.
        gain (float): Vertical scale multiplier.
        marker_size (int): Marker size for peak points.
        add_scale_bar (bool): Draw scale bar if True.
        sampling_frequency (int): Used to calculate ms scale.

    Returns:
        ax: Matplotlib axis with the propagation plot.
    """
    if gtr is None:
        raise ValueError("GraphAxonTracking object (gtr) is required.")
    
    if ax is None:
        fig, ax = plt.subplots()

    # Flatten branch->channels map to channel->branch_id
    channel_to_branch = {}
    for i, branch in enumerate(gtr.branches):
        for ch in branch['channels']:
            channel_to_branch[ch] = i

    # Filter selected channels: only those that are in a branch
    filtered_channels = [ch for ch in selected_channels if ch in channel_to_branch]
    if not filtered_channels:
        print("No selected channels found in any branch. Nothing to plot.")
        return ax

    # Prepare plotting arrays
    template_selected = template[filtered_channels]
    locs = np.array(locations)[filtered_channels]
    branches = [channel_to_branch[ch] for ch in filtered_channels]

    # Sort channels to group by branch, then peak time within branch
    idx_grouped = sorted(range(len(branches)), key=lambda i: (branches[i], np.argmin(template_selected[i])))
    template_selected = template_selected[idx_grouped]
    locs = locs[idx_grouped]
    branches = [branches[i] for i in idx_grouped]
    filtered_channels = [filtered_channels[i] for i in idx_grouped]

    ptp_glob = np.max(np.ptp(template_selected, axis=1))

    # Get branch colors that match GTR logic
    branch_colors_list = get_branch_colors_from_gtr(gtr, cmap=cmap)

    # Plot each channel waveform
    for i, temp in enumerate(template_selected):
        ch_id = filtered_channels[i]
        branch_id = branches[i]
        color = branch_colors_list[branch_id]

        temp_scaled = temp * gain
        temp_shifted = temp_scaled + i * 1.5 * ptp_glob

        ax.plot(temp_shifted, color=color, linewidth=line_thickness)

        # Peak marker
        min_idx = np.argmin(temp_shifted)
        min_val = temp_shifted[min_idx]
        ax.plot(min_idx, min_val, marker=marker, color=color,
                markersize=marker_size, markerfacecolor=markerfacecolor)

        # Channel label
        if label_channels:
            try:
                max_idx = np.argmax(temp_shifted[:min_idx])
                y_label = np.mean(temp_shifted[:max_idx])
                if np.isnan(y_label):
                    raise ValueError
            except:
                y_label = temp_shifted[0]

            ax.text(-0.1, y_label + 1, f"Ch {ch_id}", fontsize=font_size,
                    ha='left', va='bottom')

    # Branch legend
    if label_branches:
        handles = [
            plt.Line2D([0], [0], marker='o', color='w',
                       label=f'Branch {i}',
                       markerfacecolor=branch_colors_list[i],
                       markeredgecolor=branch_colors_list[i],
                       markersize=10)
            for i in range(len(gtr.branches))
        ]
        ax.legend(handles=handles, loc='upper right', fontsize=font_size)

    # Optional scale bar
    if add_scale_bar:
        y_scale = np.round(0.5 * ptp_glob / 10) * 10
        y_scaled = y_scale * gain
        x_samples = int((2 / 1000) * sampling_frequency)
        x_ms = x_samples / sampling_frequency * 1000

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        bar_x = xlim[1] - 0.05 * (xlim[1] - xlim[0]) - x_samples
        bar_y = ylim[0] + 0.02 * (ylim[1] - ylim[0])  # Move the bar closer to the x-axis label

        ax.plot([bar_x, bar_x + x_samples], [bar_y, bar_y], color='k', lw=2)
        ax.plot([bar_x, bar_x], [bar_y, bar_y + y_scaled], color='k', lw=2)
        ax.text(bar_x + x_samples / 2, bar_y - 0.02 * (ylim[1] - ylim[0]),  # Adjust x label position
                f'{x_ms:.1f} ms', ha='center', va='top', fontsize=font_size)
        ax.text(bar_x - 0.05 * (xlim[1] - xlim[0]), bar_y + y_scaled / 2,
                f'{y_scale:.1f} µV', ha='right', va='center', fontsize=font_size)

    ax.axis('off')
    return ax

def plot_template_propagation_dep3_sortedbybranch(template, locations, selected_channels, sort_templates=False,
                              color='C0', color_marker='r', fig=None, ax=None,
                              line_thickness=1.0, gtr=None, label_branches=False,
                              label_channels=False, font_size=16, gain=1.0,
                              marker_size=8, marker='o', markerfacecolor='none',
                              add_scale_bar=True, sampling_frequency=10000):
    """
    Draws a propagation-style plot of the waveform template, per-channel, grouped by branch.

    Traces are color-coded by branch, and branches are plotted contiguously.
    """
    import matplotlib.cm as cm

    if gtr is None:
        raise ValueError("GraphAxonTracking object (gtr) is required.")
    if ax is None:
        fig, ax = plt.subplots()

    # Map channel to branch
    channel_to_branch = {
        ch: i for i, branch in enumerate(gtr.branches) for ch in branch["channels"]
    }

    # Keep only channels that belong to a branch
    filtered_channels = [ch for ch in selected_channels if ch in channel_to_branch]
    if not filtered_channels:
        print("No selected channels found in any branch. Nothing to plot.")
        return ax

    # Prepare lookup arrays
    traces = []
    cmap = cm.get_cmap("tab20", len(gtr.branches))

    for branch_id in sorted(set(channel_to_branch[ch] for ch in filtered_channels)):
        branch_chs = [ch for ch in filtered_channels if channel_to_branch[ch] == branch_id]
        branch_color = cmap(branch_id)

        # Optional sorting within each branch
        if sort_templates:
            peaks = [np.argmin(template[ch]) for ch in branch_chs]
            branch_chs = [ch for _, ch in sorted(zip(peaks, branch_chs))]

        for ch in branch_chs:
            traces.append({
                "channel": ch,
                "branch": branch_id,
                "waveform": template[ch],
                "location": locations[ch],
                "color": branch_color,
            })

    # Global scaling
    ptps = [np.ptp(t["waveform"]) for t in traces]
    ptp_glob = np.max(ptps)

    # Plot trace-by-trace with branch grouping
    for i, t in enumerate(traces):
        y_offset = i * 1.5 * ptp_glob
        scaled = t["waveform"] * gain + y_offset

        ax.plot(scaled, color=t["color"], linewidth=line_thickness)

        # Min marker
        min_idx = np.argmin(scaled)
        min_val = scaled[min_idx]
        ax.plot(min_idx, min_val, marker=marker, color=t["color"],
                markersize=marker_size, markerfacecolor=markerfacecolor)

        # Channel label
        if label_channels:
            try:
                max_idx = np.argmax(scaled[:min_idx])
                y_label = np.mean(scaled[:max_idx])
                if np.isnan(y_label):
                    raise ValueError
            except:
                y_label = scaled[0]

            ax.text(-0.1, y_label + 1, f"Ch {t['channel']}", fontsize=font_size,
                    ha='left', va='bottom')

    # Branch legend
    if label_branches:
        handles = [
            plt.Line2D([0], [0], marker='o', color='w',
                       label=f'Branch {i}',
                       markerfacecolor=cmap(i),
                       markeredgecolor=cmap(i),
                       markersize=marker_size + 2)
            for i in sorted(set(t["branch"] for t in traces))
        ]
        ax.legend(handles=handles, loc='upper right', fontsize=font_size)

    # Optional scale bar
    if add_scale_bar:
        y_scale = np.round(0.2 * ptp_glob / 10) * 10
        y_scaled = y_scale * gain
        x_samples = int((2 / 1000) * sampling_frequency)
        x_ms = x_samples / sampling_frequency * 1000

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        bar_x = xlim[1] - 0.05 * (xlim[1] - xlim[0]) - x_samples
        bar_y = ylim[0] + 0.05 * (ylim[1] - ylim[0])

        ax.plot([bar_x, bar_x + x_samples], [bar_y, bar_y], color='k', lw=2)
        ax.plot([bar_x, bar_x], [bar_y, bar_y + y_scaled], color='k', lw=2)
        ax.text(bar_x + x_samples / 2, bar_y - 0.05 * (ylim[1] - ylim[0]),
                f'{x_ms:.1f} ms', ha='center', va='top', fontsize=font_size)
        ax.text(bar_x - 0.05 * (xlim[1] - xlim[0]), bar_y + y_scaled / 2,
                f'{y_scale:.1f} µV', ha='right', va='center', fontsize=font_size)

    ax.axis("off")
    return ax

def plot_template_propagation_dep2_ungrouped(template, locations, selected_channels, sort_templates=False,
                              color='C0', color_marker='r', fig=None, ax=None,
                              line_thickness=1.0, gtr=None, label_branches=False,
                              label_channels=False, font_size=12, gain=1.0,
                              marker_size=5, marker='o', markerfacecolor='none',
                              add_scale_bar=True, sampling_frequency=10000):
    """
    Draws a propagation-style plot of the waveform template, per-channel, colored by branch.

    Args:
        template (np.ndarray): Waveform array (channels x time).
        locations (list): List of (x, y) coordinates per channel.
        selected_channels (list): Channels to highlight.
        sort_templates (bool): Whether to sort by peak timing.
        color, color_marker (str): Default color options.
        fig, ax: Matplotlib objects to draw into.
        gtr (GraphAxonTracking): Required. Provides branch mapping.
        label_branches, label_channels (bool): Toggle annotations.
        font_size (int): Font size for labels.
        gain (float): Vertical scale multiplier.
        marker_size (int): Marker size for peak points.
        add_scale_bar (bool): Draw scale bar if True.
        sampling_frequency (int): Used to calculate ms scale.

    Returns:
        ax: Matplotlib axis with the propagation plot.
    """
    import matplotlib.cm as cm

    if gtr is None:
        raise ValueError("GraphAxonTracking object (gtr) is required.")
    
    if ax is None:
        fig, ax = plt.subplots()

    # Flatten branch->channels map to channel->branch_id
    channel_to_branch = {}
    for i, branch in enumerate(gtr.branches):
        for ch in branch['channels']:
            channel_to_branch[ch] = i

    # Filter selected channels: only those that are in a branch
    filtered_channels = [ch for ch in selected_channels if ch in channel_to_branch]
    if not filtered_channels:
        print("No selected channels found in any branch. Nothing to plot.")
        return ax

    # Prepare plotting arrays
    template_selected = template[filtered_channels]
    locs = np.array(locations)[filtered_channels]
    branches = [channel_to_branch[ch] for ch in filtered_channels]

    # Optional sorting by peak timing
    if sort_templates:
        peaks = np.argmin(template_selected, axis=1)
        sorted_idxs = np.argsort(peaks)
        template_selected = template_selected[sorted_idxs]
        locs = locs[sorted_idxs]
        branches = [branches[i] for i in sorted_idxs]
        filtered_channels = [filtered_channels[i] for i in sorted_idxs]

    ptp_glob = np.max(np.ptp(template_selected, axis=1))
    branch_cmap = cm.get_cmap("tab20", len(gtr.branches))

    # Plot each channel waveform
    for i, temp in enumerate(template_selected):
        ch_id = filtered_channels[i]
        branch_id = branches[i]
        color = branch_cmap(branch_id)

        temp_scaled = temp * gain
        temp_shifted = temp_scaled + i * 1.5 * ptp_glob

        ax.plot(temp_shifted, color=color, linewidth=line_thickness)

        # Peak marker
        min_idx = np.argmin(temp_shifted)
        min_val = temp_shifted[min_idx]
        ax.plot(min_idx, min_val, marker=marker, color=color,
                markersize=marker_size, markerfacecolor=markerfacecolor)

        # Channel label
        if label_channels:
            try:
                max_idx = np.argmax(temp_shifted[:min_idx])
                y_label = np.mean(temp_shifted[:max_idx])
                if np.isnan(y_label):
                    raise ValueError
            except:
                y_label = temp_shifted[0]

            ax.text(-0.1, y_label + 1, f"Ch {ch_id}", fontsize=font_size,
                    ha='left', va='bottom')

    # Branch legend
    if label_branches:
        handles = [
            plt.Line2D([0], [0], marker='o', color='w',
                       label=f'Branch {i}',
                       markerfacecolor=branch_cmap(i),
                       markeredgecolor=branch_cmap(i),
                       markersize=10)
            for i in range(len(gtr.branches))
        ]
        ax.legend(handles=handles, loc='upper right', fontsize=font_size)

    # Optional scale bar
    if add_scale_bar:
        y_scale = np.round(0.2 * ptp_glob / 10) * 10
        y_scaled = y_scale * gain
        x_samples = int((2 / 1000) * sampling_frequency)
        x_ms = x_samples / sampling_frequency * 1000

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        bar_x = xlim[1] - 0.05 * (xlim[1] - xlim[0]) - x_samples
        bar_y = ylim[0] + 0.05 * (ylim[1] - ylim[0])

        ax.plot([bar_x, bar_x + x_samples], [bar_y, bar_y], color='k', lw=2)
        ax.plot([bar_x, bar_x], [bar_y, bar_y + y_scaled], color='k', lw=2)
        ax.text(bar_x + x_samples / 2, bar_y - 0.05 * (ylim[1] - ylim[0]),
                f'{x_ms:.1f} ms', ha='center', va='top', fontsize=font_size)
        ax.text(bar_x - 0.05 * (xlim[1] - xlim[0]), bar_y + y_scaled / 2,
                f'{y_scale:.1f} µV', ha='right', va='center', fontsize=font_size)

    ax.axis('off')
    return ax

def plot_template_propagation_dep(template, locations, selected_channels, sort_templates=False,
                              color='C0', color_marker='r', fig=None, ax=None,
                              line_thickness=1.0, gtr=None, label_branches=False,
                              label_channels=False, font_size=12, gain=1.0,
                              marker_size=5, marker='o', markerfacecolor='none',
                              add_scale_bar=True, sampling_frequency=10000):
    """
    Draws a propagation-style plot of the waveform template with per-channel alignment.

    Args:
        template (np.ndarray): Waveform array (channels x time).
        locations (list): List of (x, y) coordinates per channel.
        selected_channels (list): Channels to highlight/label.
        sort_templates (bool): Whether to sort by peak timing.
        color, color_marker (str): Line and marker colors.
        ax (matplotlib axis): Where to draw.
        gtr (GraphAxonTracking): Optional, for branch coloring.
        label_branches, label_channels (bool): Add visual annotations.
        font_size (int): Label text size.
        gain (float): Scale factor for amplitude.
        marker_size (int): Marker size.
        add_scale_bar (bool): Whether to draw scale bars.
        sampling_frequency (int): Needed for scale bar timing.

    Returns:
        ax: The matplotlib axis object.
    """
    import matplotlib.cm as cm

    if ax is None:
        fig, ax = plt.subplots()

    template_selected = template[selected_channels]
    locs = np.array(locations)[selected_channels]

    # Sort by time of min peak if needed
    if sort_templates:
        peaks = np.argmin(template_selected, axis=1)
        sorted_idxs = np.argsort(peaks)
        template_selected = template_selected[sorted_idxs]
        locs = locs[sorted_idxs]
        selected_channels = np.array(selected_channels)[sorted_idxs]

    ptp_glob = np.max(np.ptp(template_selected, axis=1))

    # Optional: branch coloring
    branch_colors = None
    if gtr and label_branches:
        num_branches = len(gtr.branches)
        branch_colors = cm.get_cmap('coolwarm', num_branches)
    else:
        raise ValueError("GraphAxonTracking object is required.")
    
    # Plot each channel's waveform
    for i, temp in enumerate(template_selected):
        temp_scaled = temp * gain
        temp_shifted = temp_scaled + i * 1.5 * ptp_glob

        ax.plot(temp_shifted, color=color, linewidth=line_thickness)

        # Mark peak (min)
        min_idx = np.argmin(temp_shifted)
        min_val = temp_shifted[min_idx]

        # Determine marker color based on branch
        if gtr and label_branches and branch_colors:
            branch_idx = next((j for j, b in enumerate(gtr.branches)
                               if selected_channels[i] in b['channels']), None)
            marker_color = branch_colors(branch_idx) if branch_idx is not None else color_marker
        else:
            marker_color = color_marker

        ax.plot(min_idx, min_val, marker=marker, color=marker_color,
                markersize=marker_size, markerfacecolor=markerfacecolor)

        if label_channels:
            try:
                max_idx = np.argmax(temp_shifted[:min_idx])
                y_label = np.mean(temp_shifted[:max_idx])
                if np.isnan(y_label): raise ValueError
            except:
                y_label = temp_shifted[0]

            ax.text(-0.1, y_label + 1, f"Ch {selected_channels[i]}", fontsize=font_size,
                    ha='left', va='bottom')

    # Optional legend
    if gtr and label_branches:
        handles = [
            plt.Line2D([0], [0], marker='o', color='w', label=f'Branch {i}',
                       markerfacecolor=branch_colors(i), markersize=10)
            for i in range(len(gtr.branches))
        ]
        ax.legend(handles=handles, loc='upper right', fontsize=font_size)

    # Optional scale bar
    if add_scale_bar:
        y_scale = np.round(0.2 * ptp_glob / 10) * 10
        y_scaled = y_scale * gain
        x_samples = int((2 / 1000) * sampling_frequency)
        x_ms = x_samples / sampling_frequency * 1000

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        bar_x = xlim[1] - 0.05 * (xlim[1] - xlim[0]) - x_samples
        bar_y = ylim[0] + 0.05 * (ylim[1] - ylim[0])

        ax.plot([bar_x, bar_x + x_samples], [bar_y, bar_y], color='k', lw=2)
        ax.plot([bar_x, bar_x], [bar_y, bar_y + y_scaled], color='k', lw=2)
        ax.text(bar_x + x_samples / 2, bar_y - 0.05 * (ylim[1] - ylim[0]),
                f'{x_ms:.1f} ms', ha='center', va='top', fontsize=font_size)
        ax.text(bar_x - 0.05 * (xlim[1] - xlim[0]), bar_y + y_scaled / 2,
                f'{y_scale:.1f} µV', ha='right', va='center', fontsize=font_size)

    ax.axis('off')
    return ax

def propogation_plot_wrapper(gtr, template, selected_channels, locations,
                              unit_id, template_type, ax=None, fig=None, gain=1, cmap="rainbow"):
    """
    Wrapper to build propagation plot for a unit's template.

    Args:
        gtr (GraphAxonTracking): Reconstruction graph.
        template (np.ndarray): Waveform template.
        selected_channels (list): Channels selected for the axon.
        locations (list): Locations of all channels.
        unit_id (str): ID of the unit.
        template_type (str): 'vt' or 'dvdt'.
        ax (matplotlib axis): Axis to plot into.
        fig (matplotlib figure): Optional figure.
        gain (float): Vertical scaling gain.

    Returns:
        ax: The matplotlib axis with the plot drawn.
    """
    assert ax is not None, "An axis object must be provided."

    trans_template = template.T
    return plot_template_propagation(
        trans_template, locations, selected_channels,
        sort_templates=True, color='C0', color_marker='r',
        fig=fig, ax=ax, line_thickness=1.0, gtr=gtr,
        label_channels=True, 
        #font_size=8, 
        #marker_size=5,
        markerfacecolor='none', gain=gain,
        label_branches=True, add_scale_bar=True, sampling_frequency=10000, cmap=cmap)

def generate_reconstruction_summary_plots(template_data, reconstruction_data,
                                          reconstruction_analytics, recon_dir, unit_id):
    """
    Generates summary plots for each template type for a given unit.

    Args:
        template_data (dict): Input templates.
        reconstruction_data (dict): GTR objects.
        reconstruction_analytics (dict): Analytics for each reconstruction.
        recon_dir (Path): Directory to save figures.
        unit_id (str): Unit identifier.
    """
    for template_type in ['vt', 'dvdt']:
        if template_type not in reconstruction_data:
            continue

        gtr = reconstruction_data[template_type]
        template = template_data[template_type]
        locations = template_data['channel_locations']
        selected_channels = gtr.selected_channels

        build_reconstruction_summary_fig(
            gtr, template, selected_channels, locations,
            recon_dir, unit_id, template_type
        )

    print(f"Finished summary plots for unit {unit_id}")

def build_reconstruction_summary_fig(gtr, template, selected_channels, locations, recon_dir, unit_id, template_type):
    """
    Builds and saves individual + summary plots for a reconstructed unit, tightly arranged without titles.
    """
    unit_dir = recon_dir / f"{unit_id}_{template_type}"
    unit_dir.mkdir(parents=True, exist_ok=True)
    tick_fontsize = 15

    def save_plot(wrapper, file_prefix, fig_size, title_fontsize=12, tick_fontsize=10, legend_fontsize=None, **kwargs):
        fig, ax = plt.subplots(figsize=fig_size)
        try:
            ax = wrapper(ax=ax, **kwargs)
        except Exception:
            if ax:
                ax.clear()
                ax.set_xticks([])
                ax.set_yticks([])
            fig = wrapper(fig=fig, **kwargs)

        # ⬇️ Update ALL axes in the figure
        for ax in fig.axes:
            ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
            ax.tick_params(axis='both', which='minor', labelsize=tick_fontsize)
            if ax.get_title():
                ax.set_title(ax.get_title(), fontsize=title_fontsize)

            # ⬇️ Update legend font sizes explicitly
            legend = ax.get_legend()
            if legend and legend_fontsize:
                for text in legend.get_texts():
                    text.set_fontsize(legend_fontsize)

        plt.tight_layout()
        png_path = unit_dir / f"{file_prefix}.png"
        pdf_path = unit_dir / f"{file_prefix}.pdf"
        print(f"Saving {png_path}")
        fig.savefig(png_path, dpi=600)
        with PdfPages(pdf_path) as pdf:
            pdf.savefig(fig)
        plt.close(fig)

    def save_plot_dep(wrapper, file_prefix, fig_size, title_fontsize=12, tick_fontsize=10, **kwargs):
        fig, ax = plt.subplots(figsize=fig_size)
        try:
            ax = wrapper(ax=ax, **kwargs)
            # Adjust font sizes for titles and ticks
            if ax:
                ax.title.set_fontsize(title_fontsize)
                ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
                ax.tick_params(axis='both', which='minor', labelsize=tick_fontsize)
                
        except Exception:
            # if ax failed, make sure to clear it before doing fig
            if ax:
                ax.clear()
                ax.set_xticks([])
                ax.set_yticks([])
            
            fig = wrapper(fig=fig, **kwargs)
            for ax in fig.axes:
                ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
                ax.tick_params(axis='both', which='minor', labelsize=tick_fontsize)
                ax.title.set_fontsize(title_fontsize)
        
            # adjust legend font size
            if hasattr(fig, 'legend_'):
                fig.legend_.set_fontsize(tick_fontsize)
                
        finally:
            plt.tight_layout()
            png_path = unit_dir / f"{file_prefix}.png"
            pdf_path = unit_dir / f"{file_prefix}.pdf"
            print(f"Saving {png_path}")
            fig.savefig(png_path, dpi=600)
            with PdfPages(pdf_path) as pdf:
                pdf.savefig(fig)
            plt.close(fig)

    # --- Save Propagation Plot
    save_plot(
        propogation_plot_wrapper,
        f"{unit_id}_{template_type}_propagation_plot",
        fig_size=(8, 16),
        gtr=gtr, template=template,
        selected_channels=selected_channels,
        locations=locations,
        unit_id=unit_id,
        template_type=template_type,
        cmap="tab10"
    )

    # --- Save Clean Branches (Full)
    save_plot(
        gtr.plot_clean_branches,
        f"{unit_id}_{template_type}_clean_branches_full",
        fig_size=(12, 6),
        plot_full_template=True,
        cmap="tab10",
        title_fontsize=30,
        tick_fontsize=tick_fontsize,
        legend_fontsize=20,
    )

    # --- Save Velocity Plot
    try:
        save_plot(
            gtr.plot_velocities,
            f"{unit_id}_{template_type}_velocity_plot",
            fig_size=(8, 16),
            plot_outliers=True,
            cmap="tab10",
            title_fontsize=30,
            tick_fontsize=tick_fontsize*1.5,
            lw=5,
            markersize=15,
            fs=20,
        )
    except Exception:
        save_plot(
            gtr.plot_velocities,
            f"{unit_id}_{template_type}_velocity_plot",
            fig_size=(8, 16),
            plot_outliers=False,
            cmap="tab10",
            title_fontsize=30,
            tick_fontsize=tick_fontsize*1.5,
            legend_fontsize=20,
            lw=5,
            markersize=15,
            fs=20,
        )

    # --- Save Raw Branches (Focused)
    raw_fig, raw_ax = plt.subplots(figsize=(12, 6))
    raw_ax = gtr.plot_raw_branches(ax=raw_ax, plot_full_template=False, cmap="tab10")
    branch_coords = np.vstack([gtr.locations[branch["channels"]] for branch in gtr.branches])
    x_min, y_min = branch_coords.min(axis=0) - 100
    x_max, y_max = branch_coords.max(axis=0) + 100
    raw_ax.set_xlim(x_min, x_max)
    raw_ax.set_ylim(y_min, y_max)
    # Adjust font sizes for titles and ticks
    raw_ax.title.set_fontsize(30)
    raw_ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    raw_ax.tick_params(axis='both', which='minor', labelsize=tick_fontsize)    
    
    plt.tight_layout()
    raw_file_prefix = f"{unit_id}_{template_type}_raw_branches"
    raw_fig.savefig(unit_dir / f"{raw_file_prefix}.png", dpi=600)
    with PdfPages(unit_dir / f"{raw_file_prefix}.pdf") as pdf:
        pdf.savefig(raw_fig)
    plt.close(raw_fig)

    # --- Save Clean Branches (Focused)
    clean_fig, clean_ax = plt.subplots(figsize=(12, 6))
    clean_ax = gtr.plot_clean_branches(ax=clean_ax, plot_full_template=False, cmap="tab10")
    clean_ax.set_xlim(x_min, x_max)
    clean_ax.set_ylim(y_min, y_max)
    # Adjust font sizes for titles and ticks
    clean_ax.title.set_fontsize(30)
    clean_ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    clean_ax.tick_params(axis='both', which='minor', labelsize=tick_fontsize)
    
    plt.tight_layout()
    clean_file_prefix = f"{unit_id}_{template_type}_clean_branches"
    clean_fig.savefig(unit_dir / f"{clean_file_prefix}.png", dpi=600)
    with PdfPages(unit_dir / f"{clean_file_prefix}.pdf") as pdf:
        pdf.savefig(clean_fig)
    plt.close(clean_fig)

    # --- Compose Summary Figure (2x4 grid, no titles, tight spacing)
    fig = plt.figure(figsize=(18, 9))
    gs = plt.GridSpec(2, 4, figure=fig)

    def add_subplot(gs_position, img_file):
        ax = fig.add_subplot(gs_position)
        ax.imshow(plt.imread(unit_dir / img_file))
        ax.axis("off")

    add_subplot(gs[:, 0], f"{unit_id}_{template_type}_propagation_plot.png")
    add_subplot(gs[0, 1:3], f"{unit_id}_{template_type}_clean_branches_full.png")
    add_subplot(gs[1, 1:3], f"{unit_id}_{template_type}_clean_branches.png")
    add_subplot(gs[:, 3], f"{unit_id}_{template_type}_velocity_plot.png")

    plt.subplots_adjust(
        left=0.005, right=0.99, top=0.99, bottom=0.01,
        wspace=0.02, hspace=0.0000
    )

    summary_png = unit_dir / f"{unit_id}_reconstruction_summary.png"
    summary_pdf = unit_dir / f"{unit_id}_reconstruction_summary.pdf"
    fig.savefig(summary_png, dpi=300)
    with PdfPages(summary_pdf) as pdf:
        pdf.savefig(fig)
    plt.close(fig)

    print(f"Saved {summary_png}")
    print(f"Saved {summary_pdf}")
    print(f"Finished summary figure for unit {unit_id}")

def build_reconstruction_summary_fig_deb3_standardcolors(gtr, template, selected_channels, locations, recon_dir, unit_id, template_type):
    """
    Builds and saves individual + summary plots for a reconstructed unit, tightly arranged without titles.
    """
    # unit_dir
    unit_dir = recon_dir / f"{unit_id}_{template_type}"
    unit_dir.mkdir(parents=True, exist_ok=True)
    
    def save_plot(wrapper, file_prefix, fig_size, **kwargs):
        fig, ax = plt.subplots(figsize=fig_size)
        try:
            ax = wrapper(ax=ax, **kwargs)
        except Exception:
            fig = wrapper(fig=fig, **kwargs)
        finally:
            plt.tight_layout()
            png_path = unit_dir / f"{file_prefix}.png"
            pdf_path = unit_dir / f"{file_prefix}.pdf"
            print(f"Saving {png_path}")
            fig.savefig(png_path, dpi=600)
            with PdfPages(pdf_path) as pdf:
                pdf.savefig(fig)
            plt.close(fig)

    # --- Save Propagation Plot
    save_plot(
        propogation_plot_wrapper,
        f"{unit_id}_{template_type}_propagation_plot",
        fig_size=(8, 16),
        gtr=gtr, template=template,
        selected_channels=selected_channels,
        locations=locations,
        unit_id=unit_id,
        template_type=template_type
    )

    # --- Save Clean Branches (Full)
    save_plot(
        gtr.plot_clean_branches,
        f"{unit_id}_{template_type}_clean_branches_full",
        fig_size=(12, 6),
        plot_full_template=True
    )

    # --- Save Velocity Plot
    try: #NOTE: gtr.plot_velocities fails if there are no outliers...which is dumb
        save_plot(
            gtr.plot_velocities,
            f"{unit_id}_{template_type}_velocity_plot",
            fig_size=(8, 16),
            plot_outliers=True
            #plot_outliers=False
        )
    except Exception as e:
        save_plot(
            gtr.plot_velocities,
            f"{unit_id}_{template_type}_velocity_plot",
            fig_size=(8, 16),
            #plot_outliers=True
            plot_outliers=False
        )

    # --- Save Raw Branches (Focused)
    raw_fig, raw_ax = plt.subplots(figsize=(12, 6))
    raw_ax = gtr.plot_raw_branches(ax=raw_ax, plot_full_template=False)
    branch_coords = np.vstack([gtr.locations[branch["channels"]] for branch in gtr.branches])
    x_min, y_min = branch_coords.min(axis=0) - 100
    x_max, y_max = branch_coords.max(axis=0) + 100
    raw_ax.set_xlim(x_min, x_max)
    raw_ax.set_ylim(y_min, y_max)
    plt.tight_layout()
    raw_file_prefix = f"{unit_id}_{template_type}_raw_branches"
    raw_fig.savefig(unit_dir / f"{raw_file_prefix}.png", dpi=600)
    with PdfPages(unit_dir / f"{raw_file_prefix}.pdf") as pdf:
        pdf.savefig(raw_fig)
    plt.close(raw_fig)

    # --- Save Clean Branches (Focused)
    clean_fig, clean_ax = plt.subplots(figsize=(12, 6))
    clean_ax = gtr.plot_clean_branches(ax=clean_ax, plot_full_template=False)
    clean_ax.set_xlim(x_min, x_max)
    clean_ax.set_ylim(y_min, y_max)
    plt.tight_layout()
    clean_file_prefix = f"{unit_id}_{template_type}_clean_branches"
    clean_fig.savefig(unit_dir / f"{clean_file_prefix}.png", dpi=600)
    with PdfPages(unit_dir / f"{clean_file_prefix}.pdf") as pdf:
        pdf.savefig(clean_fig)
    plt.close(clean_fig)

    # --- Compose Summary Figure (2x4 grid, no titles, tight spacing)
    fig = plt.figure(figsize=(18, 9))
    gs = plt.GridSpec(2, 4, figure=fig)

    def add_subplot(gs_position, img_file):
        ax = fig.add_subplot(gs_position)
        ax.imshow(plt.imread(unit_dir / img_file))
        ax.axis("off")

    add_subplot(gs[:, 0], f"{unit_id}_{template_type}_propagation_plot.png")
    add_subplot(gs[0, 1:3], f"{unit_id}_{template_type}_clean_branches_full.png")
    #add_subplot(gs[1, 1], f"{unit_id}_{template_type}_raw_branches.png")
    #add_subplot(gs[1, 2], f"{unit_id}_{template_type}_clean_branches.png")
    add_subplot(gs[1, 1:3], f"{unit_id}_{template_type}_clean_branches.png")
    add_subplot(gs[:, 3], f"{unit_id}_{template_type}_velocity_plot.png")

    # ⬇️ Tight layout control
    plt.subplots_adjust(
        left=0.005, 
        right=0.99,
        top=0.99, 
        bottom=0.01,
        wspace=0.02, 
        hspace=0.0000 # hspace is the vertical space between rows
    )

    summary_png = unit_dir / f"{unit_id}_reconstruction_summary.png"
    summary_pdf = unit_dir / f"{unit_id}_reconstruction_summary.pdf"
    fig.savefig(summary_png, dpi=300)
    with PdfPages(summary_pdf) as pdf:
        pdf.savefig(fig)
    plt.close(fig)

    print(f"Saved {summary_png}")
    print(f"Saved {summary_pdf}")
    print(f"Finished summary figure for unit {unit_id}")

def build_reconstruction_summary_fig_dep2(gtr, template, selected_channels, locations, recon_dir, unit_id, template_type):
    """
    Builds and saves individual + summary plots for a reconstructed unit.

    Args:
        gtr (GraphAxonTracking): Graph-tracked axon structure.
        template (np.ndarray): Waveform data.
        selected_channels (list): Channel IDs used.
        locations (list): Channel locations.
        recon_dir (Path): Directory to save output.
        unit_id (str): Unit identifier.
        template_type (str): 'vt' or 'dvdt'
    """
    def save_plot(wrapper, file_prefix, fig_size, **kwargs):
        fig, ax = plt.subplots(figsize=fig_size)
        try:
            ax = wrapper(ax=ax, **kwargs)
        except Exception:
            fig = wrapper(fig=fig, **kwargs)
        finally:
            plt.tight_layout()
            png_path = recon_dir / f"{file_prefix}.png"
            pdf_path = recon_dir / f"{file_prefix}.pdf"
            print(f"Saving {png_path}")
            fig.savefig(png_path, dpi=600)
            with PdfPages(pdf_path) as pdf:
                pdf.savefig(fig)
            plt.close(fig)

    # --- Save Propagation Plot (left column)
    save_plot(
        propogation_plot_wrapper,
        f"{unit_id}_{template_type}_propagation_plot",
        fig_size=(5, 15),
        gtr=gtr, template=template,
        selected_channels=selected_channels,
        locations=locations,
        unit_id=unit_id,
        template_type=template_type
    )

    # --- Save Clean Branches (Full Template)
    save_plot(
        gtr.plot_clean_branches,
        f"{unit_id}_{template_type}_clean_branches_full",
        fig_size=(12, 6),
        plot_full_template=True
    )

    # --- Save Velocity Plot (Right Column)
    save_plot(
        gtr.plot_velocities,
        f"{unit_id}_{template_type}_velocity_plot",
        fig_size=(5, 15),
        plot_outliers=True
    )

    # --- Save Raw Branches (Focused View with axis-limited bounding box)
    raw_fig, raw_ax = plt.subplots(figsize=(12, 6))
    raw_ax = gtr.plot_raw_branches(ax=raw_ax, plot_full_template=False)
    branch_coords = np.vstack([gtr.locations[branch["channels"]] for branch in gtr.branches])
    x_min, y_min = branch_coords.min(axis=0) - 100
    x_max, y_max = branch_coords.max(axis=0) + 100
    raw_ax.set_xlim(x_min, x_max)
    raw_ax.set_ylim(y_min, y_max)
    plt.tight_layout()
    raw_file_prefix = f"{unit_id}_{template_type}_raw_branches"
    raw_fig.savefig(recon_dir / f"{raw_file_prefix}.png", dpi=600)
    with PdfPages(recon_dir / f"{raw_file_prefix}.pdf") as pdf:
        pdf.savefig(raw_fig)
    plt.close(raw_fig)

    # --- Save Clean Branches (Focused View)
    clean_fig, clean_ax = plt.subplots(figsize=(12, 6))
    clean_ax = gtr.plot_clean_branches(ax=clean_ax, plot_full_template=False)
    clean_ax.set_xlim(x_min, x_max)
    clean_ax.set_ylim(y_min, y_max)
    plt.tight_layout()
    clean_file_prefix = f"{unit_id}_{template_type}_clean_branches"
    clean_fig.savefig(recon_dir / f"{clean_file_prefix}.png", dpi=600)
    with PdfPages(recon_dir / f"{clean_file_prefix}.pdf") as pdf:
        pdf.savefig(clean_fig)
    plt.close(clean_fig)

    # --- Compose Summary Figure with 2x4 Grid
    fig = plt.figure(figsize=(18, 9))
    gs = plt.GridSpec(2, 4, figure=fig)

    def add_subplot(gs_position, title, img_file):
        ax = fig.add_subplot(gs_position)
        ax.set_title(title, fontsize=12)
        ax.imshow(plt.imread(recon_dir / img_file))
        ax.axis("off")

    add_subplot(gs[:, 0], "Propagation", f"{unit_id}_{template_type}_propagation_plot.png")
    add_subplot(gs[0, 1:3], "Clean Branches (Full)", f"{unit_id}_{template_type}_clean_branches_full.png")
    add_subplot(gs[1, 1], "Raw Branches", f"{unit_id}_{template_type}_raw_branches.png")
    add_subplot(gs[1, 2], "Clean Branches (Focused)", f"{unit_id}_{template_type}_clean_branches.png")
    add_subplot(gs[:, 3], "Velocity", f"{unit_id}_{template_type}_velocity_plot.png")

    plt.tight_layout()
    summary_png = recon_dir / f"{unit_id}_reconstruction_summary.png"
    summary_pdf = recon_dir / f"{unit_id}_reconstruction_summary.pdf"
    fig.savefig(summary_png, dpi=300)
    with PdfPages(summary_pdf) as pdf:
        pdf.savefig(fig)
    plt.close(fig)

    print(f"Saved {summary_png}")
    print(f"Saved {summary_pdf}")
    print(f"Finished summary figure for unit {unit_id}")

def build_reconstruction_summary_fig_dep1(gtr, template, selected_channels, locations, recon_dir, unit_id, template_type):
    """
    Builds and saves individual + summary plots for a reconstructed unit.

    Args:
        gtr (GraphAxonTracking): Graph-tracked axon structure.
        template (np.ndarray): Waveform data.
        selected_channels (list): Channel IDs used.
        locations (list): Channel locations.
        recon_dir (Path): Directory to save output.
        unit_id (str): Unit identifier.
        template_type (str): 'vt' or 'dvdt'
    """
    def save_plot(wrapper, file_prefix, fig_size, **kwargs):
        fig, ax = plt.subplots(figsize=fig_size)
        try:
            ax = wrapper(ax=ax, **kwargs)
        except:
            fig = wrapper(fig=fig, **kwargs)  # fallback
        finally:
            plt.tight_layout()
            png_path = recon_dir / f"{file_prefix}.png"
            pdf_path = recon_dir / f"{file_prefix}.pdf"
            print(f"Saving {png_path}")
            print(f"Saving {pdf_path}")
            fig.savefig(png_path, dpi=600)
            with PdfPages(pdf_path) as pdf:
                pdf.savefig(fig)
            plt.close(fig)

    # Save each plot
    # propagation plot - entire left-most column
    save_plot(propogation_plot_wrapper, f"{unit_id}_{template_type}_propagation_plot",
              fig_size=(5, 15), gtr=gtr, template=template,
              selected_channels=selected_channels, locations=locations,
              unit_id=unit_id, template_type=template_type)
    
    #clean branches - full template - two middle columns in top row
    save_plot(gtr.plot_clean_branches, f"{unit_id}_{template_type}_clean_branches_full", fig_size=(12, 6), plot_full_template=True)
    
    # raw branches - focused template - left middle column in bottom row
    # plot ahead of time and then constrain to size of branches + some buffer incase erroneous channels in template very far away
    # ...
    # then save
    save_plot(gtr.plot_raw_branches, f"{unit_id}_{template_type}_raw_branches", fig_size=(12, 6), plot_template_propagation=False)

    # clean branches - focused template - right middle column in bottom row
    # plot ahead of time and then constrain to the same size as raw branches + some buffer incase erroneous channels in template very far away
    # ...
    # then save
    save_plot(gtr.plot_clean_branches, f"{unit_id}_{template_type}_clean_branches", fig_size=(12, 6), plot_full_template=True)

    # velocity plot - entire right-most column
    save_plot(gtr.plot_velocities, f"{unit_id}_{template_type}_velocity_plot",
              fig_size=(5, 15), plot_outliers=True)

    # Assemble the final figure
    fig = plt.figure(figsize=(16, 9))
    gs = plt.GridSpec(3, 3, figure=fig)

    def add_subplot(gs_position, title, img_file):
        ax = fig.add_subplot(gs_position)
        ax.set_title(title, fontsize=12)
        ax.imshow(plt.imread(recon_dir / img_file))
        ax.axis("off")

    add_subplot(gs[:, 0], "Propagation", f"{unit_id}_{template_type}_propagation_plot.png")
    add_subplot(gs[0, 1], "Clean Branches", f"{unit_id}_{template_type}_clean_branches.png")
    add_subplot(gs[1, 1], "Raw Branches", f"{unit_id}_{template_type}_raw_branches.png")
    add_subplot(gs[:, 2], "Velocity", f"{unit_id}_{template_type}_velocity_plot.png")

    plt.tight_layout()
    fig.savefig(recon_dir / f"{unit_id}_reconstruction_summary.png", dpi=300)
    with PdfPages(recon_dir / f"{unit_id}_reconstruction_summary.pdf") as pdf:
        pdf.savefig(fig)
    plt.close(fig)
    
    png_path = recon_dir / f"{unit_id}_reconstruction_summary.png"
    pdf_path = recon_dir / f"{unit_id}_reconstruction_summary.pdf"
    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")

    print(f"Saved summary figure for unit {unit_id}")

def generate_plots(template_data, reconstruction_data, reconstruction_analytics, recon_dir, unit_id):
    """
    Wrapper to generate all relevant plots for a given unit.

    Args:
        template_data (dict): Template waveforms and metadata.
        reconstruction_data (dict): GTR objects per template.
        reconstruction_analytics (dict): Analytics computed from reconstructions.
        recon_dir (Path): Output directory for plots.
        unit_id (str): Unit identifier.
    """
    kwargs = {
        'template_data': template_data,
        'reconstruction_data': reconstruction_data,
        'reconstruction_analytics': reconstruction_analytics,
        'recon_dir': recon_dir,
        'unit_id': unit_id,
    }

    try:
        generate_reconstruction_summary_plots(**kwargs)
        print(f"Generated summary plots for unit {unit_id}")
    except Exception as e:
        print(f"Failed to generate summary plots for unit {unit_id}: {e}")
    
    #print(f"Finished plotting for unit {unit_id}")

# core logic ==========================================================
def calculate_template_rectangular_area(template_data):
    """
    Calculates the area of the rectangular bounding box around the channel locations.

    Args:
        template_data (dict): Contains 'channel_locations': List of [x, y].

    Returns:
        float: Area in square micrometers (µm²).
    """
    x_coords = [loc[0] for loc in template_data['channel_locations']]
    y_coords = [loc[1] for loc in template_data['channel_locations']]

    width = max(x_coords) - min(x_coords)
    height = max(y_coords) - min(y_coords)

    return width * height

def calculate_branch_order(branch_channels, branching_points):
    """
    Calculates the order (depth) of a branch by counting branching points it passes through.

    Args:
        branch_channels (list[int]): Channels along the branch path.
        branching_points (set[int]): Channels identified as branching points.

    Returns:
        int: Number of branching points encountered (branch order).
    """
    return sum(1 for ch in branch_channels if ch in branching_points)

def transform_data(merged_template, merged_channel_loc):
    """
    Transposes the template and flips y-coordinates for spatial consistency.

    Args:
        merged_template (np.ndarray): Waveform template, shape (channels x time).
        merged_channel_loc (list or np.ndarray): List of [x, y] locations for each channel.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (transposed_template, flipped_channel_locations)
    """
    transposed_template = merged_template.T  # Shape: (time x channels)
    flipped_locations = np.array([[x, -y] for x, y in merged_channel_loc])
    return transposed_template, flipped_locations

def compute_reconstruction_analytics(template_data, gtr):
    """
    Computes a dictionary of analytics for a reconstructed unit.

    Args:
        template_data (dict): Original template and channel locations.
        gtr (GraphAxonTracking): Reconstructed graph object.

    Returns:
        dict: Summary and branch-level analytics.
    """
    branches = gtr.branches
    branch_orders = [calculate_branch_order(b['channels'], gtr._branching_points) for b in branches]

    analytics = {
        'unit_id': template_data['unit_id'],
        'num_branches': len(branches),
        'channels_in_template': len(template_data['channel_locations']),
        'template_rect_area': calculate_template_rectangular_area(template_data),
        'template_density': len(template_data['channel_locations']) / calculate_template_rectangular_area(template_data),
        'axon_length': gtr.compute_path_length(branches[0]['channels']),
        'total_axon_matter': sum(gtr.compute_path_length(b['channels']) for b in branches),
        'maximum_branch_order': max(branch_orders),
        'branch_ids': list(range(len(branches))),
        'velocities': [b['velocity'] for b in branches],
        'offsets': [b['offset'] for b in branches],
        'r_squares': [b['r2'] for b in branches],
        'pvals': [b['pval'] for b in branches],
        'branch_lengths': [gtr.compute_path_length(b['channels']) for b in branches],
        'channels_per_branch': [len(b['channels']) for b in branches],
        'branch_channel_density': [
            gtr.compute_path_length(b['channels']) / len(b['channels']) for b in branches
        ],
        'branch_orders': branch_orders,
        '': ''  # separator column (optional aesthetic)
    }

    return analytics

def build_graph_axon_tracking(template_data, recon_dir=None, load_gtr=False,
                               params=None, template_type='dvdt'):
    """
    Loads or builds a GraphAxonTracking object for axon reconstruction.

    Args:
        template_data (dict): Contains waveform, locations, and unit ID.
        recon_dir (Path or str): Directory for saving/loading GTR object.
        load_gtr (bool): Attempt to load precomputed GTR object from disk.
        params (dict): Parameters for GraphAxonTracking. If None, defaults are used.
        template_type (str): Type of template: 'vt', 'dvdt', or 'milos'.

    Returns:
        Tuple[GraphAxonTracking, dict]: The GTR object and its parameters.
    """
    unit_id = template_data['unit_id']
    params = params or get_default_graph_velocity_params()
    recon_dir = Path(recon_dir) if recon_dir else None

    # Prepare template and locations
    template = template_data[template_type]
    locations = template_data['channel_locations']
    trans_template, trans_locations = transform_data(template, locations)

    # Define file path
    gtr_file = recon_dir / template_type / f"{unit_id}_gtr_object.dill" if recon_dir else None

    # Try to load existing GTR
    # if load_gtr and gtr_file and gtr_file.exists():
    #     try:
    #         with open(gtr_file, 'rb') as f:
    #             gtr = dill.load(f)
    #         print(f"[GTR] Loaded from {gtr_file}")
    #         # Ensure it’s usable
    #         gtr._verbose = 1
    #         gtr.select_channels()
    #         gtr.build_graph()
    #         gtr.find_paths()
    #         gtr.clean_paths(remove_outliers=False)
    #         return gtr, params
    #     except Exception as e:
    #         print(f"[GTR] Failed to load {gtr_file}, falling back to rebuild. Reason: {e}")

    # Build GTR from scratch
    print(f"[GTR] Building for unit {unit_id} using template '{template_type}'")
    gtr = GraphAxonTracking(trans_template, trans_locations, fs=10000, **params)
    gtr._verbose = 1
    gtr.select_channels()
    gtr.build_graph()
    gtr.find_paths()
    gtr.clean_paths(remove_outliers=False)

    if gtr.branches:
        print(f"[GTR] Found {len(gtr.branches)} branches")
        if len(gtr.branches) > 1:
            print()
            #print(f"[GTR] Branches: {', '.join(str(b['channels']) for b in gtr.branches)}")
    else:
        print(f"[GTR] No branches found (reconstruction may have failed)")

    return gtr, params

def reconstruct_template(template_data, template_type, recon_dir, params, logger=None):
    """
    Attempts to load or reconstruct the template into a GTR object.

    Args:
        template_data (dict): Template data and metadata.
        template_type (str): Type of template ('vt', 'dvdt').
        recon_dir (Path): Directory to save the GTR.
        params (dict): Parameters for GTR reconstruction.
        logger (Logger): Optional logger.

    Returns:
        GraphAxonTracking: Reconstructed GTR object.
    """
    try:
        gtr, _ = build_graph_axon_tracking(
            template_data, recon_dir=recon_dir,
            #load_gtr=True, 
            load_gtr=False,
            params=params, template_type=template_type
        )
    except Exception as e:
        msg = f"Failed to build GTR for unit {template_data['unit_id']} with template {template_type}: {e}"
        if logger:
            logger.error(msg)
        else:
            print(msg)
        return None

    # Save GTR to disk
    if recon_dir:
        output_dir = Path(recon_dir) / template_type
        output_dir.mkdir(parents=True, exist_ok=True)
        gtr_file = output_dir / f"{template_data['unit_id']}_gtr_object.dill"
        with open(gtr_file, 'wb') as f:
            dill.dump(gtr, f)

    return gtr

def extract_template_data(unit_id, unit_templates):
    """
    Formats and standardizes template data for processing.

    Args:
        unit_id (str or int): Unit identifier.
        unit_templates (dict): Raw templates from input data.

    Returns:
        dict: Formatted template data for reconstruction pipeline.
    """
    return {
        'unit_id': unit_id,
        'vt': unit_templates.get('merged_template'),
        'dvdt': get_time_derivative(unit_templates.get('merged_template'), sampling_rate=10000),
        'milos': [],  # Placeholder for future use
        'channel_locations': unit_templates.get('merged_channel_locs'),
    }

def process_unit_for_analysis(unit_id, unit_templates, recon_dir, analysis_options, params, logger=None):
    """
    Processes a single unit: extract template, reconstruct axon paths,
    compute analytics, and generate plots.

    Args:
        unit_id (str or int): Unit identifier.
        unit_templates (dict): Templates and locations for the unit.
        recon_dir (Path): Output directory.
        analysis_options (dict): Optional parameters (currently unused).
        params (dict): Parameters for axon tracking.
        logger (Logger): Optional logger.

    Returns:
        dict: A dictionary with template data, reconstructions, and analytics.
    """
    start = time.time()
    print(f"Analyzing unit {unit_id}...")

    template_data = extract_template_data(unit_id, unit_templates)
    reconstruction_data = {}
    reconstruction_analytics = {}

    for template_type in ['vt', 'dvdt']:  # Skip 'milos' for now
        try:
            gtr = reconstruct_template(template_data, template_type, recon_dir, params, logger)
            reconstruction_data[template_type] = gtr
            reconstruction_analytics[template_type] = compute_reconstruction_analytics(template_data, gtr)
        except Exception as e:
            if logger:
                logger.error(f"Failed to process unit {unit_id} with template {template_type}: {e}")
            else:
                print(f"Failed to process unit {unit_id} with template {template_type}: {e}")

    generate_plots(template_data, reconstruction_data, reconstruction_analytics, recon_dir, unit_id)

    print(f"Finished analyzing unit {unit_id} in {time.time() - start:.2f} seconds")

    return {
        'template_data': template_data,
        'reconstruction_data': reconstruction_data,
        'reconstruction_analytics': reconstruction_analytics,
    }

# helper functions =======================================================
def get_time_derivative(merged_template, 
                        #merged_template_filled, 
                        unit='seconds', sampling_rate=10000, axis=0):
    """
    Computes the time derivative of the input arrays along the specified axis.
    
    Parameters:
    - merged_template: numpy.ndarray
        The first input array where the y-axis describes channels and x-axis describes voltage(time) signals.
    - merged_template_filled: numpy.ndarray
        The second input array where the y-axis describes channels and x-axis describes voltage(time) signals.
    - unit: str, optional
        The time unit for the derivative, either 'seconds' or 'ms'. Default is 'seconds'.
    - sampling_rate: int, optional
        The sampling rate in samples per second. Default is 10000.
    - axis: int, optional
        The axis along which to compute the derivative. Default is 0 (assuming time signals along y-axis).
    
    Returns:
    - d_merged_template: numpy.ndarray
        The time derivative of merged_template.
    - d_merged_template_filled: numpy.ndarray
        The time derivative of merged_template_filled.
    """
    
    if unit == 'seconds':
        delta_t = 1.0 / sampling_rate
    elif unit == 'ms':
        delta_t = 1.0 / (sampling_rate / 1000)
    else:
        raise ValueError("Invalid unit. Choose either 'seconds' or 'ms'.")

    d_merged_template = np.diff(merged_template, axis=axis) / delta_t
    #d_merged_template_filled = np.diff(merged_template_filled, axis=axis) / delta_t
    return d_merged_template #, d_merged_template_filled

def analysis_results_to_dataframe(analysis_results):
    """
    Converts nested analysis results into a flattened DataFrame.

    Args:
        analysis_results (dict): Dictionary of unit analytics.

    Returns:
        pd.DataFrame: Flattened DataFrame with one row per branch.
    """
    exploded_data = []
    for unit_id, row in analysis_results.items():
        max_length = max(len(v) if isinstance(v, list) else 1 for v in row.values())
        for i in range(max_length):
            new_entry = {'unit_id': row['unit_id'], 'branch_id': i}
            for key, value in row.items():
                if key == 'unit_id':
                    continue
                if isinstance(value, list):
                    new_entry[key] = value[i] if i < len(value) else None
                elif isinstance(value, dict):
                    new_entry[key] = value.get(i, None)
                else:
                    new_entry[key] = value
            exploded_data.append(new_entry)

    return pd.DataFrame(exploded_data)

def process_results(futures, analysis_json_dir, logger):
    """
    Waits for all futures to complete and gathers results.

    Args:
        futures (dict): Mapping of Future -> unit_id.
        analysis_json_dir (Path): Directory to save JSON results.
        logger (logging.Logger): Logger instance.

    Returns:
        dict: Results keyed by unit_id.
    """
    results = {}

    for future in as_completed(futures):
        unit_id = futures[future]
        try:
            result = future.result()
            results[unit_id] = result

            # Save JSON result
            if analysis_json_dir:
                json_path = Path(analysis_json_dir) / f"{unit_id}_analysis_results.json"
                with open(json_path, 'w') as f:
                    json.dump(result, f, indent=4, default=str)

            logger.info(f"JSON saved for unit {unit_id}: {json_path}")
            #logger.info(f"Unit {unit_id} processed successfully.")
        except Exception as e:
            #logger.error(f"Error processing unit {unit_id}: {e}")
            logger.error(f"Error saving JSON for unit {unit_id}: {e}") 

    return results

def submit_tasks(executor, unit_templates, gtr_dir, analysis_options, params, unit_limit, logger):
    """
    Submits processing jobs to a ProcessPoolExecutor.

    Args:
        executor (ProcessPoolExecutor): Executor to submit jobs to.
        unit_templates (dict): Templates for all units.
        gtr_dir (Path): Directory to save GTR files.
        analysis_options (dict): Options for analysis.
        params (dict): Parameters for reconstruction.
        unit_limit (int or None): Max number of units to submit.
        logger (logging.Logger): Logger instance.

    Returns:
        dict: Future-to-unit_id mapping.
    """
    futures = {}
    for unit_id, unit_template in unit_templates.items():
        if unit_limit is None or len(futures) < unit_limit:
            future = executor.submit(
                process_unit_for_analysis,
                unit_id, unit_template, gtr_dir, analysis_options, params, logger
            )
            futures[future] = unit_id
    return futures

def aggregate_analytics(results, stream_recon_dir, stream_id):
    """
    Aggregates analytics from all processed units and saves to CSV.

    Args:
        results (dict): Reconstruction results per unit.
        stream_recon_dir (Path): Output dir for CSV.
        stream_id (str): Stream identifier.
    """
    results = dict(sorted(results.items()))  # Sort for consistency

    for template_type in ['vt', 'dvdt', 'milos']:
        analytics_data = {
            unit_id: data['reconstruction_analytics'][template_type]
            for unit_id, data in results.items()
            if template_type in data.get('reconstruction_analytics', {})
        }

        if not analytics_data:
            print(f"[{template_type}] No analytics for stream {stream_id}.")
            continue

        try:
            df = analysis_results_to_dataframe(analytics_data)
            output_file = Path(stream_recon_dir).parent / f"agg_{stream_id}_{template_type}_axon_analytics.csv"
            df.to_csv(output_file, index=False)
            print(f"[{template_type}] Analytics saved: {output_file}")
        except Exception as e:
            print(f"Error saving analytics for {stream_id} - {template_type}: {e}")

def process_units(unit_templates, gtr_dir, analysis_options, params,
                  unit_limit, analysis_json_dir, logger, parallel=True, n_jobs=1):
    """
    Processes units using either parallel or serial execution.

    Args:
        unit_templates (dict): Templates for each unit.
        gtr_dir (Path): Directory to save GTR data.
        analysis_options (dict): Options for analysis.
        params (dict): Parameters for reconstruction.
        unit_limit (int or None): Max number of units to process.
        analysis_json_dir (Path): Output path for JSON results.
        logger (logging.Logger): Logger instance.
        parallel (bool): Whether to use parallel processing.
        n_jobs (int): Number of parallel workers.

    Returns:
        dict: Results keyed by unit_id.
    """
    results = {}
    successful_units = 0

    if parallel:
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            while unit_limit is None or successful_units < unit_limit:
                remaining = unit_limit - successful_units if unit_limit else None
                futures = submit_tasks(executor, unit_templates, gtr_dir,
                                       analysis_options, params, remaining, logger)
                new_results = process_results(futures, analysis_json_dir, logger)
                results.update(new_results)
                successful_units += len(new_results)
    else:
        for unit_id, unit_template in unit_templates.items():
            
            # if unit_id != 60:
            #     continue
            
            if unit_limit is not None and successful_units >= unit_limit:
                break
            try:
                result = process_unit_for_analysis(unit_id, unit_template, gtr_dir,
                                                   analysis_options, params, logger)
                results[unit_id] = result
                successful_units += 1
                #logger.info(f"Unit {unit_id} processed successfully.")
            except Exception as e:
                logger.error(f"Error processing unit {unit_id}: {e}")
                
    
    # --- After processing, collect successful GTRs and plot them globally
    try:
        from pathlib import Path
        import dill
        from matplotlib.backends.backend_pdf import PdfPages

        for template_type in ['vt', 'dvdt']:
            gtr_objects = []
            unit_ids = []
            for unit_id, result in results.items():
                #template_type = analysis_options.get("template_type", "dvdt")
                #gtr_path = Path(gtr_dir) / f"{unit_id}_{template_type}" / f"{unit_id}_gtr_object.dill"
                gtr_path = Path(gtr_dir) / template_type / f"{unit_id}_gtr_object.dill"
                if gtr_path.exists():
                    with open(gtr_path, "rb") as f:
                        gtr = dill.load(f)
                        gtr_objects.append(gtr)
                        unit_ids.append(unit_id)

            if gtr_objects:
                fig, ax = plot_all_reconstructions(
                    gtr_objects,
                    unit_ids=unit_ids,
                    cmap_name="tab20",
                    title="All Axon Reconstructions",
                    marker_size=3,
                    alpha=0.9
                )
                fig_path = Path(gtr_dir) / f"all_units_reconstruction_{template_type}.png"
                pdf_path = Path(gtr_dir) / f"all_units_reconstruction_{template_type}.pdf"
                fig.savefig(fig_path, dpi=400)
                with PdfPages(pdf_path) as pdf:
                    pdf.savefig(fig)
                plt.close(fig)
                logger.info(f"Saved multi-unit reconstruction summary to {fig_path}")
            else:
                logger.warning("No GTR objects found to plot global summary.")

    except Exception as e:
        logger.error(f"Failed to create global reconstruction plot: {e}")

    return results

def setup_directories(recon_dir):
    """
    Sets up and returns paths for saving reconstruction, GTR, and analytics data.

    Args:
        recon_dir (str or Path): Base directory for saving outputs.

    Returns:
        Tuple[Path, Path, Path]: Paths to stream_recon_dir, gtr_dir, and analysis_json_dir
    """
    stream_recon_dir = Path(recon_dir)
    stream_recon_dir.mkdir(parents=True, exist_ok=True)

    gtr_dir = stream_recon_dir / 'unit_gtr_objects'
    gtr_dir.mkdir(parents=True, exist_ok=True)

    analysis_json_dir = stream_recon_dir / 'unit_analytics'
    analysis_json_dir.mkdir(parents=True, exist_ok=True)

    return stream_recon_dir, gtr_dir, analysis_json_dir

# main function ===========================================================================
def reconstruct_and_analyze(templates, analysis_options=None, recon_dir=None, logger=None,
                            params=None, n_jobs=None, skip_failed_units=True, unit_limit=None,
                            parallel=True, debug_mode=False, **kwargs):
    """
    Main orchestration function for axon reconstruction and analysis.

    Args:
        templates (dict): Dictionary of all templates per stream/unit.
        analysis_options (dict): Optional analysis settings.
        recon_dir (str or Path): Directory to save reconstructions and analytics.
        logger (logging.Logger): Logger instance.
        params (dict): Parameters for graph axon tracking.
        n_jobs (int): Max parallel jobs.
        unit_limit (int): Limit number of units to process.
        parallel (bool): Whether to run in parallel.
        debug_mode (bool): Forces serial execution for easier debugging.
        **kwargs: Additional options passed downstream.
    """
    #debug_mode = True
    if debug_mode:
        parallel = False

    total_units = sum(
        len(stream['units']) for tmpl in templates.values() for stream in tmpl['streams'].values()
    )
    progress_bar = tqdm(total=total_units, desc="Reconstructing units")
    
    #
    if unit_limit is None:
        unit_limit = total_units

    for dataset_id, dataset in templates.items():
        for stream_id, stream_data in dataset['streams'].items():
            unit_templates = stream_data['units']
            stream_recon_dir, gtr_dir, analysis_json_dir = setup_directories(recon_dir)

            results = process_units(
                unit_templates, gtr_dir, analysis_options, params,
                unit_limit, analysis_json_dir, logger, parallel, n_jobs
            )

            try:
                aggregate_analytics(results, stream_recon_dir, stream_id)
            except Exception as e:
                logger.error(f"Error aggregating analytics for {stream_id}: {e}")
            
            progress_bar.update(len(results))

    progress_bar.close()
    logger.info("Reconstruction and analysis complete.")