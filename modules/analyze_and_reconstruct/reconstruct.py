import sys
import os
import dill
import matplotlib.pyplot as plt

'''Define utility functions.'''
def get_git_root():
    """
    Retrieve the root directory of the git repository.
    
    Returns:
    str: The root directory of the git repository.
    """
    git_root = os.popen('git rev-parse --show-toplevel').read().strip()
    return git_root

def load_dill_file(dill_path):
    """
    Load a dill file from the specified path.
    
    Parameters:
    dill_path (str): The path to the dill file.
    
    Returns:
    object: The data loaded from the dill file.
    """
    with open(dill_path, 'rb') as f:
        data = dill.load(f)
    return data

def save_figure(fig, path):
    """
    Save the figure to the specified path.
    
    Parameters:
    fig (matplotlib.figure.Figure): The figure to save.
    path (str): The path where the figure will be saved.
    """
    fig.savefig(path)
    print(f"Figure saved to {path}")

def plot_clean_branches(gtr, fig_path, plot_full_template=True, fresh_plots=False, show_plot=True):
    """
    Plot clean branches and save the figure.
    
    Parameters:
    gtr (object): The object containing the plot_clean_branches method.
    fig_path (str): The path where the figure will be saved.
    plot_full_template (bool): Whether to plot the full template.
    fresh_plots (bool): Whether to generate fresh plots.
    """
    fig_graph = plt.figure(figsize=(12, 7))
    ax_raw = fig_graph.add_subplot(111)
    axpaths_raw = ax_raw

    if os.path.exists(fig_path) and not fresh_plots:
        print(f"Figure already exists at {fig_path} and fresh_plots is False. Skipping plot generation.")
    else:
        axpaths_raw = gtr.plot_clean_branches(
            plot_full_template=plot_full_template, ax=axpaths_raw, cmap="rainbow",
            plot_bp=True, branch_colors=None
        )
        axpaths_raw.legend(fontsize=6)
        axpaths_raw.set_title("Clean Branches")
        save_figure(fig_graph, fig_path)
        if show_plot:
            plt.show()

def generate_reconstructions_for_all_units(base_dir, plot_full_template=True, plot_focused_template=True, fresh_plots=False, show_plot=False):
    """
    Generate reconstructions for all unit dill files in a directory recursively.
    
    Parameters:
    base_dir (str): The base directory to search for dill files.
    plot_full_template (bool): Whether to plot the full template.
    fresh_plots (bool): Whether to generate fresh plots.
    """
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith('_gtr_object.dill'):
                unit_name = file.split('_gtr_object.dill')[0]
                dill_path = os.path.join(root, file)
                gtr = load_dill_file(dill_path)
                
                recon_dir = os.path.join(root, 'reconstructions')
                os.makedirs(recon_dir, exist_ok=True)
                
                if plot_full_template:
                    fig_path = os.path.join(recon_dir, f'{unit_name}_Reconstruction_full.pdf')
                    plot_clean_branches(gtr, fig_path, plot_full_template=True, fresh_plots=fresh_plots, show_plot=show_plot)
                
                if plot_focused_template:
                    fig_path = os.path.join(recon_dir, f'{unit_name}_Reconstruction_focused.pdf')
                    plot_clean_branches(gtr, fig_path, plot_full_template=False, fresh_plots=fresh_plots, show_plot=show_plot)

def plot_template_propagation(gtr, fig_dir, unit_id, title=None, fresh_plots=False, figsize=(8, 40), 
                              linewidth=1, markerfacecolor='r', markersize=5, dpi=300, 
                              max_overlapped_lines=1, color_marker='none', marker='.'):
    """
    Plot template propagation and save the figure.
    
    Parameters:
    gtr (object): The object containing the necessary attributes and methods.
    fig_dir (str): The directory where the figure will be saved.
    unit_id (int): The unit ID.
    title (str): The title of the plot.
    fresh_plots (bool): Whether to generate fresh plots.
    figsize (tuple): The size of the figure.
    linewidth (int): The width of the lines.
    markerfacecolor (str): The face color of the markers.
    markersize (int): The size of the markers.
    dpi (int): The resolution of the figure.
    max_overlapped_lines (int): The maximum number of overlapped lines.
    color_marker (str): The color of the markers.
    marker (str): The marker style.
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    fig_path = fig_dir

    if os.path.exists(fig_path) and not fresh_plots:
        print(f"Figure already exists at {fig_path} and fresh_plots is False. Skipping plot generation.")
        return

    template = gtr.template
    selected_channels = gtr.selected_channels
    locations = gtr.locations
    ax = av.plotting.plot_template_propagation(
        template, locations, selected_channels, sort_templates=True,
        color=None, color_marker=color_marker, ax=ax, 
        #linewidth=linewidth, #TODO: Implement all of these parameters in the axon_velocity function
        #markerfacecolor=markerfacecolor, 
        #markersize=markersize, 
        #max_overlapped_lines=max_overlapped_lines, 
        #marker=marker
    )
    ax.set_title(f'Template Propagation for Unit {unit_id} {title}', fontsize=16)
    save_figure(fig, fig_path)

def generate_propagation_plots_for_all_units(base_dir, title=None, fresh_plots=False, figsize=(8, 40), 
                                             linewidth=1, markerfacecolor='r', markersize=5, dpi=300, 
                                             max_overlapped_lines=1, color_marker='none', marker='.', show_plot=False):
    """
    Generate propagation plots for all unit dill files in a directory recursively.
    
    Parameters:
    base_dir (str): The base directory to search for dill files.
    title (str): The title of the plot.
    fresh_plots (bool): Whether to generate fresh plots.
    figsize (tuple): The size of the figure.
    linewidth (int): The width of the lines.
    markerfacecolor (str): The face color of the markers.
    markersize (int): The size of the markers.
    dpi (int): The resolution of the figure.
    max_overlapped_lines (int): The maximum number of overlapped lines.
    color_marker (str): The color of the markers.
    marker (str): The marker style.
    """
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith('_gtr_object.dill'):
                unit_name = file.split('_gtr_object.dill')[0]
                dill_path = os.path.join(root, file)
                gtr = load_dill_file(dill_path)
                
                recon_dir = os.path.join(root, 'propogations_plots')
                os.makedirs(recon_dir, exist_ok=True)
                
                fig_path = os.path.join(recon_dir, f'{unit_name}_Template_Propagation.pdf')
                plot_template_propagation(gtr, fig_path, unit_name, title, fresh_plots, figsize, 
                                          linewidth, 
                                          markerfacecolor, markersize, dpi, max_overlapped_lines, color_marker, marker)
                #TODO: Implement show plot in this axon_velocity function...or maybe just copy it and tweak it within my code idk. It's not something I need to push to everyone using the package right?


                
# def generate_propagation_plots_for_all_units(base_dir, fresh_plots=False, figsize=(8, 40), 
#                                              linewidth=1, markerfacecolor='r', markersize=5, dpi=300, 
#                                              max_overlapped_lines=1, color_marker='none', marker='.', show_plot=False):
#     """
#     Generate propagation plots for all unit dill files in a directory recursively.
    
#     Parameters:
#     base_dir (str): The base directory to search for dill files.
#     title (str): The title of the plot.
#     fresh_plots (bool): Whether to generate fresh plots.
#     figsize (tuple): The size of the figure.
#     linewidth (int): The width of the lines.
#     markerfacecolor (str): The face color of the markers.
#     markersize (int): The size of the markers.
#     dpi (int): The resolution of the figure.
#     max_overlapped_lines (int): The maximum number of overlapped lines.
#     color_marker (str): The color of the markers.
#     marker (str): The marker style.
#     """
#     for root, _, files in os.walk(base_dir):
#         for file in files:
#             if file.endswith('_gtr_object.dill'):
#                 unit_name = file.split('_gtr_object.dill')[0]
#                 dill_path = os.path.join(root, file)
#                 gtr = load_dill_file(dill_path)
                
#                 recon_dir = os.path.join(root, 'propogations_plots')
#                 os.makedirs(recon_dir, exist_ok=True)
                
#                 fig_path = os.path.join(recon_dir, f'{unit_name}_Template_Propagation.pdf')
#                 plot_template_propagation(gtr, fig_path, unit_name, fresh_plots, figsize, linewidth, markerfacecolor, markersize, dpi, max_overlapped_lines, color_marker, marker) 
#                 #TODO: Implement show plot in this axon_velocity function...or maybe just copy it and tweak it within my code idk. It's not something I need to push to everyone using the package right?

'''Import necessary modules and functions from the git repository.'''
# Ensure the git root is added to the system path
git_root = get_git_root()
sys.path.insert(0, git_root)
from modules.analyze_and_reconstruct.lib_analysis_functions import *
from submodules.axon_velocity_fork import axon_velocity as av