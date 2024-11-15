# Define modular functions for individual analytic calculations
import numpy as np
from scipy.stats import kurtosis, skew
import pandas as pd
import time
from pathlib import Path
from submodules.axon_velocity_fork.axon_velocity.axon_velocity import GraphAxonTracking
from submodules.axon_velocity_fork.axon_velocity import get_default_graph_velocity_params
import dill

def transform_data(merged_template, merged_channel_loc, merged_template_filled=None, merged_channel_filled_loc=None):
    transformed_template = merged_template.T  # array with time samples and amplitude values (voltage or v/s)
    trans_loc = np.array([[loc[0], loc[1] * -1] for loc in merged_channel_loc])
    
    # if merged_template_filled is not None: 
    #     transformed_template_filled = merged_template_filled.T  # array with time samples and amplitude values (voltage or v/s)
    # else: 
    #     transformed_template_filled = None
    # if merged_channel_filled_loc is not None: 
    #     trans_loc_filled = np.array([[loc[0], loc[1] * -1] for loc in merged_channel_filled_loc])
    # else: 
    #     trans_loc_filled = None
    return transformed_template, trans_loc

def build_graph_axon_tracking(template_data, load_gtr=False, recon_dir=None, params=None, template_type='dvt'):
    gtr = None
    """Build the graph for axon tracking."""
    
    if template_type == 'vt':
        template_key = 'vt_template'
    elif template_type == 'dvt':
        template_key = 'dvdt_template'
    elif template_type == 'milos':
        template_key = 'milos_template'
    else:
        raise ValueError("Invalid template type. Choose from 'vt', 'dvt', or 'milos'.")
    
    transposed_template, transposed_loc = transform_data(template_data[template_key], template_data['channel_locations'])
    params = params or get_default_graph_velocity_params() # Default parameters for graph building if not provided
    unit_id = template_data['unit_id']
    
    # if recon_dir is not None, try to load the gtr object from file    
    if recon_dir:
        recon_dir = Path(recon_dir)
        recon_dir.mkdir(parents=True, exist_ok=True)
        gtr_file = recon_dir / f"{unit_id}_gtr_object.dill"
        
        if load_gtr and gtr_file.exists():
            assert gtr_file.exists() or not load_gtr, f"File {gtr_file} does not exist. Set load_gtr to False to build the graph."
            with open(gtr_file, 'rb') as f:
                gtr = dill.load(f)
                # return gtr, params
    
    # Build the graph
    start = time.time()
    if gtr is None:
        # print("Building graph...")
        gtr = GraphAxonTracking(transposed_template, transposed_loc, 10000, **params)
        gtr._verbose = 1 # TODO: make this an option later
        gtr.select_channels()
        gtr.build_graph()
        gtr.find_paths()
        gtr.clean_paths(remove_outliers=False)
        # print(f"Graph built in {time.time() - start} seconds")
    elif gtr is not None:
        print("Graph already exists. Updating analysis...")
        # gtr._verbose = 1
        # gtr.select_channels()
        # gtr.build_graph()
        # gtr.find_paths()
        # gtr.clean_paths(remove_outliers=False)
        print(f"Graph updated in {time.time() - start} seconds")
           
    return gtr, params

def calculate_template_rectangular_area(template_data):
    """
    Calculate the rectangular 2D area occupied by the channels in the template data.

    Parameters:
    template_data (dict): A dictionary containing channel locations under the key 'channel_loc'.

    Returns:
    int: The number of channels in the template.
    float: The rectangular 2D area occupied by the channels in square micrometers.
    """
    channels_in_template = len(template_data['channel_locations'])

    # Extract x and y coordinates of all channels
    x_coords = [loc[0] for loc in template_data['channel_locations']]
    y_coords = [loc[1] for loc in template_data['channel_locations']]

    # Calculate the minimum and maximum x and y coordinates
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)

    # Calculate the width and height of the rectangular area
    width = max_x - min_x
    height = max_y - min_y

    # Calculate the rectangular 2D area
    rectangular_area = width * height

    return rectangular_area

# Calculate the branch order for each branch
def calculate_branch_order(branch, branch_points):
    order = 0
    for point in branch:
        if point in branch_points:
            order += 1
    return order

def calculate_partition_asymmetry(branches, branch_points):
    """
    Calculate the partition asymmetry for each bifurcation point in the branches.

    Parameters:
    branches (list): A list of branches, where each branch is a list of points.
    branch_points (list): A list of branching points in the reconstruction.

    Returns:
    dict: A dictionary mapping each bifurcation point to its partition asymmetry value.
    """
    def count_terminals(branch, branches):
        """
        Count the number of terminal points in a subtree.
        """
        terminals = 0
        for point in branch:
            if point not in branch_points:
                terminals += 1
            else:
                # Recursively count terminals in the sub-branches
                sub_branches = [b for b in branches if b[0] == point]
                for sub_branch in sub_branches:
                    terminals += count_terminals(sub_branch, branches)
        return terminals

    partition_asymmetry = {}
    for point in branch_points:
        # Find the branches that start at the bifurcation point
        sub_branches = [b for b in branches if b[0] == point]
        if len(sub_branches) >= 2:
            # The first sub-branch continues the original branch
            original_branch = [b for b in branches if point in b][0]
            sub_branch_1 = original_branch[original_branch.index(point) + 1:]
            sub_branch_2 = sub_branches[1]

            n1 = count_terminals(sub_branch_1, branches)
            n2 = count_terminals(sub_branch_2, branches)
            Ap = abs(n1 - n2) / (n1 + n2 - 2)
            partition_asymmetry[point] = Ap

    return partition_asymmetry

# # Example usage
# partition_asymmetry = calculate_partition_asymmetry(branches, branch_points)
# print(f"Partition asymmetry: {partition_asymmetry}")

''' Functions above this point have been tested and verified to work. '''

def build_analytics_dataframe(analytics_list):
    # Convert list of dictionaries to a DataFrame
    df = pd.DataFrame(analytics_list)
    return df

def save_results(stream_id, all_analytics, recon_dir, logger=None):
    from modules.lib_axon_velocity_functions import save_axon_analytics
    for key, data in all_analytics.items():
        suffix = key.split('_')[-1] if '_' in key else 'template'
        if 'dvdt' in key:
            suffix = 'dvdt'
        save_axon_analytics(
            stream_id, data['units'], data['extremums'], data['branch_ids'], data['velocities'], 
            data['path_lengths'], data['r2s'], data['num_channels_included'], data['channel_density'], data['init_chans'],
            recon_dir, suffix=f"_{suffix}"
        )
def calculate_num_channels(template_data):
    """Calculate the number of channels included."""
    return len(template_data['vt_template'])

def calculate_channel_density(template_data):
    """Calculate the channel density per unit area."""
    x_coords = template_data['vt_template'][:, 0]
    y_coords = template_data['vt_template'][:, 1]
    area = (np.max(x_coords) - np.min(x_coords)) * (np.max(y_coords) - np.min(y_coords))
    return len(template_data['vt_template']) / area

def calculate_spatial_extent(template_data):
    """Calculate spatial extent in terms of bounding box (width, height) and area."""
    x_coords = template_data['vt_template'][:, 0]
    y_coords = template_data['vt_template'][:, 1]
    width = np.max(x_coords) - np.min(x_coords)
    height = np.max(y_coords) - np.min(y_coords)
    area = width * height
    return (width, height), area

def calculate_signal_extremums(template_data):
    """Calculate the signal extremums (range between max and min)."""
    return np.max(template_data['vt_template']) - np.min(template_data['vt_template'])

def calculate_peak_amplitude(template_data):
    """Calculate the peak amplitude of the template."""
    return np.max(np.abs(template_data['vt_template']))

def calculate_average_amplitude(template_data):
    """Calculate the average amplitude of the template."""
    return np.mean(np.abs(template_data['vt_template']))

def calculate_variance(template_data):
    """Calculate the variance of the template signal."""
    return np.var(template_data['vt_template'])

def calculate_kurtosis(template_data):
    """Calculate kurtosis of the template."""
    return kurtosis(template_data['vt_template'], fisher=True)

def calculate_skewness(template_data):
    """Calculate skewness of the template."""
    return skew(template_data['vt_template'])

# def calculate_branch_channel_density(template_data):
#     """Calculate the channel density per branch."""
#     import axon_velocity