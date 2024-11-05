# analyze.py
import numpy as np
import pandas as pd
from pathlib import Path
# from modules.lib_axon_velocity_functions import (
#     plot_template_wrapper, 
#     generate_amplitude_map, 
#     generate_peak_latency_map, 
#     generate_axon_analytics
# )
import logging
import numpy as np
from pathlib import Path
from scipy.stats import kurtosis, skew
import logging
# from modules.lib_axon_velocity_functions import (
#     plot_template_wrapper, 
#     generate_amplitude_map, 
#     generate_peak_latency_map, 
#     calculate_snr
# )

# Define modular functions for individual analytic calculations
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

def get_git_root():
    git_root = os.popen('git rev-parse --show-toplevel').read().strip()
    return git_root

def transform_data(merged_template, merged_channel_loc, merged_template_filled=None, merged_channel_filled_loc=None):
    transformed_template = merged_template.T  # array with time samples and amplitude values (voltage or v/s)
    if merged_template_filled is not None: 
        transformed_template_filled = merged_template_filled.T  # array with time samples and amplitude values (voltage or v/s)
    else: 
        transformed_template_filled = None
    trans_loc = np.array([[loc[0], loc[1] * -1] for loc in merged_channel_loc])
    if merged_channel_filled_loc is not None: 
        trans_loc_filled = np.array([[loc[0], loc[1] * -1] for loc in merged_channel_filled_loc])
    else: 
        trans_loc_filled = None
    return transformed_template, transformed_template_filled, trans_loc, trans_loc_filled

import time
def build_graph_axon_tracking(template_data, params=None):
    """Build the graph for axon tracking."""
    global gtr
    from submodules.axon_velocity_fork.axon_velocity.axon_velocity import GraphAxonTracking
    from submodules.axon_velocity_fork.axon_velocity import get_default_graph_velocity_params
    transposed_template, _, transposed_loc, _ = transform_data(template_data['dvdt_template'], template_data['channel_loc'])
    if params is None:
        params = get_default_graph_velocity_params()
    
    #start timer
    start = time.time()
    print("Building graph...")
    gtr = GraphAxonTracking(transposed_template, transposed_loc, 10000, **params)
    gtr._verbose = 1
    gtr.build_graph()
    gtr.find_paths()
    print(f"Graph built in {time.time() - start} seconds")
    
    # Save gtr using dill
    import dill
    with open('gtr_object.dill', 'wb') as f:
        dill.dump(gtr, f)
    return gtr

# Main function to process analytics for a unit
def process_unit_for_analysis(unit_id, unit_templates, recon_dir, analysis_options, logger=None):
    from submodules.axon_velocity_fork.axon_velocity.axon_velocity import GraphAxonTracking
    
    if logger:
        logger.info(f'Analyzing unit {unit_id}')
    else:
        print(f'Analyzing unit {unit_id}')
        
    # Primary dictionaries to hold template data and analytics
    template_data = {
        'unit_id': unit_id,
        'template_segments': unit_templates.get('template_segments', []),
        'channel_loc': unit_templates.get('channels_loc'),
        'channel_loc_filled': unit_templates.get('channels_loc_filled'),
        'vt_template': unit_templates.get('merged_template'),
        'dvdt_template': unit_templates.get('dvdt'),
        'filled_vt_template': unit_templates.get('merged_template_filled'),
        'filled_dvdt_template': unit_templates.get('dvdt_filled'),
        'milos_template': []  # Placeholder for milos_template if applicable #TODO: Implement milos_template
    }

    # Dictionary for template analytics initialized with function references
    # TODO: Finish implementing the rest of the analytics
    template_analytics = {
        'num_channels_included': calculate_num_channels(template_data),
        'channel_density': calculate_channel_density(template_data),
        'gtr_object': build_graph_axon_tracking(template_data),
        'branches': gtr.branches,
        #'gtr_object': GraphAxonTracking(transformed_template, trans_loc, 10000, **params),
        #'branch_channel_density': calculate_branch_channel_density(template_data),
        #'spatial_extent': calculate_spatial_extent(unit_templates),
        'extremums': [],
        'peak_amplitude': [],
        'average_amplitude': [],
        'snr': [],
        'variance': [],
        'kurtosis': [],
        'skewness': [],
        'velocities': [],
        'path_lengths': [],
        'r2s': [],
        'latency': [],
        'init_chans': [],
        #'area': calculate_spatial_extent(unit_templates)[1]
    }
    
    return template_data, template_analytics

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

def analyze(templates, analysis_options=None, recon_dir=None, logger=None, **kwargs):
    #TODO: Bring waveforms in here and add it to template data
    analytics_list = []

    for key, tmps in templates.items():
        for stream_id, stream_templates in tmps['streams'].items():
            unit_templates = stream_templates['units']
            if recon_dir is None: recon_dir = Path(recon_dir) / tmps['date'] / tmps['chip_id'] / tmps['scanType'] / tmps['run_id'] / stream_id
            else: 
                #create recon_dir if it doesn't exist
                if not Path(recon_dir).exists():
                    Path(recon_dir).mkdir(parents=True, exist_ok=True)
                recon_dir = Path(recon_dir)
            
            for unit_id, unit_template in unit_templates.items():
                result = process_unit_for_analysis(unit_id, unit_template, recon_dir, analysis_options, logger)
                if result:
                    analytics_list.append(result)

    # Build and save the DataFrame of all analytics
    analytics_df = build_analytics_dataframe(analytics_list)
    analytics_df.to_csv(recon_dir / "axon_analytics.csv", index=False)
    if logger:
        logger.info(f"Saved analytics data to {recon_dir / 'axon_analytics.csv'}")
        
import numpy as np
import os
from pathlib import Path
import logging

def load_templates_from_directory(directory):
    templates = {'templates': {'streams': {'well000': {'units': {}}}}}
    for file in os.listdir(directory):
        if file.endswith('.npy'):
            unit_id = file.split('_')[0] if '_' in file else file.split('.')[0]
            if unit_id not in templates['templates']['streams']['well000']['units']:
                templates['templates']['streams']['well000']['units'][unit_id] = {}
                #     'merged_template': None,
                #     'merged_channel_loc': None,
                #     'template_segments': None,
                #     'dvdt_merged_template': None
                # }
            array = np.load(os.path.join(directory, file))
            if 'channels_filled' in file:
                templates['templates']['streams']['well000']['units'][unit_id]['channels_loc_filled'] = array
            elif 'channels' in file:
                templates['templates']['streams']['well000']['units'][unit_id]['channels_loc'] = array
            elif 'dvdt_filled' in file:
                templates['templates']['streams']['well000']['units'][unit_id]['dvdt_filled'] = array
            elif 'dvdt' in file:
                templates['templates']['streams']['well000']['units'][unit_id]['dvdt'] = array
            else:
                templates['templates']['streams']['well000']['units'][unit_id]['merged_template'] = array
    return templates

import sys
import os
def main():
    git_root = get_git_root()
    sys.path.insert(0, git_root)

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Define the directory containing the .npy files
    template_dir = "/home/adamm/workspace/RBS_axonal_reconstructions/development_scripts/templates_04Nov2024/241021/M06804/AxonTracking/000091/well000"

    # Load templates from the directory
    templates = load_templates_from_directory(template_dir)

    # Define analysis options if any
    analysis_options = {
        # Add any specific options for analysis here
    }

    # Define reconstruction directory
    recon_dir = "/home/adamm/workspace/RBS_axonal_reconstructions/development_scripts/templates_04Nov2024/241021/M06804/AxonTracking/000091/well000_recon"

    # Call the analyze function
    analyze(templates, analysis_options=analysis_options, recon_dir=recon_dir, logger=logger)

if __name__ == "__main__":
    main()