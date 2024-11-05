# analyze.py

import numpy as np
import pandas as pd
from pathlib import Path
from modules.lib_axon_velocity_functions import (
    plot_template_wrapper, 
    generate_amplitude_map, 
    generate_peak_latency_map, 
    generate_axon_analytics
)
import logging
import numpy as np
from pathlib import Path
from scipy.stats import kurtosis, skew
import logging
from modules.lib_axon_velocity_functions import (
    plot_template_wrapper, 
    generate_amplitude_map, 
    generate_peak_latency_map, 
    calculate_snr
)

# Define modular functions for individual analytic calculations
def calculate_num_channels(unit_templates):
    """Calculate the number of channels included."""
    return len(unit_templates['merged_channel_loc'])

def calculate_channel_density(unit_templates):
    """Calculate the channel density per unit area."""
    x_coords = unit_templates['merged_channel_loc'][:, 0]
    y_coords = unit_templates['merged_channel_loc'][:, 1]
    area = (np.max(x_coords) - np.min(x_coords)) * (np.max(y_coords) - np.min(y_coords))
    return len(unit_templates['merged_channel_loc']) / area

def calculate_spatial_extent(unit_templates):
    """Calculate spatial extent in terms of bounding box (width, height) and area."""
    x_coords = unit_templates['merged_channel_loc'][:, 0]
    y_coords = unit_templates['merged_channel_loc'][:, 1]
    width = np.max(x_coords) - np.min(x_coords)
    height = np.max(y_coords) - np.min(y_coords)
    area = width * height
    return (width, height), area

def calculate_signal_extremums(template):
    """Calculate the signal extremums (range between max and min)."""
    return np.max(template) - np.min(template)

def calculate_peak_amplitude(template):
    """Calculate the peak amplitude of the template."""
    return np.max(np.abs(template))

def calculate_average_amplitude(template):
    """Calculate the average amplitude of the template."""
    return np.mean(np.abs(template))

def calculate_variance(template):
    """Calculate the variance of the template signal."""
    return np.var(template)

def calculate_kurtosis(template):
    """Calculate kurtosis of the template."""
    return kurtosis(template, fisher=True)

def calculate_skewness(template):
    """Calculate skewness of the template."""
    return skew(template)

# Main function to process analytics for a unit
def process_unit_for_analysis(unit_id, unit_templates, recon_dir, analysis_options, logger=None):
    if logger:
        logger.info(f'Analyzing unit {unit_id}')
    else:
        print(f'Analyzing unit {unit_id}')
        
    # Primary dictionaries to hold template data and analytics
    template_data = {
        'unit_id': unit_id,
        'template_segments': unit_templates.get('template_segments'),
        'vt_template': unit_templates.get('merged_template'),
        'dvdt_template': unit_templates.get('dvdt_merged_template'),
        'milos_template': []  # Placeholder for milos_template if applicable #TODO: Implement milos_template
    }

    # Dictionary for template analytics initialized with function references
    # TODO: Finish implementing the rest of the analytics
    template_analytics = {
        'num_channels_included': calculate_num_channels(unit_templates),
        'channel_density': calculate_channel_density(unit_templates),
        'spatial_extent': calculate_spatial_extent(unit_templates),
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
        'area': calculate_spatial_extent(unit_templates)[1]
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
    analytics_list = []

    for key, tmps in templates.items():
        for stream_id, stream_templates in tmps['streams'].items():
            unit_templates = stream_templates['units']
            recon_dir = Path(recon_dir) / tmps['date'] / tmps['chip_id'] / tmps['scanType'] / tmps['run_id'] / stream_id

            for unit_id, unit_template in unit_templates.items():
                result = process_unit_for_analysis(unit_id, unit_template, recon_dir, analysis_options, logger)
                if result:
                    analytics_list.append(result)

    # Build and save the DataFrame of all analytics
    analytics_df = build_analytics_dataframe(analytics_list)
    analytics_df.to_csv(recon_dir / "axon_analytics.csv", index=False)
    if logger:
        logger.info(f"Saved analytics data to {recon_dir / 'axon_analytics.csv'}")

