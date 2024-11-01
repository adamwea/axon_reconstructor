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

def process_unit_for_analysis(unit_id, unit_templates, recon_dir, analysis_options, logger=None):
    if logger:
        logger.info(f'Analyzing unit {unit_id}')
    else:
        print(f'Analyzing unit {unit_id}')

    # Prepare data for analytics
    templates = {
        'merged_template': unit_templates['merged_template'],
        'dvdt_merged_template': unit_templates['dvdt_merged_template'],
    }

    analytics_data = {
        'unit_id': unit_id,
        'template_type': [], 
        'num_channels_included': [], 
        'channel_density': [], 
        'extremums': [], 
        'velocities': [], 
        'path_lengths': [], 
        'r2s': [], 
        'init_chans': []
    }

    x_coords = unit_templates['merged_channel_loc'][:, 0]
    y_coords = unit_templates['merged_channel_loc'][:, 1]
    width = np.max(x_coords) - np.min(x_coords)
    height = np.max(y_coords) - np.min(y_coords)
    area = width * height
    channel_density_value = len(unit_templates['merged_channel_loc']) / area  # channels / um^2

    for key, template in templates.items():
        if template is not None:
            # Append data to analytics_data dict
            analytics_data['template_type'].append(key)
            analytics_data['num_channels_included'].append(len(unit_templates['merged_channel_loc']))
            analytics_data['channel_density'].append(channel_density_value)

            # Template Plot
            try:
                plot_dir = Path(recon_dir) / f"{unit_id}_{key}_plots"
                plot_dir.mkdir(parents=True, exist_ok=True)
                kwargs = {
                    'save_path': plot_dir / f"template_plot_{unit_id}_{key}.png",
                    'title': f'Template {unit_id} {key}',
                    'fresh_plots': True,
                    'template': template,
                    'locations': unit_templates['merged_channel_loc'],
                    'lw': 0.1, # line width
                }
                plot_template_wrapper(**kwargs)
            except Exception as e:
                if logger: 
                    logger.info(f"Unit {unit_id}_{key} failed to plot template, error: {e}")

            # Amplitude Map
            try:
                generate_amplitude_map(template, unit_templates['merged_channel_loc'], plot_dir, title=f'{key}_amplitude_map', fresh_plots=True, log=False, cmap='terrain')
            except Exception as e:
                if logger:
                    logger.info(f"Unit {unit_id}_{key} failed to generate amplitude map, error: {e}")

            # Peak Latency Map
            try:
                generate_peak_latency_map(template, unit_templates['merged_channel_loc'], plot_dir, title=f'{key}_peak_latency_map', fresh_plots=True, log=False, cmap='terrain')
            except Exception as e:
                if logger:
                    logger.info(f"Unit {unit_id}_{key} failed to generate peak latency map, error: {e}")

    return analytics_data

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

