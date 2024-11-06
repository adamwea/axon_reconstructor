# analyze.py

import logging
import numpy as np
from scipy.stats import kurtosis, skew
from pathlib import Path
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import dill
import pandas as pd

'''Imports'''
import os
import sys

'''Local Imports'''
def get_git_root():
    git_root = os.popen('git rev-parse --show-toplevel').read().strip()
    return git_root
git_root = get_git_root()
sys.path.insert(0, git_root)
from modules.analyze_and_reconstruct.lib_analysis_functions import *

''' Main Functions '''
# Main function to process analytics for a unit
def process_unit_for_analysis(unit_id, unit_templates, recon_dir, analysis_options, params, logger=None):
    start = time.time()
    print(f'Analyzing unit {unit_id}...')
    # if logger:
    #     logger.info(f'Analyzing unit {unit_id}')
    # else:
    #     print(f'Analyzing unit {unit_id}')
    
    '''Input Data for Reconstruction and Analysis'''    
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

    '''Reconstruction'''
    # Dictionary for template analytics initialized with function references
    try: gtr, av_params = build_graph_axon_tracking(template_data, recon_dir=recon_dir, load_gtr=True, params=params)
    except:
        try: gtr, av_params = build_graph_axon_tracking(template_data, recon_dir=recon_dir, load_gtr=False, params=params)
        except Exception as e:
            if logger:
                logger.error(f"Failed to build graph for unit {unit_id} - Error: {e}")
            else:
                print(f"Failed to build graph for unit {unit_id} - Error: {e}")
            return None, None, None
    
    # Reconstruction data dictionary
    reconstruction_data = {
        'gtr': gtr,
        'av_params': av_params,
        'branches': gtr.branches,
        'branch_points': gtr._branching_points
    }

    '''Analysis'''   
    # Template analysis
    reconstruction_analytics = {
        'channels_in_template': len(template_data['channel_loc']),  # Number of channels in the template
        'template_rect_area': calculate_template_rectangular_area(template_data),  # Area of the template, in um^2
        'template_density': len(template_data['channel_loc']) / calculate_template_rectangular_area(template_data),  # Channel density of the template, in channels/um^2
        
        # Axon analysis
        'axon_length': gtr.compute_path_length(gtr.branches[0]['channels']),  # Length of the axon, in um
        
        # Branching analysis
        'num_branches': len(gtr.branches),  # Number of branches in the reconstruction
        'branch_lengths': [gtr.compute_path_length(branch['channels']) for branch in gtr.branches],  # Length of each branch, in um
        'branch_channel_density': [gtr.compute_path_length(branch['channels']) / len(branch['channels']) for branch in gtr.branches],  # Channel density of each branch, in channels/um^2
        'branch_orders': {index: calculate_branch_order(branch['channels'], gtr._branching_points) for index, branch in enumerate(gtr.branches)},  # Branch order of each branch
        'maximum_branch_order': max([calculate_branch_order(branch['channels'], gtr._branching_points) for branch in gtr.branches]),  # Maximum branch order in the reconstruction
        # 'partition_asymmetry': calculate_partition_asymmetry(gtr.branches, gtr._branching_points),  # Partition asymmetry of the reconstruction #TODO: Implement partition_asymmetry
        
        # Signal Propagation Analysis
        'velocities': {index: branch['velocity'] for index, branch in enumerate(gtr.branches)}  # Velocity of each branch, in um/s
        # 'velocities': [branch['velocity'] for branch in gtr.branches],  # Velocity of each branch, in um/s
    }
    print(f'Finished analyzing unit {unit_id} in {time.time() - start} seconds')
    return template_data, reconstruction_data, reconstruction_analytics

def analysis_results_to_dataframe(analysis_results):
    """
    Convert the analysis results dictionary into a pandas DataFrame.

    Parameters:
    analysis_results (dict): The dictionary containing analysis results.

    Returns:
    pd.DataFrame: A DataFrame containing the analysis results.
    """
    flattened_data = []

    for unit_id, results in analysis_results.items():
        flattened_entry = {'unit_id': unit_id}
        
        # # Flatten template data
        # for key, value in results['template_data'].items():
        #     flattened_entry[f'template_{key}'] = value
        
        # # Flatten reconstruction data
        # for key, value in results['reconstruction_data'].items():
        #     flattened_entry[f'reconstruction_{key}'] = value
        
        # Flatten reconstruction analytics
        for key, value in results['reconstruction_analytics'].items():
            flattened_entry[f'analytics_{key}'] = value
        
        flattened_data.append(flattened_entry)
    
    df = pd.DataFrame(flattened_data)
    return df

# TODO: add waveforms to the template data
def analyze(templates, analysis_options=None, recon_dir=None, logger=None, params=None, max_workers=10, **kwargs):
    # Define the ThreadPoolExecutor with a specified number of workers
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for key, tmps in templates.items():
            for stream_id, stream_templates in tmps['streams'].items():
                unit_templates = stream_templates['units']
                stream_results = {}  # Separate results for each stream

                # Set up the stream-specific reconstruction directory
                if recon_dir is None: 
                    stream_recon_dir = Path(recon_dir) / tmps['date'] / tmps['chip_id'] / tmps['scanType'] / tmps['run_id'] / stream_id
                else: 
                    stream_recon_dir = Path(recon_dir)
                if not stream_recon_dir.exists():
                    stream_recon_dir.mkdir(parents=True, exist_ok=True)

                # Submit each unit processing task
                futures = {
                    executor.submit(
                        process_unit_for_analysis,
                        unit_id, unit_template, stream_recon_dir, analysis_options, params, logger
                    ): unit_id for unit_id, unit_template in unit_templates.items()
                }

                # Process results as they complete
                for future in as_completed(futures):
                    unit_id = futures[future]
                    try:
                        template_data, reconstruction_data, reconstruction_analytics = future.result()
                        if template_data is not None:
                            stream_results[unit_id] = {
                                'template_data': template_data,
                                'reconstruction_data': reconstruction_data,
                                'reconstruction_analytics': reconstruction_analytics
                            }
                            
                            # Save each unit's result individually
                            with open(stream_recon_dir / f"{unit_id}_analysis_results.dill", 'wb') as f:
                                dill.dump(stream_results[unit_id], f)
                            print(f'Processed and saved unit {unit_id}')
                    except Exception as e:
                        if logger:
                            logger.error(f"Error processing unit {unit_id}: {e}")
                        else:
                            print(f"Error processing unit {unit_id}: {e}")

                # Convert the stream results to a DataFrame and save for the stream
                df = analysis_results_to_dataframe(stream_results)
                df.to_csv(stream_recon_dir / f"{stream_id}_axon_analytics.csv", index=False)
                print(f'Saved analytics data for stream {stream_id} to {stream_recon_dir / f"{stream_id}_axon_analytics.csv"}')

        
''' Debug Helper Functions '''
def load_templates_from_directory(directory):
    templates = {'templates': {'streams': {'well000': {'units': {}}}}}
    for file in os.listdir(directory):
        if file.endswith('.npy'):
            unit_id = file.split('_')[0] if '_' in file else file.split('.')[0]
            if unit_id not in templates['templates']['streams']['well000']['units']:
                templates['templates']['streams']['well000']['units'][unit_id] = {}
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

def main():
    git_root = get_git_root()
    sys.path.insert(0, git_root)

    # Set up logging
    # logging.basicConfig(level=logging.INFO)
    # logger = logging.getLogger(__name__)

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

    #get params
    params = get_default_graph_velocity_params()
    params['r2_threshold'] = 0.5    
    
    # Call the analyze function
    analyze(templates, analysis_options=analysis_options, recon_dir=recon_dir, params=params)

if __name__ == "__main__":
    main()