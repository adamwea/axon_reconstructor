import logging
import numpy as np
from scipy.stats import kurtosis, skew
from pathlib import Path
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import dill
import pandas as pd
from tqdm import tqdm
import json
from modules.analyze_and_reconstruct.lib_analysis_functions import *
from modules.generate_templates.process_templates import get_time_derivative

''' Main Functions '''
def process_unit_for_analysis(unit_id, unit_templates, recon_dir, analysis_options, params, failed_units, failed_units_file, logger=None):
    start = time.time()
    print(f'Analyzing unit {unit_id}...')
    
    '''Input Data for Reconstruction and Analysis'''    
    # Primary dictionaries to hold template data and analytics
    template_data = {
        'unit_id': unit_id,
        'vt_template': unit_templates.get('merged_template'),
        'dvdt_template': get_time_derivative(unit_templates.get('merged_template'), sampling_rate=10000), #TODO: get sampling rate from h5 data or something. Put it in analysis options
        'milos_template': [],  # Placeholder for milos_template if applicable #TODO: Implement milos_template
        'channel_locations': unit_templates.get('merged_channels_locs'),
        #'template_segments': unit_templates.get('template_segments', []),
        #'channel_loc': unit_templates.get('channels_loc'),
        #'channel_loc' : unit_templates.get('channel_locations'),
        #'channel_loc_filled': unit_templates.get('channels_loc_filled'),
        #'channel_loc_filled': unit_templates.get('channel_locs_filled'),
        # 'vt_template': unit_templates.get('merged_template'),
        # 'dvdt_template': unit_templates.get('dvdt'),
        # 'filled_vt_template': unit_templates.get('merged_template_filled'),
        # 'filled_dvdt_template': unit_templates.get('dvdt_filled'),        
    }

    '''Reconstruction'''
    # Dictionary for template analytics initialized with function references
    try:
        gtr, av_params = build_graph_axon_tracking(template_data, recon_dir=recon_dir, load_gtr=True, params=params)
    except:
        try:
            gtr, av_params = build_graph_axon_tracking(template_data, recon_dir=recon_dir, load_gtr=False, params=params)
        except Exception as e:
            if logger:
                logger.error(f"Failed to build graph for unit {unit_id} - Error: {e}")
            else:
                print(f"Failed to build graph for unit {unit_id} - Error: {e}")
            failed_units.add(unit_id)
            # Update the failed units JSON file immediately
            with open(failed_units_file, 'w') as f:
                json.dump(list(failed_units), f, indent=4)
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
        
        # Flatten reconstruction analytics
        for key, value in results['reconstruction_analytics'].items():
            flattened_entry[f'analytics_{key}'] = value
        
        flattened_data.append(flattened_entry)
    
    df = pd.DataFrame(flattened_data)
    return df

def reconstruct_and_analyze(templates, analysis_options=None, recon_dir=None, logger=None, params=None, max_workers=10, skip_failed_units=True, **kwargs):
    # Load previously failed units if skip_failed_units is True
    failed_units_file = Path(recon_dir) / 'failed_units.json'
    if skip_failed_units and failed_units_file.exists():
        with open(failed_units_file, 'r') as f:
            failed_units = set(json.load(f))
    else:
        failed_units = set()

    total_units = sum(len(stream_templates['units']) for tmps in templates.values() for stream_templates in tmps['streams'].values())
    progress_bar = tqdm(total=total_units, desc="Reconstructing units")

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

                #parse paths for different types of data
                gtr_dir = stream_recon_dir / 'unit_gtr_objects'
                if not gtr_dir.exists(): gtr_dir.mkdir(parents=True, exist_ok=True)
                analysis_json_dir = stream_recon_dir / 'unit_analytics'
                if not analysis_json_dir.exists(): analysis_json_dir.mkdir(parents=True, exist_ok=True)
                
                # Submit each unit processing task
                futures = {
                    executor.submit(
                        process_unit_for_analysis,
                        unit_id, unit_template, gtr_dir, analysis_options, params, failed_units, failed_units_file, logger
                    ): unit_id for unit_id, unit_template in unit_templates.items() if unit_id not in failed_units
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
                            
                            # Save each unit's result individually as JSON
                            with open(analysis_json_dir / f"{unit_id}_analysis_results.json", 'w') as f:
                                json.dump(stream_results[unit_id], f, indent=4, default=str)
                            print(f'Processed and saved unit {unit_id}')
                        else:
                            failed_units.add(unit_id)
                            # Save the failed units to a JSON file immediately
                            with open(failed_units_file, 'w') as f:
                                json.dump(list(failed_units), f, indent=4)
                    except Exception as e:
                        if logger:
                            logger.error(f"Error processing unit {unit_id}: {e}")
                        else:
                            print(f"Error processing unit {unit_id}: {e}")
                        failed_units.add(unit_id)
                        # Save the failed units to a JSON file immediately
                        with open(failed_units_file, 'w') as f:
                            json.dump(list(failed_units), f, indent=4)

                    # Update progress bar
                    progress_bar.update(1)

                # Convert the stream results to a DataFrame and save for the stream
                df = analysis_results_to_dataframe(stream_results)
                df.to_csv(stream_recon_dir / f"agg_{stream_id}_axon_analytics.csv", index=False)
                print(f'Saved analytics data for stream {stream_id} to {stream_recon_dir / f"{stream_id}_axon_analytics.csv"}')

    # Save the failed units to a JSON file
    with open(failed_units_file, 'w') as f:
        json.dump(list(failed_units), f, indent=4)

    progress_bar.close()

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
    template_dir = "./RBS_axonal_reconstructions/development_scripts/developing_nbs_for_AR_density_projects/templates_04Nov2024/241021/M06804/AxonTracking/000091/well000"
    template_dir = "/pscratch/sd/a/adammwea/RBS_axonal_reconstructions/development_scripts/developing_nbs_for_AR_density_projects/templates_04Nov2024/241021/M06804/AxonTracking/000091/well000"
    #template_dir = os.path.abspath(template_dir)
    print(f"Loading templates from {template_dir}")

    # Load templates from the directory
    templates = load_templates_from_directory(template_dir)

    # Define analysis options if any
    analysis_options = {
        # Add any specific options for analysis here
    }

    # Define reconstruction directory
    recon_dir = "./RBS_axonal_reconstructions/development_scripts/developing_nbs_for_AR_density_projects/templates_04Nov2024/241021/M06804/AxonTracking/000091/well000_reconstruction_files"
    recon_dir = "/pscratch/sd/a/adammwea/RBS_axonal_reconstructions/development_scripts/developing_nbs_for_AR_density_projects/templates_04Nov2024/241021/M06804/AxonTracking/000091/well000_reconstruction_files"
    #recon_dir = os.path.abspath(recon_dir)
    print(f"Saving reconstruction files to {recon_dir}")

    #get params
    params = get_default_graph_velocity_params()
    #params['r2_threshold'] = 0.5    
    
    # Call the reconstruct_and_analyze function
    max_workers = 20
    skip_failed_units = True
    reconstruct_and_analyze(templates, analysis_options=analysis_options, recon_dir=recon_dir, params=params, max_workers=max_workers, skip_failed_units=skip_failed_units)

if __name__ == "__main__":
    main()