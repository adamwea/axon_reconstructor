import logging
import numpy as np
from scipy.stats import kurtosis, skew
from pathlib import Path
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import dill
import pandas as pd
from tqdm import tqdm
import json
from modules.analyze_and_reconstruct.lib_analysis_functions import *
from modules.generate_templates.process_templates import get_time_derivative
import submodules.axon_velocity_fork.axon_velocity as av
import matplotlib.pyplot as plt

def plot_template_propagation(template, locations, selected_channels, sort_templates=False,
                              color=None, color_marker=None, ax=None):
    '''
    This function was originally copy pasted from the axon_velocity library, but has (will be) modified to generate propogation plots
    in a more specific way for validation of templates submitted for axon tracking.
    
    Changes made/needed:
    1. Currently, amplitudes are adjusted such that propogation signals at each channel wont overlap. With enough channels, 
        this just makes every signal look like a straight line. Allow individual lines to overlap to better visualize propogation.
    2. Add a marker to the maximum amplitude of each channel to better visualize the propogation path.
    3. To the left of each line, add a small label to indicate which branch the channel is a part of - i.e. where the signal is propogating.
        3.1. Sort the lines by branch depth instead of breadth to better visualize propogation. See screenshot for example.        
    '''
    
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    if color is None:
        color = 'C0'
    if color_marker is None:
        color_marker = 'r'

    template_selected = template[selected_channels]
    locs = locations[selected_channels]
    if sort_templates:
        peaks = np.argmin(template_selected, 1)
        sorted_idxs = np.argsort(peaks)
        template_sorted = template_selected[sorted_idxs]
        locs_sorted = locs[sorted_idxs]
    else:
        template_sorted = template_selected
        locs_sorted = locs
    dist_peaks_sorted = np.array([np.linalg.norm(loc - locs_sorted[0]) for loc in locs_sorted])
    dist_peaks_sorted /= np.max(dist_peaks_sorted)
    dist_peaks_sorted *= len(template_sorted)

    ptp_glob = np.max(np.ptp(template_sorted, 1))
    for i, temp in enumerate(template_sorted):
        temp_shifted = temp + i * 1.5 * ptp_glob  # dist_peaks_sorted[i]
        min_t = np.min(temp_shifted)
        min_idx = np.argmin(temp_shifted)

        ax.plot(temp_shifted, color=color)
        ax.plot(min_idx, min_t, marker='o', color=color_marker)

    ax.axis('off')
    return ax

def generate_propogation_plots(**kwargs):
    reconstruction_data = kwargs.get('reconstruction_data')
    reconstruction_analytics = kwargs.get('reconstruction_analytics')
    recon_dir = kwargs.get('recon_dir')
    unit_id = kwargs.get('unit_id')
    template_type = kwargs.get('template_type')
    template_data = kwargs.get('template_data')
    
    for template_type in ['vt', 'dvt', 'milos']:
        gtr = reconstruction_data[template_type]
        analytics = reconstruction_analytics[template_type]
        template = template_data[template_type]
        locations = template_data['channel_locations']
        selected_channels = gtr.channels #TODO: check if this is the correct attribute to use
        
        if template_type == 'vt':
            plot_template_propagation(template, locations, selected_channels, sort_templates=False, color='C0', color_marker='r', ax=None)
        elif template_type == 'dvt':
            plot_template_propagation(template, locations, selected_channels, sort_templates=False, color='C0', color_marker='r', ax=None)
        elif template_type == 'milos':
            #plot_template_propagation(gtr, analytics, recon_dir, unit_id, template_type)
            print(f"Milos method not yet implemented")
        else:
            print(f"Invalid template type: {template_type}")
            continue


def process_individual_unit_plots(template_data, reconstruction_data, reconstruction_analytics, recon_dir, unit_id, template_type):
    
    kwargs = {
        'template_data': template_data,
        'reconstruction_data': reconstruction_data,
        'reconstruction_analytics': reconstruction_analytics,
        'recon_dir': recon_dir,
        'unit_id': unit_id,
        'template_type': template_type
    }
    
    ## plot function 1
    try: generate_propogation_plots(**kwargs)
    except Exception as e: print(f"Failed to generate propogation plots for unit {unit_id} with template {template_type} - Error: {e}")
    ## plot function 2
    ## plot function 3
    ## plot function 4
    ## ...etc
    
    pass

def process_unit_for_analysis(unit_id, unit_templates, recon_dir, analysis_options, params, failed_units, failed_units_file, logger=None):
    start = time.time()
    print(f'Analyzing unit {unit_id}...')
    
    template_data = {
        'unit_id': unit_id,
        'vt_template': unit_templates.get('merged_template'),
        'dvdt_template': get_time_derivative(unit_templates.get('merged_template'), sampling_rate=10000),
        'milos_template': [],  # Placeholder for milos template
        'channel_locations': unit_templates.get('merged_channel_locs'),
    }

    reconstruction_data = {}
    reconstruction_analytics = {}

    for template_type in ['vt', 'dvt', 'milos']:
        try:
            gtr, av_params = build_graph_axon_tracking(template_data, recon_dir=recon_dir, load_gtr=True, params=params, template_type=template_type)
            
            if recon_dir:
                template_recon_dir = Path(recon_dir) / template_type
                template_recon_dir.mkdir(parents=True, exist_ok=True)
                gtr_file = template_recon_dir / f"{unit_id}_gtr_object.dill"
                with open(gtr_file, 'wb') as f:
                    dill.dump(gtr, f) 
        except:
            try:
                gtr, av_params = build_graph_axon_tracking(template_data, recon_dir=recon_dir, load_gtr=False, params=params, template_type=template_type)
                
                if recon_dir:
                    template_recon_dir = Path(recon_dir) / template_type
                    template_recon_dir.mkdir(parents=True, exist_ok=True)
                    gtr_file = template_recon_dir / f"{unit_id}_gtr_object.dill"
                    with open(gtr_file, 'wb') as f:
                        dill.dump(gtr, f) 
            except Exception as e:
                if logger:
                    logger.error(f"Failed to build graph for unit {unit_id} with template {template_type} - Error: {e}")
                else:
                    print(f"Failed to build graph for unit {unit_id} with template {template_type} - Error: {e}")
                failed_units.add(unit_id)
                save_failed_units(failed_units, failed_units_file)
                continue

        reconstruction_data[template_type] = gtr
        reconstruction_analytics[template_type] = {
            'channels_in_template': len(template_data['channel_locations']),
            'template_rect_area': calculate_template_rectangular_area(template_data),
            'template_density': len(template_data['channel_locations']) / calculate_template_rectangular_area(template_data),
            'axon_length': gtr.compute_path_length(gtr.branches[0]['channels']),
            'num_branches': len(gtr.branches),
            'branch_lengths': [gtr.compute_path_length(branch['channels']) for branch in gtr.branches],
            'branch_channel_density': [gtr.compute_path_length(branch['channels']) / len(branch['channels']) for branch in gtr.branches],
            'branch_orders': {index: calculate_branch_order(branch['channels'], gtr._branching_points) for index, branch in enumerate(gtr.branches)},
            'maximum_branch_order': max([calculate_branch_order(branch['channels'], gtr._branching_points) for branch in gtr.branches]),
            'velocities': {index: branch['velocity'] for index, branch in enumerate(gtr.branches)}
        }
        
    process_individual_unit_plots(template_data, reconstruction_data, reconstruction_analytics, recon_dir, unit_id, template_type)

    print(f'Finished analyzing unit {unit_id} in {time.time() - start} seconds')
    return template_data, reconstruction_data, reconstruction_analytics

def analysis_results_to_dataframe(analysis_results):
    flattened_data = []

    for unit_id, results in analysis_results.items():
        flattened_entry = {'unit_id': unit_id}
        
        for key, value in results.items():
            flattened_entry[key] = value
        
        flattened_data.append(flattened_entry)
    
    exploded_data = []
    for entry in flattened_data:
        max_length = max(len(v) if isinstance(v, list) else 1 for v in entry.values())
        for i in range(max_length):
            new_entry = {'unit_id': entry['unit_id'], 'branch_id': i}
            for key, value in entry.items():
                if key == 'unit_id':
                    continue
                if isinstance(value, list):
                    new_entry[key] = value[i] if i < len(value) else None
                elif isinstance(value, dict):
                    new_entry[key] = value.get(i, None)
                else:
                    new_entry[key] = value
            exploded_data.append(new_entry)
    
    df = pd.DataFrame(exploded_data)
    return df

def load_failed_units(failed_units_file, skip_failed_units):
    if skip_failed_units and failed_units_file.exists():
        with open(failed_units_file, 'r') as f:
            return set(json.load(f))
    return set()

def setup_directories(recon_dir):
    stream_recon_dir = Path(recon_dir)
    stream_recon_dir.mkdir(parents=True, exist_ok=True)
    
    gtr_dir = stream_recon_dir / 'unit_gtr_objects'
    gtr_dir.mkdir(parents=True, exist_ok=True)
    
    analysis_json_dir = stream_recon_dir / 'unit_analytics'
    analysis_json_dir.mkdir(parents=True, exist_ok=True)
    
    return stream_recon_dir, gtr_dir, analysis_json_dir

def submit_tasks(executor, unit_templates, failed_units, unit_limit, gtr_dir, analysis_options, params, failed_units_file, logger):
    futures = {}
    for unit_id, unit_template in unit_templates.items():
        if unit_id not in failed_units and (unit_limit is None or len(futures) < unit_limit):
            future = executor.submit(
                process_unit_for_analysis,
                unit_id, unit_template, gtr_dir, analysis_options, params, failed_units, failed_units_file, logger
            )
            futures[future] = unit_id
    return futures

def process_results(futures, analysis_json_dir, failed_units, failed_units_file, logger, progress_bar):
    stream_results = {}
    successful_units = 0
    for future in as_completed(futures):
        unit_id = futures[future]
        try:
            template_data, reconstruction_data, reconstruction_analytics = future.result()
            if reconstruction_analytics:
                if reconstruction_analytics and any(reconstruction_analytics.values()):
                    print(f'Unit {unit_id} analytics: {reconstruction_analytics}')
            if template_data is not None:
                stream_results[unit_id] = {
                    'template_data': template_data,
                    'reconstruction_data': reconstruction_data,
                    'reconstruction_analytics': reconstruction_analytics
                }
                
                if any(isinstance(v, np.int64) for v in stream_results[unit_id].values()):
                    stream_results[unit_id] = {k: int(v) if isinstance(v, (np.integer, np.int64)) else v for k, v in stream_results[unit_id].items()}
                
                with open(analysis_json_dir / f"{unit_id}_analysis_results.json", 'w') as f:
                    json.dump(stream_results[unit_id], f, indent=4, default=str)
                print(f'Processed and saved unit {unit_id}')
                successful_units += 1
            else:
                failed_units.add(unit_id)
                save_failed_units(failed_units, failed_units_file)
        except Exception as e:
            if logger:
                logger.error(f"Error processing unit {unit_id}: {e}")
            else:
                print(f"Error processing unit {unit_id}: {e}")
            failed_units.add(unit_id)
            save_failed_units(failed_units, failed_units_file)
        progress_bar.update(1)
    return stream_results, successful_units

def save_failed_units(failed_units, failed_units_file):
    failed_units = [int(unit) for unit in failed_units]
    
    with open(failed_units_file, 'w') as f:
        json.dump(failed_units, f, indent=4)  

def reconstruct_and_analyze(templates, analysis_options=None, recon_dir=None, logger=None, params=None, n_jobs=None, skip_failed_units=True, unit_limit=None, **kwargs):
    #n_jobs = 1  #debug
    if n_jobs is None:
        n_jobs = 1  #debug
    
    failed_units_file = Path(recon_dir) / 'failed_units.json'
    skip_failed_units = False
    failed_units = load_failed_units(failed_units_file, skip_failed_units)

    total_units = sum(len(stream_templates['units']) for tmps in templates.values() for stream_templates in tmps['streams'].values())
    progress_bar = tqdm(total=total_units, desc="Reconstructing units")

    successful_units = 0
    processed_units = 0
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        for key, tmps in templates.items():
            for stream_id, stream_templates in tmps['streams'].items():
                unit_templates = stream_templates['units']
                stream_recon_dir, gtr_dir, analysis_json_dir = setup_directories(recon_dir)
                
                while unit_limit is None or successful_units < unit_limit:
                    futures = submit_tasks(executor, unit_templates, failed_units, unit_limit - successful_units if unit_limit is not None else len(unit_templates), gtr_dir, analysis_options, params, failed_units_file, logger)
                    stream_results, new_successful_units = process_results(futures, analysis_json_dir, failed_units, failed_units_file, logger, progress_bar)
                    successful_units += new_successful_units
                    processed_units += len(futures)
                    print(f'Processed {successful_units} units')
                    
                    # Break the loop if all units have been processed
                    if processed_units >= total_units:
                        break
                
                agg_template_analytics = {}
                for template_type in ['vt', 'dvt', 'milos']:
                    agg_template_analytics = {unit_id: results['reconstruction_analytics'][template_type] for unit_id, results in stream_results.items() if template_type in results['reconstruction_analytics']}   
                    df = analysis_results_to_dataframe(agg_template_analytics)
                    df.to_csv(stream_recon_dir / f"agg_{stream_id}_{template_type}_axon_analytics.csv", index=False)                    
                    print(f'Saved analytics data for stream {stream_id} and template {template_type} to {stream_recon_dir / f"agg_{stream_id}_{template_type}_axon_analytics.csv"}')
                    with open(stream_recon_dir / f"agg_{stream_id}_{template_type}_axon_analytics.csv", 'r') as f:
                        print(f.read())

                # Break the outer loop if all units have been processed
                if processed_units >= total_units:
                    break

    save_failed_units(failed_units, failed_units_file)
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
    max_workers = 1 #debug
    skip_failed_units = True
    reconstruct_and_analyze(templates, analysis_options=analysis_options, recon_dir=recon_dir, params=params, max_workers=max_workers, skip_failed_units=skip_failed_units)

if __name__ == "__main__":
    main()