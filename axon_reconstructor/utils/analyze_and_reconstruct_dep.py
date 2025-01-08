'''process each unit template, generate axon reconstructions and analytics'''

from modules.lib_axon_velocity_functions import *
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
from submodules.axon_velocity_fork.axon_velocity.axon_velocity import GraphAxonTracking
from submodules.axon_velocity_fork.axon_velocity import get_default_graph_velocity_params
from copy import deepcopy
import shutil
from scipy.signal import find_peaks
from modules.analyze_and_reconstruct.process_unit import process_unit

def process_result(result, all_analytics):
    for key, data in result.items():
        if isinstance(data, Exception): continue
        for metric, values in data.items():
            all_analytics[key][metric].append(values)

def save_results(stream_id, all_analytics, recon_dir, logger=None):
    for key, data in all_analytics.items():
        suffix = key.split('_')[-1] if '_' in key else 'template'
        if 'dvdt' in key: suffix = 'dvdt'
        save_axon_analytics(
            stream_id, data['units'], data['extremums'], data['branch_ids'], data['velocities'], 
            data['path_lengths'], data['r2s'], data['num_channels_included'], data['channel_density'], data['init_chans'],
            recon_dir, suffix=f"_{suffix}"
        )

def mark_for_deletion(unit_id, plot_dir, failed_recons):
    for suffix in failed_recons[unit_id]:
        for ext in ['png', 'svg', 'jpg']:
            for file in plot_dir.glob(f"*{suffix}*.{ext}"):
                file.unlink()
                
'''Main function to analyze and reconstruct axons from templates'''
def analyze_and_reconstruct(templates, params=None, analysis_options=None, recon_dir=None, stream_select=None, n_jobs=8, logger=None, unit_select=None, **recon_kwargs):
    if params is None:
        params = get_default_graph_velocity_params()
    
    for key, tmps in templates.items():
        for stream_id, stream_templates in tmps['streams'].items():
            if stream_select is not None and stream_id != f'well{stream_select:03}':
                continue
            
            unit_templates = stream_templates['units']
            date = tmps['date']
            chip_id = tmps['chip_id']
            scanType = tmps['scanType']
            run_id = tmps['run_id']
            recon_dir = Path(recon_dir) / date / chip_id / scanType / run_id / stream_id
            successful_recons = {str(recon_dir): {"successful_units": {}}}
            failed_recons = {}
            
            logger.info(f'Processing {len(unit_templates)} units, with {n_jobs} workers')

            all_analytics = {
                key: {
                    metric: [] for metric in [
                        'units', 
                        'branch_ids', 
                        'velocities', 
                        'path_lengths', 
                        'r2s', 
                        'extremums', 
                        'num_channels_included', 
                        'channel_density', 
                        'init_chans'
                    ]
                } for key in [
                    'merged_template', 
                    'dvdt_merged_template'
                ]
            }
            
            #n_jobs = 1
            #if unit_select is not None: n_jobs = 1
            #TODO: need to disambiguate workers and n_jobs... in this case the max_workers value is being assigned to n_jobs
            #FIXME: this is a bug, n_jobs should be the number of workers, max_workers should be the number of workers
            one_job_per_unit = True #TODO: make this a setting in the params
            if n_jobs > 1 and one_job_per_unit is False:
                with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                    futures = [
                        executor.submit(process_unit, unit_id, unit_templates, recon_dir, params, analysis_options, successful_recons, failed_recons, logger)
                        for unit_id, unit_templates in unit_templates.items()
                    ]
                    for future in as_completed(futures):
                        try:
                            result = future.result()
                            if result:
                                process_result(result, all_analytics)
                        except Exception as exc:
                            logger.info(f'Unit generated an exception: {exc}')
                            if 'A process in the process pool was terminated abruptly' in str(exc):
                                raise exc
            else:
                for unit_id, unit_templates in unit_templates.items():
                    if unit_select is not None and unit_id != unit_select: continue
                    result = process_unit(unit_id, unit_templates, recon_dir, params, analysis_options, successful_recons, failed_recons, logger=logger)
                    if result:
                        process_result(result, all_analytics)
                    
            save_results(stream_id, all_analytics, recon_dir, logger=logger)