''' This file contains template-related functions that are used in the main pipeline functions. '''

''' Imports '''
import sys
import os
import h5py
import spikeinterface.full as si
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

''' Local imports '''
from modules import mea_processing_library as MPL
import modules.lib_sorting_functions as sorter
import modules.lib_waveform_functions as waveformer
#from RBS_axonal_reconstructions.modules.generate_templates.process_templates import merge_templates
#from axon_reconstructor.utils.process_templates import merge_templates
from .process_templates import merge_templates


default_n_jobs = 4

''' Logging Functions '''

def log_info(logger, message):
    if logger:
        logger.info(message)
    else:
        print(message)

def log_warning(logger, message):
    if logger:
        logger.warning(message)
    else:
        print(message)

def log_error(logger, message):
    if logger:
        logger.error(message)
    else:
        print(message)

def log_debug(logger, message):
    if logger:
        logger.debug(message)
    else:
        print(message)

''' Template Processing Functions '''

def process_template_segment(sel_unit_id, rec_name, waveforms, save_root, sel_idx, logger):
    try:
        seg_we = waveforms[rec_name]['waveforms']
    except KeyError:
        log_error(logger, f'{rec_name} does not exist')
        return None, None

    try:
        template = seg_we.get_template(sel_unit_id)
    except Exception as e:
        log_error(logger, f'Could not get templates for {sel_unit_id} in {rec_name}: {e}')
        return None, None

    if np.isnan(template).all():
        log_warning(logger, f'Unit {sel_unit_id} in segment {sel_idx} is empty. Skipping.')
        return None, None

    locs = seg_we.get_channel_locations()
    dir_path = os.path.join(save_root, 'template_segments')
    file_name = f'seg{sel_idx}_unit{sel_unit_id}.npy'
    channel_loc_file_name = f'seg{sel_idx}_unit{sel_unit_id}_channels.npy'
    channel_loc_save_file = os.path.join(dir_path, channel_loc_file_name)
    template_save_file = os.path.join(dir_path, file_name)

    if not np.isnan(template).all() and np.isnan(template).any():
        log_warning(logger, f'Unit {sel_unit_id} in segment {sel_idx} has NaN values')

    return (rec_name, {'template': template, 'path': template_save_file}), (rec_name, {'channel_locations': locs, 'path': channel_loc_save_file})

def extract_template_segments(sel_unit_id, h5_path, stream_id, waveforms, save_root=None, logger=None, max_workers=12):
    log_info(logger, f'Extracting template segments for unit {sel_unit_id}')
    full_path = h5_path
    h5 = h5py.File(full_path)
    rec_names = list(h5['wells'][stream_id].keys())

    template_segments = {}
    channel_locations = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_template_segment, sel_unit_id, rec_name, waveforms, save_root, sel_idx, logger)
            for sel_idx, rec_name in enumerate(rec_names)
        ]
        for future in as_completed(futures):
            try:
                template_result, loc_result = future.result()
                if template_result and loc_result:
                    rec_name_t, template_data = template_result
                    rec_name_l, loc_data = loc_result
                    template_segments[rec_name_t] = template_data
                    channel_locations[rec_name_l] = loc_data
                    log_debug(logger, f'Processed segment {rec_name_t} for unit {sel_unit_id}')
            except Exception as e:
                log_error(logger, f'Error processing unit {sel_unit_id} in segment {rec_name_l}: {e}')
                continue

    log_info(logger, f'Finished extracting {len(template_segments)} template segments for unit {sel_unit_id}')
    return template_segments, channel_locations

def merge_template_segments(unit_segments, channel_locations, logger=None):
    log_info(logger, f'Merging partial templates')
    template_list = [tmp['template'] for rec_name, tmp in unit_segments.items()]
    channel_locations_list = [ch_loc['channel_locations'] for rec_name, ch_loc in channel_locations.items()]
    merged_template, merged_channel_loc = merge_templates(template_list, channel_locations_list, logger=logger)
    merged_template = merged_template[0]
    merged_channel_loc = merged_channel_loc[0]
    return merged_template, merged_channel_loc

def get_existing_unit_ids(template_save_path):
    template_files = os.listdir(template_save_path)
    sel_unit_ids = []
    for f in template_files:
        try:
            sel_unit_ids.append(int(f.split('.')[0]))
        except ValueError:
            continue  # Skip files that don't start with an integer
    return sel_unit_ids

def add_to_dict(unit_templates, sel_unit_id, unit_segments, channel_locations, merged_template, merged_channel_loc, template_save_file, channel_loc_save_file):
    unit_templates[sel_unit_id] = {
        'template_segments': unit_segments,
        'channel_segments': channel_locations,
        'merged_template': merged_template,
        'merged_channel_locs': merged_channel_loc,
        'merged_template_path': template_save_file,
        'merged_channel_loc_path': channel_loc_save_file,
    }

def load_existing_template(sel_unit_id, template_save_file, channel_loc_save_file, logger):
    log_info(logger, f'Loading merged template for unit {sel_unit_id}')
    merged_template = np.load(template_save_file)
    merged_channel_loc = np.load(channel_loc_save_file)
    return merged_template, merged_channel_loc

def extract_and_merge_template(sel_unit_id, h5_path, stream_id, waveforms, te_params, save_root, logger):
    log_info(logger, f'Extracting template segments for unit {sel_unit_id}')
    n_jobs = te_params.get('n_jobs', 8)
    unit_segments, channel_locations = extract_template_segments(sel_unit_id, h5_path, stream_id, waveforms, save_root=save_root, logger=logger, max_workers=n_jobs)
    log_info(logger, f'Merging partial templates for unit {sel_unit_id}')
    merged_template, merged_channel_loc = merge_template_segments(unit_segments, channel_locations, logger=logger)
    return unit_segments, channel_locations, merged_template, merged_channel_loc

def extract_merged_templates(h5_path, stream_id, segment_sorting, waveforms, te_params, save_root=None, unit_limit=None, logger=None, template_bypass=False, unit_select=None, **temp_kwargs):
    log_info(logger, f'Extracting merged templates for {h5_path}')
    h5_details = MPL.extract_recording_details(h5_path)[0]
    date = h5_details['date']
    chip_id = h5_details['chipID']
    scanType = h5_details['scanType']
    run_id = h5_details['runID']
    #template_save_path = os.path.join(save_root, date, chip_id, scanType, run_id, stream_id) if save_root else None
    template_save_path = save_root if save_root else None
    assert template_save_path, 'No save root provided for templates'
    
    if template_bypass:
        sel_unit_ids = get_existing_unit_ids(template_save_path)
        assert sel_unit_ids, 'No existing templates found. Generating New Templates.'
    else:
        sel_unit_ids = segment_sorting.get_unit_ids()

    if not os.path.exists(template_save_path):
        os.makedirs(template_save_path)

    unit_templates = {}
    unit_count = 0

    for sel_unit_id in tqdm(sel_unit_ids):
        if unit_select is not None and sel_unit_id != unit_select:
            continue  # manage unit selection option

        template_save_file = os.path.join(template_save_path, f'{sel_unit_id}.npy')
        channel_loc_save_file = os.path.join(template_save_path, f'{sel_unit_id}_channels.npy')

        try:
            assert te_params.get('load_merged_templates', False), 'load_merged_templates is set to False. Generating New Templates.'
            assert not te_params.get('overwrite_tmp', False), 'overwrite_tmp is set to True. Generating New Templates.'

            merged_template, merged_channel_loc = load_existing_template(sel_unit_id, template_save_file, channel_loc_save_file, logger)
            add_to_dict(unit_templates, sel_unit_id, "loading segments is not currently supported", "loading segment channels is not currently supported", merged_template, merged_channel_loc, template_save_file, channel_loc_save_file)
            unit_count += 1
            if unit_limit is not None and unit_count >= unit_limit:
                break
            continue
        except Exception as e:
            log_warning(logger, f'Error loading template for unit {sel_unit_id}:\n{e}. Generating New Templates.')

        try:
            unit_segments, channel_locations, merged_template, merged_channel_loc = extract_and_merge_template(sel_unit_id, h5_path, stream_id, waveforms, te_params, save_root, logger)
            add_to_dict(unit_templates, sel_unit_id, unit_segments, channel_locations, merged_template, merged_channel_loc, template_save_file, channel_loc_save_file)

            if te_params.get('save_merged_templates', False):
                np.save(template_save_file, merged_template)
                np.save(channel_loc_save_file, merged_channel_loc)

            log_info(logger, f'Merged template saved to {save_root}/templates')
            unit_count += 1
            if unit_limit is not None and unit_count >= unit_limit:
                break
        except Exception as e:
            log_error(logger, f'Unit {sel_unit_id} encountered the following error: {e}')

    return unit_templates

def extract_templates(multirec, sorting, waveforms, h5_path, stream_id, save_root=None, te_params={}, qc_params={}, unit_limit=None, logger=None, template_bypass=False, **temp_kwargs):
    log_info(logger, f'Extracting templates for {h5_path}')
    if template_bypass:
        try: 
            unit_templates = extract_merged_templates(h5_path, stream_id, None, None, te_params, save_root=save_root, unit_limit=unit_limit, logger=logger, template_bypass=True, **temp_kwargs)
            return unit_templates
        except Exception as e: 
            log_error(logger, f'Error loading templates via bypass:\n{e}')
    cleaned_sorting = waveformer.select_units(sorting, **qc_params)
    cleaned_sorting = si.remove_excess_spikes(cleaned_sorting, multirec) 
    cleaned_sorting.register_recording(multirec)
    segment_sorting = si.SplitSegmentSorting(cleaned_sorting, multirec)
    unit_templates = extract_merged_templates(h5_path, stream_id, segment_sorting, waveforms, te_params, save_root=save_root, unit_limit=unit_limit, logger=logger, **temp_kwargs)
    return unit_templates