
'''imports'''
import sys
import os
import h5py
import spikeinterface.full as si
import numpy as np
from tqdm import tqdm
from modules import mea_processing_library as MPL
from modules import waveformer
from modules import lib_helper_functions as helper
from RBS_axonal_reconstructions.modules.generate_templates.extract_templates import extract_merged_templates



''' Deprecated functions '''

def extract_merged_templates_olderVersion(h5_path, stream_id, segment_sorting, waveforms, te_params, save_root=None, unit_limit=None, logger=None, template_bypass=False, unit_select=None, **temp_kwargs):
    def get_existing_unit_ids():
        template_files = os.listdir(template_save_path)
        sel_unit_ids = []
        for f in template_files:
            try: sel_unit_ids.append(int(f.split('.')[0]))
            except ValueError: continue  # Skip files that don't start with an integer          
        return sel_unit_ids
    
    def add_to_dict(unit_segments, channel_locations, merged_template, merged_channel_loc, 
                    #merged_template_filled, 
                    #merged_channel_locs_filled, 
                    template_save_file, channel_loc_save_file, 
                    #template_save_file_fill, 
                    #channel_loc_save_file_fill, d_merged_template, d_merged_template_filled
                    ):
        unit_templates[sel_unit_id] = {
            
            #template and channel locations
            'template_segments': unit_segments, 
            'channel_segments': channel_locations,
            'merged_template': merged_template,
            'merged_channel_locs': merged_channel_loc,
            
            #save paths
            'merged_template_path': template_save_file,
            'merged_channel_loc_path': channel_loc_save_file,

            #'dvdt_merged_template': d_merged_template,
            #'merged_channel_loc_path': channel_loc_save_file,
            #'merged_channel_loc': merged_channel_loc,
            #'merged_template_filled_path': template_save_file_fill,
            #'dvdt_merged_template_filled': d_merged_template_filled,
            #'merged_template_filled': merged_template_filled,
            #'merged_channel_locs_filled_path': channel_loc_save_file_fill,
            #'merged_channel_locs_filled': merged_channel_locs_filled, 
        }
    
    if logger is not None: logger.info(f'Extracting merged templates for {h5_path}')
    else: print(f'Extracting merged templates for {h5_path}')        

    h5_details = MPL.extract_recording_details(h5_path)
    h5_details = h5_details[0]
    date = h5_details['date']
    chip_id = h5_details['chipID']
    scanType = h5_details['scanType']
    run_id = h5_details['runID']
    if save_root is not None: template_save_path = save_root+f'/{date}/{chip_id}/{scanType}/{run_id}/{stream_id}'
    if template_bypass: 
        sel_unit_ids = get_existing_unit_ids()
        assert len(sel_unit_ids) > 0, 'No existing templates found. Generating New Templates.'    
    else: sel_unit_ids = segment_sorting.get_unit_ids()
    if not os.path.exists(template_save_path): os.makedirs(template_save_path)
        
    unit_templates = {}
    unit_count = 0
    for sel_unit_id in tqdm(sel_unit_ids):
        if unit_select is not None and sel_unit_id != unit_select: continue # manage unit selection option
        
        template_save_file = os.path.join(template_save_path, str(sel_unit_id) + '.npy')
        #dvdt_template_save_file = os.path.join(template_save_path, str(sel_unit_id) + '_dvdt.npy')
        channel_loc_save_file = os.path.join(template_save_path, str(sel_unit_id) + '_channels.npy')
        #template_save_file_fill = os.path.join(template_save_path, str(sel_unit_id) + '_filled.npy')
        #dvdt_template_save_file_fill = os.path.join(template_save_path, str(sel_unit_id) + '_dvdt_filled.npy')
        #channel_loc_save_file_fill = os.path.join(template_save_path, str(sel_unit_id) + '_channels_filled.npy')

        try:
            assert te_params.get('load_merged_templates', False) == True, 'load_merged_templates is set to False. Generating New Templates.'
            assert te_params.get('overwrite_tmp', False) == False, 'overwrite_tmp is set to True. Generating New Templates.'
            
            if logger is not None: logger.info(f'Loading merged template for unit {sel_unit_id}')
            else: print(f'Loading merged template for unit {sel_unit_id}')                
            merged_template = np.load(template_save_file)
            #dvdt_template_save_file = np.load(dvdt_template_save_file)
            merged_channel_loc = np.load(channel_loc_save_file)
            #merged_template_filled = np.load(template_save_file_fill)
            #dvdt_template_save_file_fill = np.load(dvdt_template_save_file_fill)
            #merged_channel_locs_filled = np.load(channel_loc_save_file_fill)
            add_to_dict(
                "loading segments is not currently supported", 
                "loading segment channels is not currently supported", 
                merged_template, merged_channel_loc, 
                #merged_template_filled, merged_channel_locs_filled,
                template_save_file, channel_loc_save_file, 
                #template_save_file_fill, channel_loc_save_file_fill,
                #dvdt_template_save_file, dvdt_template_save_file_fill
            )
            unit_count += 1
            if unit_limit is not None and unit_count >= unit_limit: break
            continue 
        except Exception as e: 
            if logger is not None: logger.warning(f'loading template for unit {sel_unit_id}:\n{e}. Generating New Templates.')
            else: print(f'Error loading template for unit {sel_unit_id}:\n{e}. Generating New Templates.'); pass 

        try:
            if logger is not None: logger.info(f'Extracting template segments for unit {sel_unit_id}')
            else: print(f'Extracting template segments for unit {sel_unit_id}')
                
            n_jobs = te_params.get('n_jobs', 8)
            unit_segments, channel_locations = extract_template_segments(sel_unit_id, h5_path, stream_id, waveforms, save_root=save_root, logger=logger, max_workers=n_jobs)            
            if logger is not None: logger.info(f'Merging partial templates for unit {sel_unit_id}')
            else: print(f'Merging partial templates for unit {sel_unit_id}')
                
            #merged_template, merged_channel_loc, merged_template_filled, merged_channel_locs_filled = merge_template_segments(unit_segments, channel_locations, logger=logger)
            merged_template, merged_channel_loc = merge_template_segments(unit_segments, channel_locations, logger=logger)
            #dvdt_merged_template, dvdt_merged_template_filled = get_time_derivative(merged_template, merged_template_filled)
            
            add_to_dict(
                unit_segments, channel_locations, merged_template, merged_channel_loc, 
                #merged_template_filled, 
                #merged_channel_locs_filled, 
                template_save_file, channel_loc_save_file, 
                #template_save_file_fill, 
                #channel_loc_save_file_fill, dvdt_merged_template, dvdt_merged_template_filled
            )
            
            if te_params.get('save_merged_templates', False):
                np.save(template_save_file, merged_template)
                #np.save(dvdt_template_save_file, dvdt_merged_template)
                np.save(channel_loc_save_file, merged_channel_loc)
                #np.save(template_save_file_fill, merged_template_filled)
                #np.save(dvdt_template_save_file_fill, dvdt_merged_template_filled)
                #np.save(channel_loc_save_file_fill, merged_channel_locs_filled)
            if logger is not None: logger.info(f'Merged template saved to {save_root}/templates')
            else: print(f'Merged template saved to {save_root}/templates')
            unit_count += 1
            if unit_limit is not None and unit_count >= unit_limit: break                
        except Exception as e:
            if logger is not None: logger.error(f'Unit {sel_unit_id} encountered the following error: {e}')
            else: print(f'Unit {sel_unit_id} encountered the following error:\n {e}')
    return unit_templates