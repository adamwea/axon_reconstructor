import sys
import os

# Use git to get the root directory of the repository
def get_git_root():
    git_root = os.popen('git rev-parse --show-toplevel').read().strip()
    return git_root

git_root = get_git_root()
sys.path.insert(0, git_root)

from modules import mea_processing_library as MPL
import modules.lib_helper_functions as helper
from modules.axon_reconstructor import AxonReconstructor

def run_pipeline(h5_parent_dirs, mode='normal', **kwargs):
    '''
    Main function to run the pipeline.
    Supports two modes: 'normal' and 'lean'.
    '''
    h5_files = helper.get_list_of_h5_files(h5_parent_dirs, **kwargs)
    
    if mode == 'normal':
        '''
        Run the pipeline normally
        '''
        reconstructor = AxonReconstructor(h5_files, **kwargs)
        reconstructor.run_pipeline(**kwargs)
    
    elif mode == 'lean':
        '''
        Hacky way to run the pipeline on max two plates, one well at a time to minimize size of temp data
        '''
        for h5_file in h5_files:
            max_two_wells = 6  # 6 wells total
            max_two_wells_analyzed = 0
            
            while max_two_wells_analyzed < max_two_wells:
                # Get reconstructor ID
                h5_details = MPL.extract_recording_details([h5_file])
                date = h5_details[0]['date']
                chipID = h5_details[0]['chipID']
                runID = h5_details[0]['runID']
                
                if 'stream_select' not in kwargs:
                    kwargs['stream_select'] = max_two_wells_analyzed  # should go 0 through 5
                else:
                    if max_two_wells_analyzed != kwargs['stream_select']:
                        max_two_wells_analyzed += 1
                        continue
                
                projectName = h5_details[0]['projectName']
                reconstructorID = f'{date}_{chipID}_{runID}_well00{kwargs["stream_select"]}'
                
                # Set log files to be unique for each reconstructor
                kwargs['log_file'] = f'{kwargs["output_dir"]}/{projectName}/{reconstructorID}_axon_reconstruction.log'
                kwargs['error_log_file'] = f'{kwargs["output_dir"]}/{projectName}/{reconstructorID}_axon_reconstruction_error.log'
                kwargs['recordings_dir'] = os.path.join(kwargs['output_dir'], projectName, 'temp_data/recordings')
                kwargs['sortings_dir'] = os.path.join(kwargs['output_dir'], projectName, 'temp_data/sortings')
                kwargs['waveforms_dir'] = os.path.join(kwargs['output_dir'], projectName, 'temp_data/waveforms')
                kwargs['templates_dir'] = os.path.join(kwargs['output_dir'], projectName, 'temp_data/templates')
                kwargs['recon_dir'] = os.path.join(kwargs['output_dir'], projectName, 'reconstructions')
                kwargs['reconstructor_dir'] = os.path.join(kwargs['output_dir'], projectName, 'reconstructors')
                
                # Run the pipeline
                reconstructor = AxonReconstructor([h5_file], **kwargs)
                reconstructor.run_pipeline(**kwargs)
                
                max_two_wells_analyzed += 1