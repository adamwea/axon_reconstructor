# run_pipeline.py with custom params as needed.
'''Setup Python environment '''
import setup_environment
setup_environment.set_pythonpath()

''' Import the necessary modules '''
import os
import pandas as pd
from run_pipeline import run_pipeline

''' Import HDF5 plugin to os.environ '''
os.environ['HDF5_PLUGIN_PATH'] = '/pscratch/sd/a/adammwea/RBS_axonal_reconstructions/modules/maxwell_hdf5_plugin/Linux'

''' Helper functions '''
def get_axon_tracking_file_paths(h5_parent_dirs):
    axon_tracking_dirs = []
    for h5_parent_dir in h5_parent_dirs:
        for root, dirs, files in os.walk(h5_parent_dir):
            for file in files:
                if file.endswith('.h5') and 'AxonTracking' in root:
                    file_path = os.path.join(root, file)
                    axon_tracking_dirs.append(file_path)
    return axon_tracking_dirs

''' Parameters '''
# Reconstructor parameters
kwargs = {
    # runtime options
    'sorting_params': {
        'allowed_scan_types': ['AxonTracking'],
        'load_existing_sortings': True,
        'keep_good_only': False,
        #'use_gpu': False,
    },
    'te_params': {
        'load_merged_templates': False,
        'save_merged_templates': True,
        'time_derivative': True,
    },
    'av_params': {
        # 'upsample': 2,  # Upsampling factor to better capture finer details (unitless). Higher values increase temporal resolution.
        # 'min_selected_points': 20,  # Minimum number of selected points to include more potential signals (unitless). Determines the minimum dataset size for analysis.
        # 'verbose': False,  # Verbosity flag for debugging (boolean). If True, additional logs are printed.

        # # Channel selection
        # 'detect_threshold': 0.01,  # Threshold for detecting signals, sensitive to smaller amplitudes (relative or absolute units, depending on 'detection_type'). Works with 'detection_type'.
        # 'detection_type': 'relative',  # Type of detection threshold ('relative' or 'absolute'). Determines if 'detect_threshold' is a relative or absolute value.
        # 'kurt_threshold': 0.2,  # Kurtosis threshold for noise filtering (unitless, kurtosis measure). Lower values allow more channels through noise filtering.
        # 'peak_std_threshold': 0.5,  # Peak time standard deviation threshold (milliseconds). Filters out channels with high variance in peak times.
        # 'init_delay': 0.05,  # Initial delay threshold to include faster signals (milliseconds). Minimum delay for considering a channel.
        # 'peak_std_distance': 20.0,  # Distance for calculating peak time standard deviation (micrometers). Defines the neighborhood for peak time calculation.
        # 'remove_isolated': True,  # Flag to remove isolated channels (boolean). If True, only connected channels are kept for analysis.

        # # Graph
        # 'init_amp_peak_ratio': 0.2,  # Ratio between initial amplitude and peak time (unitless). Used to sort channels for path search.
        # 'max_distance_for_edge': 150.0,  # Maximum distance for creating graph edges (micrometers). Defines the maximum allowed distance between connected channels.
        # 'max_distance_to_init': 300.0,  # Maximum distance to initial channel for creating edges (micrometers). Determines the initial connectivity range.
        # #'max_distance_to_init': 4000.0,  # Maximum distance to initial channel for creating edges (micrometers). Determines the initial connectivity range.
        # 'n_neighbors': 9,  # Maximum number of neighbors (edges) per channel (unitless). Enhances connectivity by increasing the number of edges.
        # 'distance_exp': 1.5,  # Exponent for distance calculation (unitless). Adjusts the influence of distance in edge creation.
        # 'edge_dist_amp_ratio': 0.2,  # Ratio between distance and amplitude for neighbor selection (unitless). Balances the importance of proximity and amplitude in selecting edges.

        # # Axonal reconstruction
        # 'min_path_length': 80.0,  # Minimum path length to include shorter paths (micrometers). Ensures that only significant paths are considered.
        # 'min_path_points': 3,  # Minimum number of channels in a path (unitless). Defines the minimum size of a valid path.
        # 'neighbor_radius': 80.0,  # Radius to include neighboring channels (micrometers). Defines the search radius for neighbors.
        # 'min_points_after_branching': 2,  # Minimum points after branching to keep paths (unitless). Ensures that branches have enough data points.

        # # Path cleaning/velocity estimation
        # 'mad_threshold': 10.0,  # Median Absolute Deviation threshold for path cleaning (unitless). Higher values allow more variability in the path.
        # 'split_paths': True,  # Flag to enable path splitting (boolean). If True, paths can be split for better velocity fit.
        # 'max_peak_latency_for_splitting': 0.7,  # Maximum peak latency jump for splitting paths (milliseconds). Allows capturing more variations by splitting paths at significant jumps.
        # 'r2_threshold': 0.75,  # R-squared threshold for velocity fit (unitless). Lower values include more paths with less perfect fits.
        # 'r2_threshold_for_outliers': 0.95,  # R-squared threshold for outlier detection (unitless). Defines the threshold below which the algorithm looks for outliers.
        # 'min_outlier_tracking_error': 40.0,  # Minimum tracking error to consider a channel as an outlier (micrometers). Sets the error tolerance for tracking outliers.
    },
    'analysis_options': {
        'generate_animation': True,
        'generate_summary': False,
    },
    'save_reconstructor_object': True,
    'reconstructor_save_options': {
        'recordings': True, 
        'multirecordings': True, 
        'sortings': True,
        'waveforms': False,
        'templates': True,
    },
    'reconstructor_load_options': {
        'load_reconstructor': True,
        
        #Only relevant if load_reconstructor is True:
        'load_multirecs': True,
        'load_sortings': True,
        'load_wfs': False,
        'load_templates': False,
        'load_templates_bypass': False,  # This is a new parameter that allows the user to bypass pre-processing steps and load the templates directly. 
                                        # Useful if there isn't any need to reprocess the templates.
        'restore_environment': False,
    },
    'verbose': True,
    'debug_mode': True,
    'n_jobs': 1,
    'max_workers': 6,
    'logger_level': 'DEBUG',
    'run_lean': True,
}

kwargs['project_name'] = None # Use project name to create subdirectories, if true the paths below can be relative
kwargs['output_dir'] = "/pscratch/sd/a/adammwea/zRBS_axon_reconstruction_output" # Output directory for the reconstructions when running on NERSC
#kwargs['output_dir'] = "/home/adamm/workspace/zRBS_axon_reconstruction_output" # Output directory for the reconstructions when running on lab server
kwargs['mode'] = 'lean'

'''Run the pipeline '''
h5_parent_dirs = [
    "/pscratch/sd/a/adammwea/RBS_synology_rsync/B6J_DensityTest_10012024_AR", #running on NERSC server, pulling from rsynced data from synology
    #"/mnt/ben-shalom_nas/rbsmaxtwo/media/rbs-maxtwo/harddisk20tb/B6J_DensityTest_10012024_AR/B6J_DensityTest_10012024_AR", #Running on lab server, pulling from synology
    #"/home/adamm/workspace/data/B6J_DensityTest_10012024_AR/B6J_DensityTest_10012024_AR" #Running on Gubuntu, pulling from local data
]
assert all([os.path.exists(h5_parent_dir) for h5_parent_dir in h5_parent_dirs]), "One or more h5_parent_dirs do not exist."
axon_tracking_file_paths = get_axon_tracking_file_paths(h5_parent_dirs)
#print(axon_tracking_file_paths)

run_pipeline(axon_tracking_file_paths, **kwargs)



# '''Run the pipeline '''
# project_csv_dir = '/home/adamm/workspace/RBS_axonal_reconstructions/analysis_queue/B6J_DensityTest_10012024_AR.csv'
# #index = 18 # not enough spikes, units get cleaned in waveform qc
# index = 11 # Points to the row in the CSV file to process
# run_pipeline_from_queue(project_csv_dir, index=index, **kwargs)
# # /home/adamm/workspace/RBS_axonal_reconstructions/analysis_queue/B6J_DensityTest_10012024_AR.csv

# # h5_parent_dirs = [
# #     #E:\aw data\B6J_DensityTest_10012024_AR\B6J_DensityTest_10012024_AR\241017\M08029\AxonTracking\000073\data.raw.h5
# #     #"/mnt/g/aw data/B6J_DensityTest_10012024_AR/B6J_DensityTest_10012024_AR/241017/M08029/AxonTracking/000073/data.raw.h5", #on ssd
# #     #path on synology: /volume1/MEA_Backup/rbsmaxtwo/media/rbs-maxtwo/harddisk20tb/B6J_DensityTest_10012024_AR/B6J_DensityTest_10012024_AR/...
# #     "/mnt/ben-shalom_nas/rbsmaxtwo/media/rbs-maxtwo/harddisk20tb/B6J_DensityTest_10012024_AR/B6J_DensityTest_10012024_AR", 
    
# # ]

# #'''Run the pipeline '''
# #run_pipeline(h5_parent_dirs, mode = mode, **kwargs) # Run the pipeline in lean mode (usually)
# #new mode to run pipeline on file paths in csv queue