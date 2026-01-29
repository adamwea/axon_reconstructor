def estimate_background_noise(template, peak_threshold=0.01):
    """
    Cuts out windows around each spike in the template and estimates the background noise.
    
    Parameters:
    - template: 2D array where each row corresponds to a channel and each column is a time point.
    - peak_threshold: Threshold multiplier for peak detection based on the maximum amplitude across all channels.
    
    Returns:
    - noise_std: Estimated standard deviation of the background noise for each channel.
    """
    
    num_channels, num_timepoints = template.shape
    
    # Calculate the global maximum amplitude across all channels
    global_max_amplitude = np.max(np.abs(template))
    
    # Set the threshold for peak detection
    threshold = peak_threshold * global_max_amplitude
    
    # Initialize an array to store the noise data
    noise_data = []
    
    for channel in range(num_channels):
        # Get the template for the current channel
        signal = template[channel, :]
        
        # Calculate the baseline (average of the signal)
        baseline = np.mean(signal)
        
        # Find peaks above the threshold
        peaks, _ = find_peaks(signal, height=threshold)
        
        # Initialize a mask to include all data points
        mask = np.ones(num_timepoints, dtype=bool)
        
        # Exclude windows around each detected peak
        for peak in peaks:
            # Find the start of the peak (when it leaves the baseline)
            start = peak
            while start > 0 and signal[start] > baseline:
                start -= 1
            
            # Find the end of the peak (when it returns to the baseline)
            end = peak
            while end < num_timepoints and signal[end] > baseline:
                end += 1
            
            # Update the mask to exclude the peak window
            mask[start:end] = False
        
        # Collect noise data excluding windows around peaks
        noise_data.append(signal[mask])
    
    return noise_data

def skeletonize(template, locations, params, plot_dir, fresh_plots, unit_id):
    # Ensure the plot directory exists
    frame_dir = os.path.join(plot_dir, f"frames_unit_{unit_id}")
    os.makedirs(frame_dir, exist_ok=True)
    
    # Template Propagation - show signal propagation through channels selected
    skeleton_params = params.copy()

    skeleton_params.update({
        'detect_threshold': 0.05,
        'remove_isolated': False,
        'peak_std_threshold': None, 
        'init_delay': 0.1,
    })
    gtr = GraphAxonTracking(template, locations, 10000, **skeleton_params)
    gtr.select_channels()
    try: 
        plot_template_propagation(gtr, plot_dir, unit_id, title='test_prop', fresh_plots=fresh_plots)
    except Exception as e: 
        plt.close()
    # Template Plot
    try:
        kwargs = {
            'save_path': f"{plot_dir}/template_ploted_{unit_id}.png",
            'title': f'Template {unit_id}',
            'fresh_plots': True,
            'template': gtr.template,
            'locations': gtr.locations,
            'lw': 0.1, # line width
        }
        plot_template_wrapper(**kwargs)
    except Exception as e:
        plt.close()

    direct_links = []
    selected_channels = gtr.selected_channels

    # Loop through selected channels from last to first
    for i in range(len(selected_channels) - 1, 0, -1):
        current_peak = selected_channels[i]
        next_peak = selected_channels[i - 1]
        distance = np.linalg.norm(gtr.locations[current_peak] - gtr.locations[next_peak])
        if distance <= 100:
            direct_links.append((current_peak, next_peak))

            # Plot the direct links incrementally
            plt.figure(figsize=(10, 8))
            plt.scatter(gtr.locations[:, 0], gtr.locations[:, 1], c='gray', label='All Channels')
            plt.scatter(gtr.locations[selected_channels, 0], gtr.locations[selected_channels, 1], c='blue', label='Selected Channels')
            
            for link in direct_links:
                start, end = link
                plt.plot([gtr.locations[start, 0], gtr.locations[end, 0]], 
                         [gtr.locations[start, 1], gtr.locations[end, 1]], 
                         c='red', lw=2)

            plt.xlabel('X Coordinate')
            plt.ylabel('Y Coordinate (Inverted)')
            plt.title(f'Direct Links between Channels for Unit {unit_id} - Frame {len(direct_links)}')
            plt.legend()
            plt.savefig(os.path.join(frame_dir, f"frame_{len(direct_links)}.png"))
            plt.close()

    return direct_links

def approx_milos_tracking(template_dvdt, locations, params, plot_dir, fresh_plots, **kwargs):
    #skeletonize
    unit_id = 0
    skeleton = skeletonize(template_dvdt, locations, params, plot_dir, fresh_plots, unit_id)
    
    simulated_background_signals = estimate_background_noise(template_dvdt, peak_threshold=params['detect_threshold'])
    noise = np.array([np.std(signal) for signal in simulated_background_signals])
    noise = np.nanmean(noise)

    #First pass
    std_level = 9
    params9 = params.copy()
    params9.update({
        'detection_type': 'absolute',
        'detect_threshold': noise*std_level,
        'kurt_threshold': params['kurt_threshold']*std_level,
        'remove_isolated': False,
        'peak_std_threshold': None,
        'init_delay': 0.05,
        'max_distance_to_init': 4000.0,
        'max_distance_for_edge': 4000.0,
        'edge_dist_amp_ratio': 1,
    })
    milosh_dir = Path(os.path.join(plot_dir, 'milos_frames'))
    if os.path.exists(milosh_dir) is False: os.makedirs(milosh_dir)
    suffix = 'frame1'
    gtr9 = GraphAxonTracking(template_dvdt, locations, 10000, **params9)
    gtr9.select_channels()
    plot_selected_channels_helper(f"Selected after detection threshold: {gtr9._detect_threshold} uV", np.array(list(gtr9._selected_channels_detect)), gtr9.locations, f"detection_threshold{suffix}.png", milosh_dir, fresh_plots=fresh_plots)
    plot_selected_channels_helper(f"Selected after kurtosis threshold: {gtr9._kurt_threshold}", np.array(list(gtr9._selected_channels_kurt)), gtr9.locations, f"kurt_threshold{suffix}.png", milosh_dir, fresh_plots=fresh_plots)
    plot_selected_channels_helper(f"Selected after init delay threshold: {gtr9._init_delay} ms", np.array(list(gtr9._selected_channels_init)), gtr9.locations, f"init_delay_threshold{suffix}.png", milosh_dir, fresh_plots=fresh_plots)
    plot_selected_channels_helper("Selected after all thresholds", gtr9.selected_channels, gtr9.locations, f"all_thresholds{suffix}.png", milosh_dir, fresh_plots=fresh_plots)
    print()