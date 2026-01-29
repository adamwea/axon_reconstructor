
def process_unit(unit_id, unit_templates, recon_dir, params, analysis_options, successful_recons, failed_recons, logger=None):
    if logger is not None: 
        logger.info(f'Processing unit {unit_id}')
    else: 
        print(f'Processing unit {unit_id}')

    templates = {
        'merged_template': unit_templates['merged_template'],
        'dvdt_merged_template': unit_templates['dvdt_merged_template'],
    }

    transformed_templates = {}
    for key, template in templates.items():
        if template is not None:
            transformed_template, transformed_template_filled, trans_loc, trans_loc_filled = transform_data(
                template, 
                unit_templates['merged_channel_loc'], 
                unit_templates['merged_template_filled'], 
                unit_templates['merged_channel_locs_filled']
            )
            transformed_templates[key] = (transformed_template, transformed_template_filled, trans_loc, trans_loc_filled)

    plot_dir = create_plot_dir(recon_dir, unit_id)
    analytics_data = {key: {
        'units': [], 'branch_ids': [], 'velocities': [], 'path_lengths': [], 'r2s': [], 'extremums': [], 
        'num_channels_included': [], 'channel_density': [], 'init_chans': []
    } for key in templates.keys()}

    x_coords = unit_templates['merged_channel_loc'][:, 0]
    y_coords = unit_templates['merged_channel_loc'][:, 1]
    width = np.max(x_coords) - np.min(x_coords)
    height = np.max(y_coords) - np.min(y_coords)
    area = width * height
    channel_density_value = len(unit_templates['merged_channel_loc']) / area  # channels / um^2

    failed_recons[unit_id] = []

    for key, (transformed_template, transformed_template_filled, trans_loc, trans_loc_filled) in transformed_templates.items():
        paths_files_generated = []
        suffix = key.split('_')[-1] if '_' in key else 'template'
        if 'dvdt' in key: suffix = 'dvdt'
        gtr = None
        # Generate Gtr object
        try: gtr = GraphAxonTracking(transformed_template, trans_loc, 10000, **params)
        except Exception as e:
            if logger: logger.info(f"unit {unit_id}_{suffix} failed to initialize GraphAxonTracking for {key}, error: {e}")

        if gtr is not None:
            analytics_data[key]['num_channels_included'].append(len(unit_templates['merged_channel_loc']))
            analytics_data[key]['channel_density'].append(channel_density_value)

            # Template Plot
            try:
                kwargs = {
                    'save_path': f"{plot_dir}/template_ploted_{unit_id}_{suffix}.png",
                    'title': f'Template {unit_id} {suffix}',
                    'fresh_plots': True,
                    'template': gtr.template,
                    'locations': gtr.locations,
                    'lw': 0.1, # line width
                }
                plot_template_wrapper(**kwargs)
                paths_files_generated.append(kwargs['save_path'])
            except Exception as e:
                if logger: 
                    logger.info(f"unit {unit_id}_{suffix} failed to plot template for {key}, error: {e}")
                plt.close()
            
            # Amplitude Maps
            try:
                title = f'{key}_{suffix}'
                generate_amplitude_map(transformed_template_filled, trans_loc_filled, plot_dir, title=f'{key}_{suffix}', fresh_plots=True, log=False, 
                                       cmap='terrain',
                                       )
                paths_files_generated.append(plot_dir / f"{title}_amplitude_map.png")
            except Exception as e:
                if logger: 
                    logger.info(f"unit {unit_id}_{suffix} failed to generate amplitude_map for {key}, error: {e}")
                plt.close()

            # Peak Latency Map
            try:
                generate_peak_latency_map(transformed_template_filled, trans_loc_filled, plot_dir, title=f'{key}_{suffix}', fresh_plots=True, log=False, 
                                       cmap='terrain',
                                       )
                paths_files_generated.append(f"{plot_dir}/{key}_{suffix}_peak_latency_map.png")
            except Exception as e:
                if logger: 
                    logger.info(f"unit {unit_id}_{suffix} failed to generate peak_latency_map for {key}, error: {e}")
                plt.close()
            
            # Select Channels
            try:
                fresh_plots = True
                gtr.select_channels()
                plot_selected_channels_helper(f"Selected after detection threshold: {gtr._detect_threshold} uV", np.array(list(gtr._selected_channels_detect)), gtr.locations, f"detection_threshold{suffix}.png", plot_dir, fresh_plots=fresh_plots)
                plot_selected_channels_helper(f"Selected after kurtosis threshold: {gtr._kurt_threshold}", np.array(list(gtr._selected_channels_kurt)), gtr.locations, f"kurt_threshold{suffix}.png", plot_dir, fresh_plots=fresh_plots)
                plot_selected_channels_helper(f"Selected after peak std threshold: {gtr._peak_std_threhsold} ms", np.array(list(gtr._selected_channels_peakstd)), gtr.locations, f"peak_std_threshold{suffix}.png", plot_dir, fresh_plots=fresh_plots)
                plot_selected_channels_helper(f"Selected after init delay threshold: {gtr._init_delay} ms", np.array(list(gtr._selected_channels_init)), gtr.locations, f"init_delay_threshold{suffix}.png", plot_dir, fresh_plots=fresh_plots)
                plot_selected_channels_helper("Selected after all thresholds", gtr.selected_channels, gtr.locations, f"all_thresholds{suffix}.png", plot_dir, fresh_plots=fresh_plots)
                paths_files_generated.append(f"{plot_dir}/detection_threshold{suffix}.png")
                paths_files_generated.append(f"{plot_dir}/kurt_threshold{suffix}.png")
                paths_files_generated.append(f"{plot_dir}/peak_std_threshold{suffix}.png")
                paths_files_generated.append(f"{plot_dir}/init_delay_threshold{suffix}.png")
                paths_files_generated.append(f"{plot_dir}/all_thresholds{suffix}.png")
            except Exception as e:
                if logger: 
                    logger.info(f"unit {unit_id}_{suffix} failed to select and plot channels for {key}, error: {e}")
                plt.close()

            switch = False
            if switch:
                gtr_milos = None
                if 'dvdt' in key:
                    kwargs = {
                        'template': transformed_template,
                        'locations': trans_loc,
                        'params': params,
                        'plot_dir': plot_dir,
                        'fresh_plots': True,
                    }
                    try: gtr_milos = approx_milos_tracking(**kwargs)
                    except Exception as e:
                        if logger: logger.info(f"unit {unit_id}_{suffix} failed to track axons in milos style for {key}, error: {e}")

            # Track Axons
            try: 
                gtr.track_axons()
            except Exception as e:
                if logger: logger.info(f"unit {unit_id}_{suffix} failed to track axons for {key}, error: {e}")

            # Template Propagation - show signal propagation through channels selected
            try:
                title = key
                fig_dir = plot_dir / f"{title}_template_propagation_unit_{unit_id}.png"
                plot_template_propagation(gtr, fig_dir, unit_id, title=key, figsize = (200, 50), 
                                          fresh_plots=True, linewidth=0.1, markersize=0.1, color_marker = 'r', markerfacecolor='none', 
                                          dpi=1000, max_overlapped_lines=10)
                paths_files_generated.append(fig_dir)
            except Exception as e:
                if logger: 
                    logger.info(f"unit {unit_id}_{suffix} failed to plot template propagation for {key}, error: {e}")
                plt.close()

            # Axon Reconstruction Heuristics
            try:
                generate_axon_reconstruction_heuristics(gtr, plot_dir, unit_id, suffix=f"_{suffix}", fresh_plots=True, figsize=(20, 10))
                paths_files_generated.append(plot_dir / f"axon_reconstruction_heuristics{suffix}.png")
            except Exception as e:
                if logger: 
                    logger.info(f"unit {unit_id}_{suffix} failed to generate axon_reconstruction_heuristics for {key}, error: {e}")
                plt.close()

            # Axon Reconstruction Raw
            try:
                generate_axon_reconstruction_raw(gtr, plot_dir, unit_id, recon_dir, successful_recons, suffix=f"_{suffix}", fresh_plots=True, plot_full_template=False)
                generate_axon_reconstruction_raw(gtr, plot_dir, unit_id, recon_dir, successful_recons, suffix=f"_{suffix}_full", fresh_plots=True, plot_full_template=True)
                paths_files_generated.append(plot_dir / f"axon_reconstruction_raw{suffix}.png")
                paths_files_generated.append(plot_dir / f"axon_reconstruction_raw{suffix}_full.png")
            except Exception as e:
                if logger: 
                    logger.info(f"unit {unit_id}_{suffix} failed to generate raw axon_reconstruction for {key}, error: {e}")
                plt.close()

            # Axon Reconstruction Clean
            try:
                generate_axon_reconstruction_clean(gtr, plot_dir, unit_id, recon_dir, successful_recons, suffix=f"_{suffix}", fresh_plots=True, plot_full_template=False)
                generate_axon_reconstruction_clean(gtr, plot_dir, unit_id, recon_dir, successful_recons, suffix=f"_{suffix}_full", fresh_plots=True, plot_full_template=True)
                paths_files_generated.append(plot_dir / f"axon_reconstruction_clean{suffix}.png")
                paths_files_generated.append(plot_dir / f"axon_reconstruction_clean{suffix}_full.png")
            except Exception as e:
                if logger: 
                    logger.info(f"unit {unit_id}_{suffix} failed to generate clean axon_reconstruction for {key}, error: {e}")
                plt.close()

            # Axon Reconstruction Velocities
            try:
                generate_axon_reconstruction_velocities(gtr, plot_dir, unit_id, suffix=f"_{suffix}")
                paths_files_generated.append(plot_dir / f"axon_reconstruction_velocities{suffix}.png")
            except Exception as e:
                if logger: 
                    logger.info(f"unit {unit_id}_{suffix} failed to generate axon_reconstruction_velocities for {key}, error: {e}")
                plt.close()

            switch = False
            if switch:
                try:
                    assert analysis_options['generate_animation'] == True, "generate_animation set to False. Skipping template map animation generation."
                    kwargs = {
                        'template': transformed_template_filled,
                        'locations': trans_loc_filled,
                        'gtr': gtr,
                        'elec_size':8, 
                        'cmap':'viridis',
                        'log': False,
                        'save_path': f"{plot_dir}/template_map_{unit_id}_{suffix}.gif",
                    }
                    play_template_map_wrapper(**kwargs)
                    paths_files_generated.append(kwargs['save_path'])
                except Exception as e:
                    if logger: 
                        logger.info(f"unit {unit_id}_{suffix} failed to play template map for {key}, error: {e}")
                    plt.close()

            try:
                assert len(gtr.branches)>0, f"No axon branches found, deleting all generated plots for unit {unit_id}_{suffix}"
                generate_axon_analytics(gtr, 
                                        analytics_data[key]['units'], 
                                        analytics_data[key]['branch_ids'], 
                                        analytics_data[key]['velocities'], 
                                        analytics_data[key]['path_lengths'], 
                                        analytics_data[key]['r2s'], 
                                        analytics_data[key]['extremums'],
                                        analytics_data[key]['init_chans'],
                                        analytics_data[key]['num_channels_included'],
                                        analytics_data[key]['channel_density'], 
                                        unit_id, 
                                        transformed_template, 
                                        trans_loc)
            except Exception as e:
                if logger: 
                    logger.info(f"unit {unit_id}_{suffix} failed to generate axon_analytics for {key}, error: {e}")
                for file in paths_files_generated:
                    try: Path(file).unlink()
                    except: pass
                analytics_data[key] = e

    return analytics_data