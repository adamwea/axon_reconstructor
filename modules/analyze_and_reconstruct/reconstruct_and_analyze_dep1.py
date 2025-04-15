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
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import time
from axon_velocity.axon_velocity import GraphAxonTracking, get_default_graph_velocity_params

'''Refactored Functions'''
def reconstruct_and_analyze(templates, analysis_options=None, recon_dir=None, logger=None, params=None, 
                            n_jobs=None, skip_failed_units=True, unit_limit=None, parallel=True, debug_mode=False, **kwargs):
    # debugging?
    #debug_mode = True #comment out when not debugging
    if debug_mode: parallel = False
    #else: parallel = True
    
    """
    Main function to orchestrate reconstruction and analysis with modular subfunctions.
    """

    def setup_directories(recon_dir):
        """Set up and return paths for reconstruction, GTR, and analytics."""
        stream_recon_dir = Path(recon_dir) / "stream_recon"
        gtr_dir = Path(recon_dir) / "gtr"
        analysis_json_dir = Path(recon_dir) / "analysis_json"
        stream_recon_dir.mkdir(parents=True, exist_ok=True)
        gtr_dir.mkdir(parents=True, exist_ok=True)
        analysis_json_dir.mkdir(parents=True, exist_ok=True)
        return stream_recon_dir, gtr_dir, analysis_json_dir

    def submit_tasks(executor, unit_templates, gtr_dir, analysis_options, params, unit_limit, logger):
        """Submit reconstruction tasks to the executor."""
        futures = {}
        for unit_id, unit_template in unit_templates.items():
            if unit_limit is None or len(futures) < unit_limit:
                future = executor.submit(
                    process_unit_for_analysis, 
                    unit_id, unit_template, gtr_dir, analysis_options, params, logger
                )
                futures[future] = unit_id
        return futures

    def process_results(futures, analysis_json_dir, logger):
        """Process completed tasks and collect results."""
        results = {}
        for future in as_completed(futures):
            unit_id = futures[future]
            try:
                result = future.result()
                results[unit_id] = result
                logger.info(f"Unit {unit_id} processed successfully.")
            except Exception as e:
                logger.error(f"Error processing unit {unit_id}: {e}")
        return results

    def aggregate_analytics(results, stream_recon_dir, stream_id):
        def analysis_results_to_dataframe(analysis_results):
            flattened_data = []

            # for unit_id, results in analysis_results.items():
            #     flattened_entry = {'unit_id': unit_id}
                
            #     for key, value in results.items():
            #         flattened_entry[key] = value
                
            #     flattened_data.append(flattened_entry)
            
            exploded_data = []
            for unit_id, row in analysis_results.items():
                max_length = max(len(v) if isinstance(v, list) else 1 for v in row.values())
                for i in range(max_length):
                    new_entry = {'unit_id': row['unit_id'], 'branch_id': i}
                    for key, value in row.items():
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
        
        """Aggregate analytics and save to CSV."""
        #sort results by key (unit_id)
        results = dict(sorted(results.items()))
        for template_type in ['vt', 'dvdt', 'milos']:
            agg_template_analytics = {
                unit_id: data['reconstruction_analytics'][template_type]
                for unit_id, data in results.items()
                if template_type in data.get('reconstruction_analytics', {})
            }
            #include unit_id in the dataframe
            # for unit_id, data in results.items():
            #     if template_type in data.get('reconstruction_analytics', {}):
            #         agg_template_analytics[unit_id]['unit_id'] = unit_id
            
            # if agg_template_analytics == {}:
            #     print(f"No analytics data found for stream {stream_id} and template {template_type}")
            #     continue
            
            try: 
                df = analysis_results_to_dataframe(agg_template_analytics)
                
                # df = pd.DataFrame.from_dict(data, orient='index')
                
                # # reset index to avoid duplicate labels
                # df = df.reset_index(drop=True)
                
                # # explode the 'branch_ids' column
                # df = df.explode('branch_ids').reset_index(drop=True)
                
                # output_file = stream_recon_dir / f"agg_{stream_id}_{template_type}_axon_analytics.csv"
                output_dir = os.path.dirname(stream_recon_dir)
                output_dir = Path(output_dir)
                output_file = output_dir / f"agg_{stream_id}_{template_type}_axon_analytics_v2.csv"
                df.to_csv(output_file, index=False)
                print(f"Saved analytics data for stream {stream_id} and template {template_type} to {output_file}")
            except Exception as e:
                print(f"Error saving analytics data for stream {stream_id} and template {template_type}: {e}")
                pass

    def process_units(unit_templates, gtr_dir, analysis_options, params, unit_limit, analysis_json_dir, logger):
        """Process units either in parallel or serially."""
        successful_units = 0
        results = {}
        if parallel:
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                if unit_limit is None: unit_limit = len(unit_templates)
                while successful_units < unit_limit:
                    futures = submit_tasks(
                        executor, unit_templates, gtr_dir, analysis_options, params, unit_limit - successful_units, logger
                    )
                    new_results = process_results(futures, analysis_json_dir, logger)
                    results.update(new_results)
                    successful_units += len(new_results)
        else:
            for unit_id, unit_template in unit_templates.items():
                if unit_limit is not None and successful_units >= unit_limit:
                    break
                try:
                    result = process_unit_for_analysis(
                        unit_id, unit_template, gtr_dir, analysis_options, params, logger
                    )
                    results[unit_id] = result
                    successful_units += 1
                    logger.info(f"Unit {unit_id} processed successfully.")
                except Exception as e:
                    logger.error(f"Error processing unit {unit_id}: {e}")
        return results

    # Main processing logic
    total_units = sum(
        len(stream_templates['units']) for tmps in templates.values() for stream_templates in tmps['streams'].values()
    )
    progress_bar = tqdm(total=total_units, desc="Reconstructing units")

    for key, tmps in templates.items():
        for stream_id, stream_templates in tmps['streams'].items():
            unit_templates = stream_templates['units']
            stream_recon_dir, gtr_dir, analysis_json_dir = setup_directories(recon_dir)
            
            results = process_units(
                unit_templates, gtr_dir, analysis_options, params, unit_limit, analysis_json_dir, logger
            )
            try: aggregate_analytics(results, stream_recon_dir, stream_id)
            except Exception as e: 
                logger.error(f"Error aggregating analytics: {e}")
                pass
            progress_bar.update(len(results))
    
    #plot all reconstructions together
    # print("Plotting all reconstructions together...")
    # # from submodules.axon_velocity_fork import axon_velocity as av
    # # from submodules.axon_velocity_fork.axon_velocity import plotting
    # from axon_velocity import axon_velocity as av
    # from axon_velocity.axon_velocity import plotting
    
    # # create a grid that's 4000 by 2000 um with a pitch of 17.5 um
    # locations_mea1k = np.array([[x, y] for x in range(0, 4000, 17) for y in range(0, 2000, 17)])
    # probe_mea1k = av.plotting.get_probe(locations_mea1k)

    # fig_mea1k, ax = plt.subplots(figsize=(10, 7))
    # _ = plotting.plot_probe(probe_mea1k, ax=ax, contacts_kargs={"alpha": 0.1}, probe_shape_kwargs={"alpha": 0.1})
    # ax.axis("off")

    # i = 0
    # i_sel = 0
    # cmap = "tab20"
    # cm = plt.get_cmap(cmap)
    # for i, gtr in gtrs_mea1k.items():
        
    #     if i in mea1k_selected_unit_idxs:
    #         color = f"C{i_sel}"
    #         lw = 3
    #         alpha = 1
    #         zorder = 10
    #         i_sel += 1
    #     else:
    #         color = cm(i / len(gtrs_mea1k))
    #         lw = 1
    #         alpha = 1
    #         zorder = 1
    #     if len(gtr.branches) > 0:
    #         ax.plot(gtr.locations[gtr.init_channel, 0], gtr.locations[gtr.init_channel, 1], 
    #                 marker="o", markersize=5, color=color, alpha=alpha, zorder=zorder)

    #         if i not in mea1k_selected_unit_idxs:
    #             # for visualization purposes, plot raw branches
    #             for b_i, path in enumerate(gtr._paths_raw):
    #                 if b_i == 0:
    #                     ax.plot(gtr.locations[path, 0], gtr.locations[path, 1], marker="", color=color,
    #                             lw=lw, alpha=alpha, zorder=zorder, label=i)
    #                 else:
    #                     ax.plot(gtr.locations[path, 0], gtr.locations[path, 1], marker="", color=color,
    #                             lw=lw, alpha=alpha, zorder=zorder)
    #         else:
    #             for b_i, br in enumerate(gtr.branches):
    #                 if b_i == 0:
    #                     ax.plot(gtr.locations[br["channels"], 0], gtr.locations[br["channels"], 1], marker="", 
    #                             color=color, lw=lw, alpha=alpha, zorder=zorder, label=i)
    #                 else:
    #                     ax.plot(gtr.locations[br["channels"], 0], gtr.locations[br["channels"], 1], marker="", 
    #                             color=color, lw=lw, alpha=alpha, zorder=zorder)

    # ax.plot([0, 500], [1900, 1900], color="k", marker="|")
    # ax.text(20, 1950, "500$\mu$m", color="k", fontsize=18)
    # ax.set_title("")
    
    
    progress_bar.close()
    logger.info("Reconstruction and analysis complete.")

def process_unit_for_analysis_dep(unit_id, unit_templates, recon_dir, analysis_options, params, 
                              #failed_units, failed_units_file, 
                              logger=None):
    start = time.time()
    print(f'Analyzing unit {unit_id}...')
    
    template_data = {
        'unit_id': unit_id,
        'vt_template': unit_templates.get('merged_template', None),
        'dvdt_template': get_time_derivative(unit_templates.get('merged_template', None), sampling_rate=10000),
        'milos_template': [],  # Placeholder for milos template
        'channel_locations': unit_templates.get('merged_channel_locs', None),
    }

    reconstruction_data = {}
    reconstruction_analytics = {}

    for template_type in ['vt', 'dvdt', 'milos']:
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
                # failed_units.add(unit_id)
                # save_failed_units(failed_units, failed_units_file)
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

def process_unit_for_analysis(unit_id, unit_templates, recon_dir, analysis_options, params, logger=None):
    """
    Orchestrates the processing of a single unit, including template extraction,
    reconstruction, analytics computation, and plotting.
    """
    start = time.time()
    print(f"Analyzing unit {unit_id}...")

    # Extract template data
    template_data = extract_template_data(unit_id, unit_templates)

    # Initialize data storage
    reconstruction_data = {}
    reconstruction_analytics = {}

    # Process each template type
    #for template_type in ['vt', 'dvdt', 'milos']: #only doing 'vt' and 'dvdt' for now
    for template_type in ['vt', 'dvdt']:
        try:
            gtr = reconstruct_template(template_data, template_type, recon_dir, params, logger)
            reconstruction_data[template_type] = gtr
            reconstruction_analytics[template_type] = compute_reconstruction_analytics(template_data, gtr)
        except Exception as e:
            if logger:
                logger.error(f"Failed to process unit {unit_id} with template {template_type}: {e}")
            else:
                print(f"Failed to process unit {unit_id} with template {template_type}: {e}")
            continue

    # Generate plots
    generate_plots(template_data, reconstruction_data, reconstruction_analytics, recon_dir, unit_id)

    print(f"Finished analyzing unit {unit_id} in {time.time() - start:.2f} seconds")
    
    result = {
        'template_data': template_data,
        'reconstruction_data': reconstruction_data,
        'reconstruction_analytics': reconstruction_analytics,
    }
    
    return result

def extract_template_data(unit_id, unit_templates):
    """
    Extracts template data from the given unit templates.
    """
    # return {
    #     'unit_id': unit_id,
    #     'vt_template': unit_templates.get('merged_template'),
    #     'dvdt_template': get_time_derivative(unit_templates.get('merged_template'), sampling_rate=10000),
    #     'milos_template': [],  # Placeholder for milos template
    #     'channel_locations': unit_templates.get('merged_channel_locs'),
    # }    
    return {
        'unit_id': unit_id,
        'vt': unit_templates.get('merged_template', None),
        'dvdt': get_time_derivative(unit_templates.get('merged_template', None), sampling_rate=10000),
        'milos': [],  # Placeholder for milos template
        'channel_locations': unit_templates.get('merged_channel_locs', None),
    }
    
def reconstruct_template(template_data, template_type, recon_dir, params, logger=None):
    """
    Reconstructs the template using graph-based axon tracking.
    """
    try:
        gtr, av_params = build_graph_axon_tracking(
            template_data, recon_dir=recon_dir, load_gtr=True, params=params, template_type=template_type
        )
    except:
        gtr, av_params = build_graph_axon_tracking(
            template_data, recon_dir=recon_dir, load_gtr=False, params=params, template_type=template_type
        )
    
    if recon_dir:
        template_recon_dir = Path(recon_dir) / template_type
        template_recon_dir.mkdir(parents=True, exist_ok=True)
        gtr_file = template_recon_dir / f"{template_data['unit_id']}_gtr_object.dill"
        with open(gtr_file, 'wb') as f:
            dill.dump(gtr, f)
    return gtr

def compute_reconstruction_analytics(template_data, gtr):
    """
    Computes analytics for the given reconstruction graph.
    """
    
    branches = gtr.branches
    
    analytics = {
        'unit_id': template_data['unit_id'],
        'num_branches': len(gtr.branches),
        'channels_in_template': len(template_data['channel_locations']),
        'template_rect_area': calculate_template_rectangular_area(template_data),
        'template_density': len(template_data['channel_locations']) / calculate_template_rectangular_area(template_data),
        'axon_length': gtr.compute_path_length(gtr.branches[0]['channels']),
        
        #total axon matter - computed as summed length of all branches
        'total_axon_matter': sum([gtr.compute_path_length(branch['channels']) for branch in gtr.branches]),
        
        'maximum_branch_order': max(
            calculate_branch_order(branch['channels'], gtr._branching_points) for branch in gtr.branches
        ),
        
        #include empty column to seperate total axon recon stats from branch specific stats
        '': '',
        
        #'branch_id': [index for index, branch in enumerate(gtr.branches)],
        #'velocity': {index: branch['velocity'] for index, branch in enumerate(gtr.branches)},
        #'offset': {index: branch['offset'] for index, branch in enumerate(gtr.branches)},
        #'r_square': {index: branch['r2'] for index, branch in enumerate(gtr.branches)},
        #'pval': {index: branch['pval'] for index, branch in enumerate(gtr.branches)},
        
        'branch_ids': [index for index, branch in enumerate(gtr.branches)],
        'velocities': [branch['velocity'] for branch in gtr.branches],
        'offsets': [branch['offset'] for branch in gtr.branches],
        'r_squares': [branch['r2'] for branch in gtr.branches],
        'pvals': [branch['pval'] for branch in gtr.branches],
        
        'branch_lengths': [gtr.compute_path_length(branch['channels']) for branch in gtr.branches],
        'channels_per_branch': [len(branch['channels']) for branch in gtr.branches],
        'branch_channel_density': [gtr.compute_path_length(branch['channels']) / len(branch['channels']) for branch in gtr.branches],
        # 'branch_orders': {
        #     index: calculate_branch_order(branch['channels'], gtr._branching_points)
        #     for index, branch in enumerate(gtr.branches)
        # },
        # do branch order as a list instead:
        'branch_orders': [calculate_branch_order(branch['channels'], gtr._branching_points) for branch in gtr.branches],
    }
    
    return analytics

def generate_plots(template_data, reconstruction_data, reconstruction_analytics, recon_dir, unit_id): #generate a list of different types of plots with the data
    """
    Generates plots for the given unit's template and reconstruction data.
    """
    kwargs = {
        'template_data': template_data,
        'reconstruction_data': reconstruction_data,
        'reconstruction_analytics': reconstruction_analytics,
        'recon_dir': recon_dir,
        'unit_id': unit_id,
    }

    # try:
    #     generate_propogation_plots(**kwargs)
    # except Exception as e:
    #     print(f"Failed to generate propagation plots for unit {unit_id}: {e}")
    
    try:
        generate_reconstruction_summary_plots(**kwargs)
    except Exception as e:
        print(f"Failed to generate reconstruction summary plots for unit {unit_id}: {e}")
    
    # Placeholder for additional plot functions
    # try: generate_additional_plot(**kwargs)
    # except Exception as e: print(f"Failed to generate additional plot for unit {unit_id}: {e}")
    
    print(f"Finished generating plots for unit {unit_id}")

def generate_reconstruction_summary_plots(**kwargs): #make identical summary plots for each template type
    """
    Generates a summary of the reconstruction results.
    """
    template_data = kwargs.get('template_data', None)
    reconstruction_data = kwargs.get('reconstruction_data', None)
    reconstruction_analytics = kwargs.get('reconstruction_analytics', None)
    recon_dir = kwargs.get('recon_dir', None)
    unit_id = kwargs.get('unit_id', None)
    
    for template_type in ['vt', 'dvdt', 'milos']:
        if template_type not in reconstruction_data.keys():
            continue
        
        gtr = reconstruction_data[template_type]
        analytics = reconstruction_analytics[template_type]
        template = template_data[template_type]
        locations = template_data['channel_locations']
        selected_channels = gtr.selected_channels
        
        if template_type == 'vt':
            build_reconstruction_summary_fig(gtr, template, selected_channels, locations, recon_dir, unit_id, template_type)
        elif template_type == 'dvdt':
            build_reconstruction_summary_fig(gtr, template, selected_channels, locations, recon_dir, unit_id, template_type)
        elif template_type == 'milos':
            #plot_template_propagation(gtr, analytics, recon_dir, unit_id, template_type)
            print(f"Milos method not yet implemented")
            
    print(f"Finished generating reconstruction summary plots for unit {unit_id}")

def build_reconstruction_summary_fig_dep(gtr, template, selected_channels, locations, recon_dir, unit_id, template_type):
    """
    Assemble a reconstruction summary figure using pre-saved individual plots.

    Parameters:
        gtr: Data object containing information for plotting.
        template: Template data for the unit.
        selected_channels: Channels selected for analysis.
        locations: Locations of the channels or data points.
        recon_dir: Directory to save the reconstruction summary figure.
        unit_id: Identifier for the unit being analyzed.
        template_type: Type of template being used.
    """
    # Save individual plots with custom figure sizes
    #width = round(16/3
    save_individual_plot(
        propogation_plot_wrapper, gtr, template, selected_channels, locations, gain=8,
        #fig_kwargs={'figwidth': 9, 'figheight': 5},
        #fig_kwargs={'figsize': (9, 5)},
        fig_size=(5, 15),
        output_dir=recon_dir,
        file_prefix=f"{unit_id}_{template_type}_propagation_plot",
        unit_id=unit_id, template_type=template_type
    )
    
    def clean_branches_wrapper(gtr, plot_full_template=False, ax=None, cmap="rainbow",
                            plot_bp=False, branch_colors=None, 
                            unit_id=None, #lazy fix. should be removed
                            template_type=None, #lazy fix. should be removed
                            ):
        
        try:
            ax = gtr.plot_clean_branches(plot_full_template=plot_full_template, ax=ax, cmap=cmap,
                            plot_bp=plot_bp, branch_colors=branch_colors)
            return ax
        except:
            pass
    
    save_individual_plot(
        clean_branches_wrapper, gtr, fig_size=(12, 6), plot_full_template=False,
        output_dir=recon_dir,
        file_prefix=f"{unit_id}_{template_type}_clean_branches",
        unit_id=unit_id, template_type=template_type
    )
    
    def raw_branches_plot_wrapper(gtr, plot_full_template=False, ax=None, cmap="rainbow",
                            plot_bp=False, 
                            #branch_colors=None, 
                            unit_id=None, #lazy fix. should be removed
                            template_type=None, #lazy fix. should be removed
                            ):
        
        try:
            ax = gtr.plot_raw_branches(plot_full_template=plot_full_template, ax=ax, cmap=cmap,
                            plot_bp=plot_bp, 
                            #branch_colors=branch_colors
                            )
            return ax
    
        except:
            pass

    save_individual_plot(
        raw_branches_plot_wrapper, gtr, fig_size=(12, 6), plot_full_template=False,
        output_dir=recon_dir,
        file_prefix=f"{unit_id}_{template_type}_raw_branches",
        unit_id=unit_id, template_type=template_type
    )
    
    def velocity_plot_wrapper(gtr, fig=None, plot_outliers=False, cmap='rainbow', markersize=10, alpha=0.3,
                            alpha_outliers=0.7, fs=15, lw=2, markersize_out=10):
        # Ensure axes are cleared before plotting
        for ax in fig.axes:
            ax.remove()

        fig = gtr.plot_velocities(fig=fig, plot_outliers=plot_outliers, cmap=cmap,
                                markersize=markersize, alpha=alpha, alpha_outliers=alpha_outliers,
                                fs=fs, lw=lw, markersize_out=markersize_out)
        return fig

    fig_size = (5, 15)
    fig, ax = plt.subplots(figsize=fig_size)  # Define figure with initial axis
    fig = velocity_plot_wrapper(gtr, fig=fig, plot_outliers=True)

    # Set x-ticks explicitly to avoid overlaps
    #ax = fig.axes[0]  # Access the existing axis
    #ax.set_xticks(np.arange(0, len(gtr.branches), 1))
    #ax.set_xticklabels(np.arange(0, len(gtr.branches), 1))

    # Save plot
    output_dir = recon_dir
    file_prefix = f"{unit_id}_{template_type}_velocity_plot"
    output_path = output_dir / f"{file_prefix}.png"
    fig.savefig(output_path, dpi=600)

    pdf_path = output_dir / f"{file_prefix}.pdf"
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig)
    plt.close(fig)
    print(f"Saved {file_prefix} as PNG and PDF to {output_dir}")

    # Assemble the final figure
    fig = plt.figure(figsize=(16, 9))
    gs = plt.GridSpec(3, 3, figure=fig, width_ratios=[1, 1, 1], height_ratios=[1, 1, 1])

    # Left: Propagation Plot
    ax0 = fig.add_subplot(gs[:, 0])
    ax0.set_title(f"Unit {unit_id} - Propagation Plot")
    ax0.imshow(plt.imread(recon_dir / f"{unit_id}_{template_type}_propagation_plot.png"))
    ax0.axis("off")

    # Middle Top: Lean Branches Plot
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.set_title("Lean Branches")
    ax1.imshow(plt.imread(recon_dir / f"{unit_id}_{template_type}_lean_branches.png"))
    ax1.axis("off")

    # Middle Bottom: Raw Branches Plot
    ax2 = fig.add_subplot(gs[1, 1])
    ax2.set_title("Raw Branches")
    ax2.imshow(plt.imread(recon_dir / f"{unit_id}_{template_type}_raw_branches.png"))
    ax2.axis("off")

    # Right: Velocity Plot
    ax3 = fig.add_subplot(gs[:, 2])
    ax3.set_title(f"Unit {unit_id} - Velocity Plot")
    ax3.imshow(plt.imread(recon_dir / f"{unit_id}_{template_type}_velocity_plot.png"))
    ax3.axis("off")

    # Adjust layout for aesthetics
    plt.tight_layout()

    # Save the final figure
    final_png_path = recon_dir / f"{unit_id}_reconstruction_summary.png"
    final_pdf_path = recon_dir / f"{unit_id}_reconstruction_summary.pdf"
    plt.savefig(final_png_path, dpi=300)
    with PdfPages(final_pdf_path) as pdf:
        pdf.savefig(fig)
    plt.close(fig)

    print(f"Saved reconstruction summary as PNG and PDF to {recon_dir}")

def save_individual_plot_dep(wrapper_func, *args, 
                         #fig_kwargs, 
                         fig_size,
                         output_dir, file_prefix, **kwargs):
    """
    Save an individual plot as PNG and PDF using a wrapper function.

    Parameters:
        wrapper_func: Function to create the plot.
        *args: Positional arguments to pass to the wrapper function.
        fig_kwargs: Dictionary of figure customization options (e.g., figsize).
        output_dir: Directory to save the files.
        file_prefix: Prefix for the saved file names.
        **kwargs: Keyword arguments to pass to the wrapper function.
    """
    # Create the figure and axis
    #fig, ax = plt.subplots(**fig_kwargs)
    fig, ax = plt.subplots(figsize=fig_size)
    wrapper_func(ax=ax, *args, **kwargs)
    plt.tight_layout()

    # Save PNG and PDF
    png_path = output_dir / f"{file_prefix}.png"
    pdf_path = output_dir / f"{file_prefix}.pdf"
    fig.savefig(png_path, dpi=600)
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig)
    plt.close(fig)
    print(f"Saved {file_prefix} as PNG and PDF to {output_dir}")

def build_reconstruction_summary_fig(gtr, template, selected_channels, locations, recon_dir, unit_id, template_type):
    """
    Assemble a reconstruction summary figure using pre-saved individual plots.

    Parameters:
        gtr: Data object containing information for plotting.
        template: Template data for the unit.
        selected_channels: Channels selected for analysis.
        locations: Locations of the channels or data points.
        recon_dir: Directory to save the reconstruction summary figure.
        unit_id: Identifier for the unit being analyzed.
        template_type: Type of template being used.
    """
    def save_individual_plot(wrapper_func, *args, fig_size, output_dir, file_prefix, **kwargs):
        """Helper function to save individual plots as PNG and PDF."""
        fig, ax = plt.subplots(figsize=fig_size)
        try:
            kwargs['ax'] = ax
            #kwargs['fig'] = fig
            kwargs.pop('fig', None)
            ax = wrapper_func(*args, **kwargs)
        except Exception as e:
            try:
                #kwargs['ax'] = ax
                kwargs['fig'] = fig
                kwargs.pop('ax', None)
                fig = wrapper_func(*args, **kwargs)
            except Exception as e:
                try:
                    #both
                    kwargs['ax'] = ax
                    kwargs['fig'] = fig
                    ax = wrapper_func(*args, **kwargs)
                except Exception as e:
                    #neither
                    kwargs.pop('ax', None)
                    kwargs.pop('fig', None)
                    ax = wrapper_func(*args, **kwargs)
                
        #wrapper_func(ax=ax, fig=fig, *args, **kwargs)
        plt.tight_layout()
        
        # Save PNG and PDF
        png_path = output_dir / f"{file_prefix}.png"
        pdf_path = output_dir / f"{file_prefix}.pdf"
        fig.savefig(png_path, dpi=600)
        with PdfPages(pdf_path) as pdf:
            pdf.savefig(fig)
        plt.close(fig)
        print(f"Saved {file_prefix} as PNG and PDF to {output_dir}")

    # Save individual plots
    save_individual_plot(
        propogation_plot_wrapper, gtr, template, selected_channels, locations, gain=8, template_type=template_type, unit_id=unit_id,
        fig_size=(5, 15), output_dir=recon_dir,
        file_prefix=f"{unit_id}_{template_type}_propagation_plot"
    )
    #clear ax as needed:
    try:
        for ax in fig.axes:
            ax.remove()
    except:pass
    save_individual_plot(
        gtr.plot_clean_branches, fig_size=(12, 6), plot_full_template=False,
        output_dir=recon_dir, file_prefix=f"{unit_id}_{template_type}_clean_branches"
    )
    #clear ax as needed:
    try:
        for ax in fig.axes:
            ax.remove()
    except:pass
    save_individual_plot(
        gtr.plot_raw_branches,fig_size=(12, 6), plot_full_template=False,
        output_dir=recon_dir, file_prefix=f"{unit_id}_{template_type}_raw_branches"
    )
    #clear ax as needed:
    try:
        for ax in fig.axes:
            ax.remove()
    except:pass
    save_individual_plot(
        gtr.plot_velocities, fig_size=(5, 15), plot_outliers=True,
        output_dir=recon_dir, file_prefix=f"{unit_id}_{template_type}_velocity_plot"
    )

    # Assemble the final figure
    fig = plt.figure(figsize=(16, 9))
    gs = plt.GridSpec(3, 3, figure=fig, width_ratios=[1, 1, 1], height_ratios=[1, 1, 1])

    def add_subplot(gs_position, title, img_path):
        ax = fig.add_subplot(gs_position)
        ax.set_title(title, fontsize=12)
        ax.imshow(plt.imread(img_path))
        ax.axis("off")
        return ax

    add_subplot(gs[:, 0], f"Unit {unit_id} - Propagation Plot",
                recon_dir / f"{unit_id}_{template_type}_propagation_plot.png")
    add_subplot(gs[0, 1], "Clean Branches",
                recon_dir / f"{unit_id}_{template_type}_clean_branches.png")
    add_subplot(gs[1, 1], "Raw Branches",
                recon_dir / f"{unit_id}_{template_type}_raw_branches.png")
    add_subplot(gs[:, 2], f"Unit {unit_id} - Velocity Plot",
                recon_dir / f"{unit_id}_{template_type}_velocity_plot.png")

    plt.tight_layout()

    # Save final figure
    summary_png = recon_dir / f"{unit_id}_reconstruction_summary.png"
    summary_pdf = recon_dir / f"{unit_id}_reconstruction_summary.pdf"
    fig.savefig(summary_png, dpi=300)
    with PdfPages(summary_pdf) as pdf:
        pdf.savefig(fig)
    plt.close(fig)
    print(f"Saved reconstruction summary as PNG and PDF to {recon_dir}")

    # Save as a slide
    #save_as_slide(recon_dir / f"{unit_id}_reconstruction_summary.pptx", fig)

def propogation_plot_wrapper(gtr, template, selected_channels, locations, unit_id, template_type, ax=None, fig=None, gain=1):
    
    assert ax is not None, "An axis object must be provided."
    
    # get channels in reconstruction
    branches = gtr.branches
    selected_branch_channels = []
    for branch in branches:
        branch_channels = branch['channels']
        for channel in branch_channels:
            if channel in selected_channels:
                selected_branch_channels.append(channel)
    
    trans_template = template.T
    # #debug
    # fig, ax = plt.subplots()
    # #make fig 16:9 aspect ratio
    # fig.set_figwidth(16)
    # fig.set_figheight(9)    
    # #debug end
    ax = plot_template_propagation(trans_template, locations, selected_branch_channels, 
                                sort_templates=True, color='C0', color_marker='r', fig=fig, ax=ax,
                                line_thickness=1.0, gtr=gtr, 
                                label_channels=True, font_size=8, marker_size=5, #marker='o'
                                markerfacecolor='none',
                                gain = 8,
                                label_branches=True,
                                add_scale_bar=True, sampling_frequency=10000)

    #ax.figure.savefig(f"{unit_id}_{template_type}_propagation_plot.png", dpi=600)
    #print(f"Saved propogation plot for unit {unit_id} with template {template_type} to {unit_id}_{template_type}_propogation_plot.png")
    return ax

    ## old code---------------------------------------------------------------------
        # #debug code
        # fig2, ax2 = plt.subplots()
        # #make fig 2 16:9 aspect ratio
        # # fig2.set_figwidth(16)
        # # fig2.set_figheight(9)
        # # #fig2 = gtr.plot_branches()
        # # ax2 = gtr.plot_clean_branches(ax=ax2)
        # # fig2.savefig('branch_plot_clean.png', dpi=600)
        # # branches = gtr.branches if gtr else None
        # # fig2, ax2 = plt.subplots()
        # # #make fig 2 16:9 aspect ratio
        # # fig2.set_figwidth(16)
        # # fig2.set_figheight(9)
        # # ax2 = gtr.plot_raw_branches(ax=ax2)
        # # fig2.savefig('branch_plot_raw.png', dpi=600)
        # #debug code end

        # trans_template = template.T
        # fig, ax = plt.subplots()
        # # make fig very tall
        # #get channels to plot by getting channels in selected channels that are also in 
        # #gtr.branches
        # branches = gtr.branches
        # selected_branch_channels = []
        # for branch in branches:
        #     branch_channels = branch['channels']
        #     for channel in branch_channels:
        #         if channel in selected_channels:
        #             selected_branch_channels.append(channel)

        # len_channels = len(selected_branch_channels)
        # #matplotlib_inches_per_channel = 0.30
        # #fig_height = int(len_channels * matplotlib_inches_per_channel)
        # #fig.set_figheight(fig_height)
        # fig.set_figheight(8)
        # fig.set_figwidth(6)
        # ax = plot_template_propagation(trans_template, locations, selected_branch_channels, 
        #                             sort_templates=True, color='C0', color_marker='r', fig=fig, ax=ax,
        #                             line_thickness=1.0, gtr=gtr, 
        #                             label_channels=True, font_size=8, marker_size=5, #marker='o'
        #                             markerfacecolor='none',
        #                             gain = 5,
        #                             label_branches=True,
        #                             add_scale_bar=True, sampling_frequency=10000)
        # #debug_plot_template_propagation(ax)
        # # plt.savefig(recon_dir / f"{unit_id}_{template_type}_propogation_plot.png")
        # # print(f"Saved propogation plot for unit {unit_id} with template {template_type} to {recon_dir / f'{unit_id}_{template_type}_propogation_plot.png'}")

        # plt.tight_layout()
        # plt.savefig(f"{unit_id}_{template_type}_propogation_plot.png", dpi=600)
        # print(f"Saved propogation plot for unit {unit_id} with template {template_type} to {unit_id}_{template_type}_propogation_plot.png")
        #plot_axes[template_type] = ax

def plot_template_propagation(template, locations, selected_channels, sort_templates=False,
                              color=None, color_marker=None, fig=None, ax=None, 
                              line_thickness=1.0, gtr=None, label_branches=False, label_channels=False,
                              font_size=12, gain=1.0, marker_size=5, marker='o', markerfacecolor='none',
                              add_scale_bar=True, sampling_frequency=10000):
    '''
    This function generates propagation plots for validating templates submitted for axon tracking.

    Additional Features:
    - Adds branch-specific coloring using a gradient when label_branches is enabled.
    - Adds a dynamic scale bar based on pre-gain signal amplitude.
    '''
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import cm  # For gradient color map

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

    # Apply gradient for branch colors
    branch_colors = None
    if gtr and label_branches:
        num_branches = len(gtr.branches)
        branch_colors = cm.get_cmap('coolwarm', num_branches)  # Blue to red gradient

    # Dynamic line spacing adjustment
    ptp_glob = np.max(np.ptp(template_sorted, 1))
    
    for i, temp in enumerate(template_sorted):
        if gain != 1.0:
            temp = temp * gain
                    
        temp_shifted = temp + i * 1.5 * ptp_glob
        min_t = np.min(temp_shifted)
        min_idx = np.argmin(temp_shifted)
        
        # Assign marker color based on branch
        if label_branches and branch_colors:
            branch_idx = next((j for j, branch in enumerate(gtr.branches) if selected_channels[i] in branch['channels']), None)
            if branch_idx is not None:
                print(f"Channel {selected_channels[i]} is in branch {branch_idx}")
                marker_color = branch_colors(branch_idx)
            else:
                marker_color = color_marker  # Default if no branch
        else:
            marker_color = color_marker

        ax.plot(temp_shifted, color=color, linewidth=line_thickness)
        ax.plot(min_idx, min_t, marker=marker, color=marker_color, markersize=marker_size, markerfacecolor=marker_color)

        if label_channels:
            try:
                # Aesthetically appealing way to label channels
                max_idx = np.argmax(temp_shifted[:min_idx]) 
                mean_before_peak = np.mean(temp_shifted[:max_idx])
                if np.isnan(mean_before_peak):
                    raise ValueError("mean_before_peak is NaN")
            except (ValueError, IndexError):
                try:
                    mean_before_peak = np.mean(temp_shifted[:12])
                    if np.isnan(mean_before_peak):
                        raise ValueError("Fallback mean is NaN")
                except (ValueError, IndexError):
                    mean_before_peak = temp_shifted[0]

            ax.text(
                -0.1,  # Static x-coordinate for alignment
                mean_before_peak + 1,  # Dynamic y-coordinate
                f'Ch {selected_channels[i]}',
                ha='left',
                va='bottom',
                fontsize=font_size,
            )

    if gtr and label_branches:
        # Add a legend for branches
        legend_handles = [
            plt.Line2D(
                [0], [0],
                marker='o', color='w', markerfacecolor=branch_colors(i),
                markersize=10, label=f'Branch {i}'
            ) for i in range(len(gtr.branches))
        ]
        ax.legend(handles=legend_handles, loc='upper right', fontsize=font_size)

    if add_scale_bar:
        # Scale bar sizes (rounded for aesthetics)
        y_scale_bar_size = np.round(0.2 * ptp_glob, 1)  # Voltage in mV
        y_scale_bar_size = np.round(y_scale_bar_size/10)*10         #round y_scale_bar_size to nearest 10 or 5
        y_scale_bar_size_gain = y_scale_bar_size * gain  # Apply gain (if gain is 1 this does nothing...obviously)
        x_scale_bar_duration = 2  # 2 ms as the default
        x_scale_bar_duration = x_scale_bar_duration/1000  # Convert to seconds
        x_scale_bar_samples = int(x_scale_bar_duration * sampling_frequency)  # Convert time to samples
        x_scale_bar_size = x_scale_bar_samples / sampling_frequency * 1000  # Convert to ms

        # Get plot limits for positioning
        x_lim = ax.get_xlim()
        y_lim = ax.get_ylim()

        # Define scale bar position (bottom-right corner)
        bar_x = x_lim[1] - 0.05 * (x_lim[1] - x_lim[0]) - x_scale_bar_samples  # Slight padding
        bar_y = y_lim[0] + 0.05 * (y_lim[1] - y_lim[0])

        # Plot horizontal scale bar (time, x-axis)
        ax.plot(
            [bar_x, bar_x + x_scale_bar_samples], [bar_y, bar_y],
            color='k', linewidth=2
        )

        # Annotate horizontal scale bar
        ax.text(
            bar_x + x_scale_bar_samples / 2, bar_y - 0.05 * (y_lim[1] - y_lim[0]),
            f'{x_scale_bar_size:.1f} ms',
            ha='center', va='top', fontsize=font_size
        )

        # Plot vertical scale bar (voltage, y-axis)
        ax.plot(
            [bar_x, bar_x], [bar_y, bar_y + y_scale_bar_size_gain],
            color='k', linewidth=2
        )

        # Annotate vertical scale bar
        ax.text(
            bar_x - 0.05 * (x_lim[1] - x_lim[0]), bar_y + y_scale_bar_size_gain / 2,
            f'{y_scale_bar_size:.1f} uV',
            ha='right', va='center', fontsize=font_size
        )

    ax.axis('off')
    return ax

def generate_propogation_plots_dep(**kwargs):
    reconstruction_data = kwargs.get('reconstruction_data', None)
    reconstruction_analytics = kwargs.get('reconstruction_analytics', None)
    recon_dir = kwargs.get('recon_dir', None)
    unit_id = kwargs.get('unit_id', None)
    #template_type = kwargs.get('template_type')
    template_data = kwargs.get('template_data', None)
    plot_axes = kwargs.get('plot_axes', None)
    
    #fix template keys...
    template_data['vt'] = template_data['vt_template']
    template_data['dvdt'] = template_data['dvdt_template']
    template_data['milos'] = template_data['milos_template']
    
    for template_type in ['vt', 'dvdt', 'milos']:
        if template_type not in reconstruction_data.keys():
            continue
        
        gtr = reconstruction_data[template_type]
        analytics = reconstruction_analytics[template_type]
        template = template_data[template_type]
        locations = template_data['channel_locations']
        selected_channels = gtr.selected_channels
        

        if template_type == 'vt':
            generate_plot_template_summary(template, locations, selected_channels)
        elif template_type == 'dvdt':
            plot_template_propagation(trans_template, locations, selected_channels, 
                                      sort_templates=False, color='C0', color_marker='r', ax=None)
            plot_axes[template_type] = ax
        elif template_type == 'milos':
            #plot_template_propagation(gtr, analytics, recon_dir, unit_id, template_type)
            print(f"Milos method not yet implemented")
        else:
            print(f"Invalid template type: {template_type}")
            continue

def get_time_derivative(merged_template, 
                        #merged_template_filled, 
                        unit='seconds', sampling_rate=10000, axis=0):
    """
    Computes the time derivative of the input arrays along the specified axis.
    
    Parameters:
    - merged_template: numpy.ndarray
        The first input array where the y-axis describes channels and x-axis describes voltage(time) signals.
    - merged_template_filled: numpy.ndarray
        The second input array where the y-axis describes channels and x-axis describes voltage(time) signals.
    - unit: str, optional
        The time unit for the derivative, either 'seconds' or 'ms'. Default is 'seconds'.
    - sampling_rate: int, optional
        The sampling rate in samples per second. Default is 10000.
    - axis: int, optional
        The axis along which to compute the derivative. Default is 0 (assuming time signals along y-axis).
    
    Returns:
    - d_merged_template: numpy.ndarray
        The time derivative of merged_template.
    - d_merged_template_filled: numpy.ndarray
        The time derivative of merged_template_filled.
    """
    
    if unit == 'seconds':
        delta_t = 1.0 / sampling_rate
    elif unit == 'ms':
        delta_t = 1.0 / (sampling_rate / 1000)
    else:
        raise ValueError("Invalid unit. Choose either 'seconds' or 'ms'.")

    d_merged_template = np.diff(merged_template, axis=axis) / delta_t
    #d_merged_template_filled = np.diff(merged_template_filled, axis=axis) / delta_t
    return d_merged_template #, d_merged_template_filled

def build_graph_axon_tracking(template_data, load_gtr=False, recon_dir=None, params=None, template_type='dvdt'):
    gtr = None
    """Build the graph for axon tracking."""
    
    # if template_type == 'vt':
    #     template_key = 'vt_template'
    # elif template_type == 'dvt':
    #     template_key = 'dvdt_template'
    # elif template_type == 'milos':
    #     template_key = 'milos_template'
    # else:
    #     raise ValueError("Invalid template type. Choose from 'vt', 'dvt', or 'milos'.")
    
    #transposed_template, transposed_loc = transform_data(template_data[template_key], template_data['channel_locations'])
    transposed_template, transposed_loc = transform_data(template_data[template_type], template_data['channel_locations'])
    params = params or get_default_graph_velocity_params() # Default parameters for graph building if not provided
    unit_id = template_data['unit_id']
    
    # if recon_dir is not None, try to load the gtr object from file    
    if recon_dir:
        recon_dir = Path(recon_dir)
        recon_dir.mkdir(parents=True, exist_ok=True)
        gtr_file = recon_dir / f"{unit_id}_gtr_object.dill"
        
        if load_gtr and gtr_file.exists():
            assert gtr_file.exists() or not load_gtr, f"File {gtr_file} does not exist. Set load_gtr to False to build the graph."
            with open(gtr_file, 'rb') as f:
                gtr = dill.load(f)
            gtr._verbose = 1
            gtr.select_channels()
            gtr.build_graph()
            gtr.find_paths()
            gtr.clean_paths(remove_outliers=False)
            print(f"Loaded gtr object from {gtr_file}")
            return gtr, params
                # return gtr, params
    
    # Build the graph
    print(f"Reconstructing Axonal Morphology for unit {unit_id} using {template_type} template")
    start = time.time()
    gtr = GraphAxonTracking(transposed_template, transposed_loc, 10000, **params) #TODO: hard coded sampling rate...
    gtr._verbose = 1 # TODO: make this an option later
    gtr.select_channels()
    gtr.build_graph()
    gtr.find_paths()
    gtr.clean_paths(remove_outliers=False)
    if gtr.branches is not None:
        print(f"Found {len(gtr.branches)} branches")
    else:
        print("No branches found - reconstruction failed")   
    print(f"Reconstruction took {time.time() - start:.2f} seconds")           
    return gtr, params

def transform_data(merged_template, merged_channel_loc, merged_template_filled=None, merged_channel_filled_loc=None):
    transformed_template = merged_template.T  # array with time samples and amplitude values (voltage or v/s)
    trans_loc = np.array([[loc[0], loc[1] * -1] for loc in merged_channel_loc])
    
    # if merged_template_filled is not None: 
    #     transformed_template_filled = merged_template_filled.T  # array with time samples and amplitude values (voltage or v/s)
    # else: 
    #     transformed_template_filled = None
    # if merged_channel_filled_loc is not None: 
    #     trans_loc_filled = np.array([[loc[0], loc[1] * -1] for loc in merged_channel_filled_loc])
    # else: 
    #     trans_loc_filled = None
    return transformed_template, trans_loc

def calculate_template_rectangular_area(template_data):
    """
    Calculate the rectangular 2D area occupied by the channels in the template data.

    Parameters:
    template_data (dict): A dictionary containing channel locations under the key 'channel_loc'.

    Returns:
    int: The number of channels in the template.
    float: The rectangular 2D area occupied by the channels in square micrometers.
    """
    channels_in_template = len(template_data['channel_locations'])

    # Extract x and y coordinates of all channels
    x_coords = [loc[0] for loc in template_data['channel_locations']]
    y_coords = [loc[1] for loc in template_data['channel_locations']]

    # Calculate the minimum and maximum x and y coordinates
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)

    # Calculate the width and height of the rectangular area
    width = max_x - min_x
    height = max_y - min_y

    # Calculate the rectangular 2D area
    rectangular_area = width * height

    return rectangular_area

def calculate_branch_order(branch, branch_points): # Calculate the branch order for each branch
    order = 0
    for point in branch:
        if point in branch_points:
            order += 1
    return order

def process_individual_unit_plots(template_data, reconstruction_data, reconstruction_analytics, recon_dir, unit_id, template_type):
    
    kwargs = {
        'template_data': template_data,
        'reconstruction_data': reconstruction_data,
        'reconstruction_analytics': reconstruction_analytics,
        'recon_dir': recon_dir,
        'unit_id': unit_id,
        'template_type': template_type,
        'plot_axes': {}
    }
    
    ## plot function 1
    try: generate_propogation_plots(**kwargs)
    except Exception as e: print(f"Failed to generate propogation plots for unit {unit_id} with template {template_type} - Error: {e}")
    ## plot function 2
    ## plot function 3
    ## plot function 4
    ## ...etc
    
    pass

''' Functions'''  


        
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

# def load_failed_units(failed_units_file, skip_failed_units):
#     if skip_failed_units and failed_units_file.exists():
#         with open(failed_units_file, 'r') as f:
#             return set(json.load(f))
#     return set()

def setup_directories(recon_dir):
    stream_recon_dir = Path(recon_dir)
    stream_recon_dir.mkdir(parents=True, exist_ok=True)
    
    gtr_dir = stream_recon_dir / 'unit_gtr_objects'
    gtr_dir.mkdir(parents=True, exist_ok=True)
    
    analysis_json_dir = stream_recon_dir / 'unit_analytics'
    analysis_json_dir.mkdir(parents=True, exist_ok=True)
    
    return stream_recon_dir, gtr_dir, analysis_json_dir

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
    
''' deprecated code '''
def plot_template_propagation_stable(template, locations, selected_channels, sort_templates=False,
                              color=None, color_marker=None, fig=None, ax=None, 
                              line_thickness=1.0, gtr=None, label_branches=False, label_channels=False,
                              font_size=12, gain=1.0, marker_size=5, marker='o', markerfacecolor='none'):
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
    
    import matplotlib.pyplot as plt
    import numpy as np

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
    
    # # Apply gain to the templates
    # template_sorted = np.array([temp * gain for temp in template_sorted])

    # Dynamic line spacing adjustment
    ptp_glob = np.max(np.ptp(template_sorted, 1))
    
    for i, temp in enumerate(template_sorted):
        
        if gain != 1.0:
            temp = temp * gain
                    
        temp_shifted = temp + i * 1.5 * ptp_glob  # dist_peaks_sorted[i]
        min_t = np.min(temp_shifted)
        min_idx = np.argmin(temp_shifted)

        ax.plot(temp_shifted, color=color, linewidth=line_thickness)
        ax.plot(min_idx, min_t, marker=marker, color=color_marker, markersize=marker_size, markerfacecolor=markerfacecolor)

        if label_channels:
            
            ## aesthetically appealing way to label channels
            try:
                # Assume max peak comes first, then negative peak (typical AP waveform)
                max_idx = np.argmax(temp_shifted[:min_idx]) 
                mean_before_peak = np.mean(temp_shifted[:max_idx])
                
                # Check for NaN explicitly
                if np.isnan(mean_before_peak):
                    raise ValueError("mean_before_peak is NaN")
                
            except (ValueError, IndexError) as e:
                # Handle cases where the max value is the first value or mean is NaN
                print(f"Fallback for channel {selected_channels[i]} due to: {e}")
                try:
                    mean_before_peak = np.mean(temp_shifted[:12])  # Fallback to the first 12 values
                    if np.isnan(mean_before_peak):
                        raise ValueError("Fallback mean is NaN")
                except (ValueError, IndexError):
                    # Final fallback to the first value if all else fails
                    mean_before_peak = temp_shifted[0]
            
            #print(f"Adding label for channel {selected_channels[i]}")
            ax.text(
                -0.1,  # Static x-coordinate for alignment
                mean_before_peak+0.1,  # Dynamic y-coordinate
                f'Ch {selected_channels[i]}',
                ha='left',
                #va='center',
                va='bottom',
                fontsize=font_size,
                #transform=ax.transAxes
            )

    if gtr and label_branches:
        for i, branch in enumerate(gtr.branches):
            branch_channels = branch['channels']
            branch_loc = branch['location']
            branch_loc = branch_loc / np.linalg.norm(branch_loc)
            branch_loc = branch_loc * 0.9 * np.max([temp + i * 1.5 * ptp_glob * len(template_sorted), 1])
            branch_loc = branch_loc + locs_sorted[0]
            ax.text(branch_loc[0], branch_loc[1], f"{i}", fontsize=font_size, ha='right', va='center')

    ax.axis('off')
    return ax
