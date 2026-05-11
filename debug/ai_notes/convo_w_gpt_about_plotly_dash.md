ChatGPT

Um, it is my understanding, my un-researched understanding that Plotly is a package that can, like, you can upload data to it, um, or you can prepare data for it, and then, like, on a server, on like an HTML sort of interface, you can sort of pivot around the data and look at different histograms and bar plots and pie charts and line graphs and whatever. So you don't, like, explicitly need to prepare every different type of graph and all that. Is that, is that basically true?

Basically yes, but with one important distinction:

Plotly itself is mainly a plotting/visualization library. In Python, you give it dataframes/arrays and make interactive figures: scatter plots, histograms, heatmaps, 3D plots, line plots, etc. The output can be an interactive HTML figure, where you can zoom, pan, hover, select points, toggle traces, save images, and sometimes manipulate axes.

But the thing you are describing more fully is probably Dash, which is Plotly’s web-app/dashboard framework.

So the split is:

Plotly
: Makes interactive plots. You usually still define the plot type yourself, e.g. histogram, scatter, line, bar, heatmap.

Plotly Express
: Higher-level wrapper around Plotly. Very convenient for quickly saying: “make a histogram of this column grouped by this condition,” etc.

Dash
: Lets you build a browser-based dashboard/server app with dropdowns, sliders, filters, buttons, tabs, and callbacks. This is where you can “pivot around” data interactively in a UI.

Dash AG Grid / DataTable
: Lets you show spreadsheet-like tables where you can sort, filter, select rows, and sometimes use the table interaction to update plots.

The part that is not quite true is: Plotly does not automatically become a Tableau/PowerBI-style tool where you upload arbitrary data and it magically gives you every plot type without programming. You usually define the dashboard structure and plot options in Python. But once you define the interface, the user can interactively choose columns, filters, grouping variables, plot types, etc.

For your use case, something like this is realistic:

# User chooses:
x = "axon_length"
color = "genotype"
plot_type = "histogram"

# App dynamically renders:
px.histogram(df, x=x, color=color)

So yes: you can build a local or server-hosted HTML dashboard where you load your processed dataframe and interactively explore distributions, relationships, grouped histograms, line plots over DIV, scatter plots of morphology features, etc. But you or your code still define the allowed interactions and plotting logic.

Okay, so I guess I'm already discovered what I would want to use this for, like all the different axon metrics that I can get, so length and number of branches and number of nodes per branch, average number of branches per unit, and all that sort of stuff. I feel like I can derive all these metrics, make a big table or several tables depending on how, like, summarized the data are, or actually just a big table with like the most individualized metrics, and I feel like you can arrive at the summarized metrics after that. And then, I guess, pass it to a dashboard. Is that something you can do for free?

I’ll separate the “can I run it for free locally?” question from “can I host/share it for free?” because those have very different answers.

Yes. For your use case, you can do most or all of this for free, especially if you are running it locally or on your own lab server.

Plotly/Dash breaks down like this:
Use case	Free?	Notes
Generate interactive Plotly figures in Python	Yes	Plotly.py is open-source/browser-based. You can save figures as standalone .html files.
Build a local dashboard with Dash	Yes	dash is installable with pip install dash and can run on your own machine/server.
Host it on your lab server / localhost / Tailscale-accessible machine	Yes	You manage the server yourself. This is probably the best match for your workflow.
Publish to Plotly Cloud	Limited free tier	Plotly’s current pricing page lists a free tier with 1 public app and 1 Dash or Plotly Studio app, with paid tiers for more/private usage.
Use Dash Enterprise	No	That is the paid corporate/enterprise product for managed deployment, auth, scaling, SSO, etc.

For your axon-reconstruction metrics, the natural architecture would be:

raw recon outputs
    ↓
per-unit / per-branch / per-node metrics tables
    ↓
Parquet/CSV/SQLite database
    ↓
Dash dashboard
    ↓
interactive plots, filters, summaries, comparisons

I would strongly favor starting with one “long-form” detailed table plus maybe a few derived summary tables. For example:

dataset_id
well_id
DIV
genotype
condition
unit_id
branch_id
node_id
axon_length_um
branch_length_um
num_nodes_branch
num_branches_unit
mean_branch_length_unit
reconstruction_quality_score
...

Then the dashboard can dynamically group by things like:

genotype
DIV
well
recording
unit
branch
condition
drug treatment

and generate plots such as:

axon length distribution by genotype
branch count per unit over DIV
mean branch length per well
node count per branch histogram
quality score vs total axon length
reconstruction yield per recording

The key point: you do not need to pre-render every plot. You can write a Dash app where dropdowns choose x, y, color, facet, groupby, plot_type, etc., and the app dynamically calls Plotly Express.

Something like:

import plotly.express as px

if plot_type == "histogram":
    fig = px.histogram(df, x=x_col, color=color_col, facet_col=facet_col)
elif plot_type == "scatter":
    fig = px.scatter(df, x=x_col, y=y_col, color=color_col)
elif plot_type == "box":
    fig = px.box(df, x=group_col, y=value_col, color=color_col)

For your pipeline, I would probably not make this depend on Plotly Cloud at all. I’d make a local/lab-server app:

pip install dash plotly pandas pyarrow
python dashboard.py

Then access it at something like:

http://localhost:8050

or over Tailscale:

http://<lab-server-tailscale-ip>:8050

That would cost nothing beyond the machine you already have. The paid stuff only becomes relevant if you want polished multi-user hosting, private cloud sharing, authentication, organization-level deployment, or enterprise controls.

Cool, and then once I have the dashboard kind of set up and I can pivot through, you know, histograms of interest, and I can collapse or expand data or whatever, can I then export, you know, the PDFs and stuff like that?

I’ll treat this as a practical workflow question: static figure export, whole-dashboard report export, and reproducible “saved views” are separate capabilities.

Yes. There are three levels of export, and they differ in how polished/automatic they are.

1. Export individual plots: easy and free.
Plotly figures can be exported as PNG, JPEG, WebP, SVG, or PDF using Kaleido, and Plotly’s docs show fig.write_image(...) for this.

fig.write_image("axon_length_by_genotype.pdf")
fig.write_image("axon_length_by_genotype.svg")
fig.write_image("axon_length_by_genotype.png", scale=3)

This is the cleanest path for publication-ish figures. For papers/slides, I would usually export SVG or PDF for vector graphics, and high-resolution PNG when vector export gets weird.

2. Export the interactive figure/view as HTML: also easy and free.
Plotly can save interactive figures as standalone .html files, preserving hover, zoom, legend toggling, etc.

fig.write_html("interactive_axon_length_dashboard_view.html")

That is useful for sharing a specific interactive plot with your PI/lab without deploying the whole dashboard.

3. Export a whole dashboard/report: possible, but you design it.
Dash can include download buttons using dcc.Download, which supports app-generated downloads such as spreadsheets, images, and text files. For a full PDF report, the professional approach is usually: current dashboard filters → recompute selected plots/tables → export figures via Kaleido → assemble a PDF with reportlab, weasyprint, fpdf, or similar.

Conceptually:

dashboard filters:
  genotype = CDKL5
  DIV = 21
  metric = axon_length_um
  plot_type = histogram

        ↓

generate:
  figure PDF/SVG/PNG
  summary table CSV
  methods/metadata JSON
  optional full report PDF

A good setup for your pipeline would have buttons like:

Download current plot as PDF
Download current plot as SVG
Download current filtered data as CSV
Download summary table as CSV
Download report bundle

For your actual use case, I’d make the dashboard track a “current view spec”: metric, grouping variable, filters, plot type, aggregation level, etc. Then every exported figure/report can include that spec as provenance. That matters because six months later you want to know whether “axon length by genotype” meant all units, only QC-passing units, per-branch rows, per-unit summaries, DIV 14 only, etc.

The main caveat: Plotly/Dash does not automatically know how to export your whole exploratory session into a beautiful paper-ready PDF unless you build that report layer. But exporting individual plots, filtered tables, and reproducible report snapshots is very doable and free.

Okay, and say, you know, every time I run a data set, a well, or any, you know, I build into my pipeline to launch the dashboard for that data set or any combination of data sets, I feel like you actually need a table, at least per well, that either get, you know, combined on the fly or the dashboard can handle multiple wells somehow. Can I build that in, or can you save the HTML file?

Yes. That is a very reasonable architecture, and I would build it exactly that way: pipeline emits metrics tables; dashboard consumes one or more metrics tables; dashboard can either run live or export static HTML snapshots.

The clean pattern would be:

one dataset / well / run
    ↓
pipeline writes metrics artifacts
    ↓
dashboard loads artifacts
    ↓
user filters, groups, plots, exports

For the data layout, I would avoid making one giant manually curated summary file as the “source of truth.” Instead, write normalized metric tables per run/well, then combine them when needed.

Example output folder:

outputs/
  dataset_A/
    well_A1/
      metrics/
        units.parquet
        branches.parquet
        nodes.parquet
        recon_summary.parquet
        qc_summary.json
    well_A2/
      metrics/
        units.parquet
        branches.parquet
        nodes.parquet
        recon_summary.parquet
        qc_summary.json

Then the dashboard can load:

df_units = load_all("*/well_*/metrics/units.parquet")
df_branches = load_all("*/well_*/metrics/branches.parquet")
df_nodes = load_all("*/well_*/metrics/nodes.parquet")

Each table should include enough identity/provenance columns to combine safely:

experiment_id
dataset_id
run_id
well_id
recording_id
DIV
genotype
condition
unit_id
branch_id
node_id
qc_status

Then the dashboard can let you choose aggregation level:

node-level
branch-level
unit-level
well-level
dataset-level
genotype-level
condition-level
DIV-level

So yes, you can have one dashboard handle multiple wells. It just needs each row to know what well/dataset/condition/genotype/etc. it came from.

The two dashboard modes I would distinguish are:

Live dashboard app

axon-recon dashboard --input outputs/dataset_A

or:

axon-recon dashboard --input outputs/dataset_A outputs/dataset_B

This launches a Dash server where you can pivot, filter, select wells, compare datasets, and generate plots interactively.

Saved HTML report/snapshot

axon-recon dashboard-export --input outputs/dataset_A --out report.html

This can save either:

a standalone interactive Plotly HTML figure
a static HTML report with selected plots/tables
a full dashboard-like HTML snapshot, depending on how you build it

Important distinction: a Plotly figure can be saved as a standalone HTML very easily. A full Dash app is normally a running Python web app, not just one static HTML file. But you can build an export pathway that saves the current selected plots/tables into an HTML report.

So the robust design is:

metrics tables = durable source of truth
Dash app = exploratory interface
HTML/PDF/SVG/CSV exports = saved views / reports

For your pipeline, I would add a dedicated stage like:

compute_metrics
  - write units.parquet
  - write branches.parquet
  - write nodes.parquet
  - write well_summary.parquet
  - write metric_manifest.json

launch_dashboard
  - load one or more metric manifests
  - combine tables
  - expose filters/plots

The metric_manifest.json is useful because it tells the dashboard what files exist, what schema version they use, what dataset/well they came from, and what pipeline version generated them.

Something like:

{
  "schema_version": "axon_metrics_v1",
  "pipeline_version": "0.3.0",
  "dataset_id": "dataset_A",
  "well_id": "A1",
  "tables": {
    "units": "units.parquet",
    "branches": "branches.parquet",
    "nodes": "nodes.parquet",
    "well_summary": "well_summary.parquet"
  }
}

So the answer is: yes, build that in. Have each well/run emit table artifacts, then let the dashboard load one manifest, many manifests, one dataset, or many datasets. And yes, you can save HTML outputs, but I would treat static HTML as an export/report artifact, while the full dashboard remains a local/server app for exploratory analysis.

Okay, well then, then we need to design a table of metrics, metric artifact that gets written per table, I mean per well, because I think well is sort of like the individualized unit of pipeline run anyway. But inside of each well, we're going to have some number of units, each number of units, we can have some number of branches, each branch is going to trace some number of nodes, and each node has electrode IDs that we can tie to some units, which are neurons basically, will traverse the same nodes. At each node, there is going to be one waveform template, but it could also be a number of waveforms across the units that traverse it. So, like, there's a wealth of data we can get here. I guess I wonder at what point... I think we can stop at looking at individual waveforms. We're not going to do that. We're going to look at the waveform templates for now. That's the smallest, per electrode ID, per unit, because different waveform templates will pertain to different units. But yeah, how do we make, like what would be the shape, you think, of the table for one well?

I’m going to frame this as a schema design problem: separate “source-of-truth entity tables” from “analysis-ready metric views,” because forcing everything into one table will create duplication and ambiguous aggregation.

For one well, I would not make one gigantic table where every row tries to represent unit × branch × node × electrode × waveform. That will explode row count and duplicate metadata everywhere. I would write a well-level metrics artifact folder with several linked tables. The natural shape is a small relational schema:

well_metrics/
  manifest.json
  units.parquet
  branches.parquet
  branch_nodes.parquet
  unit_node_templates.parquet
  node_summary.parquet
  well_summary.parquet

The core hierarchy is:

well
  └── units
        └── branches
              └── nodes along branch
                    └── unit-specific waveform/template at electrode/node

But there is one complication: nodes/electrodes can be shared across units. So a node/electrode is not strictly “owned” by one branch or one unit. That means you want both:

branch_nodes table:
  "this branch traverses this node/electrode at this path index"

unit_node_templates table:
  "this unit has this waveform/template measurement at this node/electrode"

That separation is important.
1. units.parquet

One row per reconstructed/sorted unit in the well.

unit_id
well_id
dataset_id
recording_id
DIV
genotype
condition
sorter
unit_qc_label
unit_qc_score
num_spikes
firing_rate_hz
template_peak_channel_id
template_peak_electrode_id
soma_x_um
soma_y_um
num_branches
num_terminal_branches
num_nodes_total
num_unique_electrodes
total_axon_length_um
max_branch_length_um
mean_branch_length_um
median_branch_length_um
total_path_length_um
max_path_distance_um
mean_conduction_velocity_m_per_s
median_conduction_velocity_m_per_s
reconstruction_quality_score

This is the main unit-level analysis table. Most biological plots probably start here:

axon length per unit
branch count per unit
reconstruction yield per well
mean velocity per unit
genotype comparisons
DIV trends

2. branches.parquet

One row per branch for each unit.

unit_id
branch_id
well_id
dataset_id
recording_id
DIV
branch_order
parent_branch_id
is_terminal
num_nodes
num_unique_electrodes
branch_length_um
path_length_um
euclidean_length_um
tortuosity
start_node_id
end_node_id
start_electrode_id
end_electrode_id
start_x_um
start_y_um
end_x_um
end_y_um
mean_velocity_m_per_s
median_velocity_m_per_s
min_velocity_m_per_s
max_velocity_m_per_s
mean_template_amplitude_uv
max_template_amplitude_uv
mean_latency_ms
latency_range_ms
branch_qc_score

This gives you branch-level distributions:

branch length histograms
branch order distributions
terminal vs non-terminal branches
branch tortuosity
branch velocity
node count per branch

3. branch_nodes.parquet

One row per node along each reconstructed branch path.

This is your morphology path table.

unit_id
branch_id
node_id
well_id
path_index
electrode_id
channel_id
x_um
y_um
distance_from_branch_start_um
distance_from_soma_um
cumulative_path_length_um
parent_node_id
next_node_id
is_branchpoint
is_terminal_node
node_degree
latency_ms
relative_latency_ms
local_velocity_m_per_s
template_amplitude_uv
template_peak_to_peak_uv
template_snr
template_polarity

This table is where you reconstruct geometry. It is also useful for plotting the branch as a path.

Important: node_id should probably be a well-local physical/geometric node ID, while the combination of unit_id + branch_id + path_index tells you how that unit’s branch traverses it.

For example, the same physical electrode/node may appear in multiple units:

node_id = E12345
unit_id = unit_001, branch_id = b03, path_index = 12
unit_id = unit_017, branch_id = b01, path_index = 8
unit_id = unit_022, branch_id = b04, path_index = 19

That is okay. It is not duplication error; it represents shared spatial occupancy.
4. unit_node_templates.parquet

One row per unit × node/electrode template measurement.

This is the smallest level I would keep for your dashboard without storing full waveforms in the main metrics table.

unit_id
node_id
well_id
electrode_id
channel_id
x_um
y_um
template_id
template_peak_time_ms
latency_ms
relative_latency_ms
amplitude_uv
peak_to_peak_uv
snr
noise_std_uv
trough_uv
peak_uv
template_width_ms
template_energy
template_similarity_to_unit_peak
is_on_reconstructed_branch
nearest_branch_id
nearest_branch_distance_um

This table says:

For this unit, at this electrode/node, what does the template look like numerically?

I would not put the full waveform vector directly in this table unless necessary. Instead, store summary scalar metrics here and optionally keep waveform arrays separately.

For full templates, use something like:

templates.zarr
templates.npy
templates.h5

and point to them with:

template_id
template_array_path
template_sample_start
template_sample_end

Reason: waveform arrays are not naturally tabular. They are tensor data:

unit × channel/electrode × time

or maybe:

unit × node × time

That belongs in Zarr/HDF5/NumPy, not in a Parquet table full of repeated arrays.
5. node_summary.parquet

One row per physical node/electrode in the well, aggregated across units.

node_id
well_id
electrode_id
channel_id
x_um
y_um
num_units_traversing
num_branches_traversing
num_templates_detected
mean_amplitude_uv
median_amplitude_uv
max_amplitude_uv
mean_snr
median_snr
mean_latency_ms
latency_std_ms
num_genotypes_or_conditions_if_combined
is_hotspot

This table is useful for spatial dashboard views:

which electrodes are used by many reconstructed units?
where are propagation hotspots?
are some regions of the chip overrepresented?
how many units traverse each node?

6. well_summary.parquet

One row per well.

well_id
dataset_id
recording_id
DIV
genotype
condition
num_units_sorted
num_units_reconstructed
num_units_qc_pass
num_branches_total
num_nodes_total
num_unique_nodes
num_unique_electrodes
mean_axon_length_um
median_axon_length_um
mean_branches_per_unit
median_branches_per_unit
mean_branch_length_um
median_branch_length_um
mean_conduction_velocity_m_per_s
median_conduction_velocity_m_per_s
reconstruction_success_rate
mean_reconstruction_quality_score

This is the high-level dashboard entry point.
The most important design principle

Do not make the “smallest row” the only table. You can derive summaries from low-level tables, but dashboards become slow and error-prone if every plot starts from unit × branch × node × template.

Instead, write both:

atomic tables:
  branch_nodes
  unit_node_templates

derived tables:
  units
  branches
  node_summary
  well_summary

The atomic tables are your provenance. The derived tables are your dashboard workhorses.
Recommended artifact shape per well

I would make the well artifact look like this:

well_A1_metrics/
  manifest.json

  tables/
    units.parquet
    branches.parquet
    branch_nodes.parquet
    unit_node_templates.parquet
    node_summary.parquet
    well_summary.parquet

  arrays/
    templates.zarr
    branch_geometry.zarr        # optional
    template_waveform_index.parquet

  reports/
    qc_summary.json
    schema_report.json

The manifest.json should record:

{
  "artifact_type": "axon_recon_well_metrics",
  "schema_version": "axon_metrics_v1",
  "dataset_id": "dataset_001",
  "well_id": "A1",
  "recording_id": "recording_001",
  "DIV": 21,
  "tables": {
    "units": "tables/units.parquet",
    "branches": "tables/branches.parquet",
    "branch_nodes": "tables/branch_nodes.parquet",
    "unit_node_templates": "tables/unit_node_templates.parquet",
    "node_summary": "tables/node_summary.parquet",
    "well_summary": "tables/well_summary.parquet"
  },
  "arrays": {
    "templates": "arrays/templates.zarr"
  }
}

Minimal viable first version

For the first dashboard version, I would start with only these four:

units.parquet
branches.parquet
branch_nodes.parquet
well_summary.parquet

Then add:

unit_node_templates.parquet
node_summary.parquet
templates.zarr

once you are ready to inspect spatial template behavior.

The practical schema boundary should be:

Parquet tables:
  scalar metrics, IDs, coordinates, labels, QC values, lengths, counts, velocities

Zarr/HDF5/NPY arrays:
  full waveform templates, dense template tensors, large geometry matrices

So, for one well, the “shape” should be not one table, but a well-scoped metrics artifact with linked tables. The dashboard can then load one well, multiple wells, one dataset, or many datasets by concatenating tables with consistent schema and preserving well_id, dataset_id, recording_id, DIV, genotype, and condition.
