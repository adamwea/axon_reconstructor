
import json
from pathlib import Path
from spikeinterface import load_extractor
from spikeinterface.sorters import run_sorter_local

if __name__ == '__main__':
    # this __name__ protection help in some case with multiprocessing (for instance HS2)
    # load recording in container
    json_rec = Path('/home/adamm/workspace/RBS_axonal_reconstructions/AxonReconPipeline/data/temp_data/sortings/241017/M08029/AxonTracking/000073/in_container_recording.json')
    pickle_rec = Path('/home/adamm/workspace/RBS_axonal_reconstructions/AxonReconPipeline/data/temp_data/sortings/241017/M08029/AxonTracking/000073/in_container_recording.pickle')
    if json_rec.exists():
        recording = load_extractor(json_rec)
    else:
        recording = load_extractor(pickle_rec)

    # load params in container
    with open('/home/adamm/workspace/RBS_axonal_reconstructions/AxonReconPipeline/data/temp_data/sortings/241017/M08029/AxonTracking/000073/in_container_params.json', encoding='utf8', mode='r') as f:
        sorter_params = json.load(f)

    # run in container
    output_folder = '/home/adamm/workspace/RBS_axonal_reconstructions/AxonReconPipeline/data/temp_data/sortings/241017/M08029/AxonTracking/000073/well000'
    sorting = run_sorter_local(
        'kilosort2', recording, output_folder=output_folder,
        remove_existing_folder=False, delete_output_folder=False,
        verbose=True, raise_error=True, with_output=True, **sorter_params
    )
    sorting.save(folder='/home/adamm/workspace/RBS_axonal_reconstructions/AxonReconPipeline/data/temp_data/sortings/241017/M08029/AxonTracking/000073/well000/in_container_sorting')
