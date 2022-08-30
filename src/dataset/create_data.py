import os, sys

sys.path.append("..")

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import pandas as pd
import numpy as np

from torch import device
import scipy.io.wavfile

from config import DATA_ROOT

from synth.synth_architecture import SynthModular
from synth.parameters_sampling import ParametersSampler
from synth.synth_constants import synth_structure
from utils.gpu_utils import get_device


def create_dataset(preset: str, output_dir: str, split: str, size: int, signal_duration: float, note_off_time: float,
                   device: device, batch_size: int = 1000, seed: int = 42):
    """
    Create a dataset by randomizing synthesizer parameters and generating sound.

    cfg.dataset_mode may be WAV or MEL_SPEC:
        WAV mode creates audio files
        MEL_SPEC mode creates tensors with the audio files converted using mel spectrogram and amplitude to dB
        transformations
        both creates also a csv with the synth parameters.

    DATASET_TYPE may be TRAIN or TEST. Both create the same dataset, but for different purpose and are saved in different
    locations

    Configurations settings are inside config file.
    """

    print(f"Creating dataset \n Size = {size}")
    print(f" Type = {split} \n")

    # init paths
    dataset_dir_path = os.path.join(output_dir, split.lower(), '')

    wav_files_dir = os.path.join(dataset_dir_path, 'wav_files', '')
    parameters_pickle_path = os.path.join(dataset_dir_path, "params_dataset.pkl")
    parameters_csv_path = os.path.join(dataset_dir_path, "params_dataset.csv")

    os.makedirs(wav_files_dir, exist_ok=True)

    # Other inits
    np.random.seed(seed)
    synth_obj = SynthModular(preset, synth_structure, device)
    params_sampler = ParametersSampler(synth_structure)

    train = (split.lower() == 'train')
    dataset_parameters = []

    # Create data
    num_batches = size // batch_size
    for batch_idx in range(num_batches):

        # Generate batch
        sampled_parameters = params_sampler.generate_activations_and_chains(synth_obj.synth_matrix, signal_duration,
                                                                            note_off_time, num_sounds_=batch_size)
        synth_obj.update_cells_from_dict(sampled_parameters)
        synth_obj.generate_signal(signal_duration=signal_duration, batch_size=batch_size)
        audio = synth_obj.get_final_signal()

        # Save samples
        for j in range(batch_size):
            sample_idx = (batch_size * batch_idx) + j

            sample_params = extract_single_sample_params(sampled_parameters, j)
            dataset_parameters.append(sample_params)

            file_name = f"sound_{sample_idx}"
            audio_path = os.path.join(wav_files_dir, f"{file_name}.wav")
            if audio.dim() > 1:
                c_audio = audio[j]
            else:
                c_audio = audio
            c_audio = c_audio.squeeze()
            c_audio = c_audio.detach().cpu().numpy()

            if c_audio.dtype == 'float64':
                c_audio = np.float32(c_audio)

            scipy.io.wavfile.write(audio_path, synth_structure.sample_rate, c_audio)
            print(f"Generated {file_name}")

    parameters_dataframe = pd.DataFrame(dataset_parameters)
    parameters_dataframe.to_pickle(str(parameters_pickle_path))
    parameters_dataframe.to_csv(parameters_csv_path)


def extract_single_sample_params(params_dict, idx):
    single_params_dict = {}

    for cell_index, cell_params in params_dict.items():
        if cell_params['operation'] is not None:
            operation = cell_params['operation']
        else:
            operation = 'None'
        if cell_params['parameters'] is not None and len(cell_params['parameters']) > 0:
            if isinstance(list(cell_params['parameters'].values())[0], float):
                parameters = {k: v for k, v in cell_params['parameters'].items()}
            else:
                parameters = {k: v[idx] for k, v in cell_params['parameters'].items()}
        else:
            parameters = 'None'
        single_params_dict[cell_index] = {'operation': operation, 'parameters': parameters}

    return single_params_dict


if __name__ == '__main__':

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter, description='AI Synth dataset creation')
    parser.add_argument('-g', '--gpu_index', help='index of gpu (if exist, torch indexing) -1 for cpu',
                        type=int, default=-1)
    parser.add_argument('-s', '--split', required=True)
    parser.add_argument('-k', '--size', required=True, type=int)
    parser.add_argument('-n', '--name', required=True, help='name of dataset')
    parser.add_argument('-p', '--preset', required=False, help='Synth preset', default='MODULAR')

    parser.add_argument('-sd', '--signal_duration', required=False, type=float, default=4.0)
    parser.add_argument('-no', '--note_off', required=False, type=float, default=3.0)

    parser.add_argument('-bs', '--batch_size', required=False, type=int, default=1000)

    args = parser.parse_args()

    output_dir = os.path.join(DATA_ROOT, args.name, '')
    os.makedirs(output_dir, exist_ok=True)

    device = get_device(args.gpu_index)
    create_dataset(preset=args.preset, output_dir=output_dir, split=args.split, size=args.size,
                   signal_duration=args.signal_duration, note_off_time=args.note_off, batch_size=args.batch_size,
                   device=device)


