import os, sys

sys.path.append("..")

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import pandas as pd
import numpy as np
import sys
import subprocess

from torch import device
import scipy.io.wavfile

from synth.synth_architecture import SynthModular
from synth.parameters_sampling import ParametersSampler
from synth.synth_constants import synth_constants
from utils.gpu_utils import get_device
from utils.train_utils import get_project_root


def create_dataset(chain: str, output_dir: str, split: str, size: int, signal_duration: float, note_off_time: float,
                   device: device, batch_size: int = 1000, seed: int = 26):
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
    synth_obj = SynthModular(chain, synth_constants, device)
    params_sampler = ParametersSampler(synth_constants)

    train = (split.lower() == 'train')
    dataset_parameters = []

    # Create data
    samples_created = 0
    while samples_created < size:

        # Generate batch
        sampled_parameters = params_sampler.generate_activations_and_chains(synth_obj.synth_matrix, signal_duration,
                                                                            note_off_time, num_sounds_=batch_size)
        synth_obj.update_cells_from_dict(sampled_parameters)
        synth_obj.generate_signal(signal_duration=signal_duration, batch_size=batch_size)
        audio = synth_obj.get_final_signal()

        # Save samples
        for j in range(batch_size):
            sample_idx = samples_created

            sample_params = extract_single_sample_params(sampled_parameters, j)
            if not _verify_activity(sample_params):
                continue

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

            scipy.io.wavfile.write(audio_path, synth_constants.sample_rate, c_audio)
            print(f"Generated {file_name}")

            samples_created += 1
            if samples_created >= size:
                break

    parameters_dataframe = pd.DataFrame(dataset_parameters)
    parameters_dataframe.to_pickle(str(parameters_pickle_path))
    parameters_dataframe.to_csv(parameters_csv_path)

    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('utf-8').strip()

    args = " ".join(sys.argv[1:])
    txt_path = os.path.join(dataset_dir_path, 'commit_and_args.txt')

    with open(txt_path, 'w') as f:
        f.write(f"Git commit: {commit}\n")
        f.write(f"Arguments: {args}\n")

def _verify_activity(sample_params_dict):

    has_active_osc = False

    for key in sample_params_dict.keys():
        if sample_params_dict[key]:
            operation = sample_params_dict[key]['operation']
            if operation in ['osc', 'osc_sine', 'osc_saw', 'osc_square', 'fm_saw', 'fm_sine', 'fm_square', 'fm',
                             'fm_lfo', 'saw_square_osc', 'surrogate_lfo', 'surrogate_fm_sine', 'surrogate_fm_saw',
                             'osc_sine_no_activeness', 'osc_square_no_activeness', 'osc_saw_no_activeness',
                             'osc_sine_no_activeness_cont_freq',
                             'osc_square_no_activeness_cont_freq',
                             'osc_saw_no_activeness_cont_freq']:
                is_active = sample_params_dict[key]['parameters'].get('active', True)
                has_active_osc = has_active_osc or is_active
        else:
            continue

    # todo: commented code works only for MODULAR chain. make sure above code generalizes
    # if sample_params_dict.get((0, 2)):
    #     sine_osc_activeness = sample_params_dict[(0, 2)]['parameters']['active']
    # else:
    #     sine_osc_activeness = False
    #
    # if sample_params_dict.get((1, 2)):
    #     saw_osc_activeness = sample_params_dict[(1, 2)]['parameters']['active']
    # else:
    #     saw_osc_activeness = False
    #
    # if sample_params_dict.get((2, 2)):
    #     square_osc_activeness = sample_params_dict[(2, 2)]['parameters']['active']
    # else:
    #     square_osc_activeness = False
    #
    # has_active_osc = sine_osc_activeness or square_osc_activeness or saw_osc_activeness

    return has_active_osc


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
    parser.add_argument('-c', '--chain', required=False, help='Synth chain', default='MODULAR')

    parser.add_argument('-sd', '--signal_duration', required=False, type=float, default=4.0)
    parser.add_argument('-no', '--note_off', required=False, type=float, default=3.0)

    parser.add_argument('-bs', '--batch_size', required=False, type=int, default=1000)

    args = parser.parse_args()

    root = get_project_root()
    EXP_ROOT = root.joinpath('experiments')
    DATA_ROOT = root.joinpath('data')
    #
    # EXP_ROOT = os.path.join(root, 'experiments')
    # DATA_ROOT = os.path.join(root, 'data')

    #todo: change os to Path
    output_dir = os.path.join(DATA_ROOT, args.name, '')
    os.makedirs(output_dir, exist_ok=True)

    device = get_device(args.gpu_index)
    create_dataset(chain=args.chain, output_dir=output_dir, split=args.split, size=args.size,
                   signal_duration=args.signal_duration, note_off_time=args.note_off, batch_size=args.batch_size,
                   device=device)


