import os, sys

from synth.synth_constants import SynthConstants

sys.path.append("..")

import pandas as pd
import scipy.io.wavfile
import torch
from model import helper
from config import Config
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from synth.synth_architecture import SynthModular
import numpy as np


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


def create_dataset(output_dir: str, split: str, size: int, device: torch.device,
                   batch_size: int = 1000, seed: int = 123):

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
    synth_obj = SynthModular(synth_cfg, device=device)

    train = (split.lower() == 'train')
    dataset_parameters = []

    # Create data
    num_batches = size // batch_size
    for batch_idx in range(num_batches):

        # Generate batch
        if synth_cfg.preset == 'MODULAR':
            synth_obj.generate_activations_and_chains(num_sounds_=batch_size, train=train)

        synth_obj.generate_random_params(num_sounds_=batch_size)
        synth_obj.generate_signal(batch_size=batch_size)
        audio = synth_obj.signal

        # Save samples
        for j in range(batch_size):
            sample_idx = (batch_size * batch_idx) + j

            params_dict = synth_obj.get_parameters(index=j)
            # params_dict = {}
            # for layer in range(synth_cfg.num_layers):
            #     for channel in range(synth_cfg.num_channels):
            #         cell = synth_obj.synth_matrix[channel][layer]
            #         if cell.operation is not None:
            #             operation = cell.operation
            #         else:
            #             operation = 'None'
            #         if cell.parameters is not None:
            #             if isinstance(list(cell.parameters.values())[0], float):
            #                 parameters = {k: v for k, v in cell.parameters.items()}
            #             else:
            #                 parameters = {k: v[j] for k, v in cell.parameters.items()}
            #         else:
            #             parameters = 'None'
            #         params_dict[cell.index] = {'operation': operation, 'parameters': parameters}
            dataset_parameters.append(params_dict)

            file_name = f"sound_{sample_idx}"
            audio_path = os.path.join(wav_files_dir, f"{file_name}.wav")
            if audio.dim() > 1:
                c_audio = audio[j]
            else:
                c_audio = audio
            c_audio = torch.squeeze(c_audio)
            c_audio = c_audio.detach().cpu().numpy()

            if c_audio.dtype == 'float64':
                c_audio = np.float32(c_audio)

            scipy.io.wavfile.write(audio_path, synth_cfg.sample_rate, c_audio)
            print(f"Generated {file_name}")

    parameters_dataframe = pd.DataFrame(dataset_parameters)
    parameters_dataframe.to_pickle(str(parameters_pickle_path))
    parameters_dataframe.to_csv(parameters_csv_path)


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter, description='Train AI Synth')
    parser.add_argument('-g', '--gpu_index', help='index of gpu (if exist, torch indexing) -1 for cpu',
                        type=int, default=-1)
    parser.add_argument('-s', '--split', required=True)
    parser.add_argument('-k', '--size', required=True, type=int)
    parser.add_argument('-n', '--name', required=True, help='name of dataset')
    args = parser.parse_args()

    cfg = Config()
    synth_cfg = SynthConstants()
    dataset_cfg = DatasetConfig(args.name)

    device = helper.get_device(args.gpu_index)
    create_dataset(split=args.split, size=args.size, dataset_cfg=dataset_cfg, synth_cfg=synth_cfg, cfg=cfg,
                   device=device)


