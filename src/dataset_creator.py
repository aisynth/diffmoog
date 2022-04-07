import os

import pandas as pd
import scipy.io.wavfile
import torch
from pathlib import Path
import helper
from synth.synth_architecture import SynthModular
from src.synth import synth_modular_presets
from config import SynthConfig, DatasetConfig, Config
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from synth.synth_architecture import SynthModularCell, SynthModular

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


def create_dataset(train: bool, dataset_cfg: DatasetConfig, synth_cfg: SynthConfig, cfg: Config, device: torch.device):
    dataset_parameters = []
    print(f"Creating dataset \n Size = {dataset_cfg.dataset_size}")
    print(" Type = Train \n") if train else print(" Type = Test \n")

    # init paths
    if train:
        dataset_dir_path = dataset_cfg.train_dataset_dir_path
    else:
        dataset_dir_path = dataset_cfg.test_dataset_dir_path

    wav_files_dir = os.path.join(dataset_dir_path, 'wav_files', '')
    os.makedirs(wav_files_dir, exist_ok=True)

    parameters_pickle_path = os.path.join(dataset_dir_path, "params_dataset.pkl")
    parameters_csv_path = os.path.join(dataset_dir_path, "params_dataset.csv")

    for i in range(dataset_cfg.dataset_size):
        file_name = f"sound_{i}"

        synth_obj = SynthModular(synth_cfg=synth_cfg,
                                 sample_rate=cfg.sample_rate,
                                 signal_duration_sec=cfg.signal_duration_sec,
                                 num_sounds=1,
                                 device=device,
                                 preset=synth_cfg.preset)

        synth_obj.generate_random_params(synth_cfg=synth_cfg,
                                         num_sounds=1)
        synth_obj.generate_signal()

        audio = synth_obj.signal

        params_dict = {}
        for layer in range(synth_cfg.num_layers):
            for channel in range(synth_cfg.num_channels):
                cell = synth_obj.architecture[channel][layer]
                if cell.operation is not None:
                    operation = cell.operation
                else:
                    operation = 'None'
                if cell.parameters is not None:
                    parameters = cell.parameters
                else:
                    parameters = 'None'
                params_dict[cell.index] = {'operation': operation, 'parameters': parameters}
        dataset_parameters.append(params_dict)

        audio_path = os.path.join(wav_files_dir, f"{file_name}.wav")

        audio = torch.squeeze(audio)
        audio = audio.detach().cpu().numpy()

        scipy.io.wavfile.write(audio_path, cfg.sample_rate, audio)
        print(f"Generated {file_name}")

    parameters_dataframe = pd.DataFrame(dataset_parameters)
    parameters_dataframe.to_pickle(str(parameters_pickle_path))
    parameters_dataframe.to_csv(parameters_csv_path)


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter, description='Train AI Synth')
    parser.add_argument('-g', '--gpu_index', help='index of gpu (if exist, torch indexing) -1 for cpu',
                        type=int, default=0)
    parser.add_argument('-t', '--train', action='store_true', default=False)
    args = parser.parse_args()

    cfg = Config('basic_test')
    synth_cfg = SynthConfig()
    dataset_cfg = DatasetConfig('basic_toy_dataset')

    device = helper.get_device(args.gpu_index)
    create_dataset(train=args.train, dataset_cfg=dataset_cfg, synth_cfg=synth_cfg, cfg=cfg, device=device)


