import pandas as pd
import scipy.io.wavfile
import torch
from pathlib import Path
import helper
from synth.synth_architecture import SynthModular
from src.synth import synth_modular_presets
from config import SynthConfig, DatasetConfig, Config
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


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


def create_dataset(train: bool,data_cfg: DatasetConfig, synth_cfg: SynthConfig, sample_rate):
    dataset = []
    print(f"Creating dataset \n size = {data_cfg.dataset_size} \n")

    path_parent = Path(__file__).parent.parent
    if train:
        dataset_dir_path = path_parent.joinpath('dataset', 'train')
    else:
        dataset_dir_path = path_parent.joinpath('dataset', 'test')

    for i in range(data_cfg.dataset_size):
        file_name = f"sound_{i}"

        # update_params = [
        #     SynthModularCell(index=(0, 0), parameters={'amp': 1, 'freq': 3, 'waveform': 'sine'}),
        #     SynthModularCell(index=(0, 1), parameters={'amp_c': 0.9, 'freq_c': 220, 'waveform': 'square',
        #                                                'mod_index': 10}),
        #     SynthModularCell(index=(1, 0), parameters={'amp': 1, 'freq': 1, 'waveform': 'sine'}),
        #     SynthModularCell(index=(1, 1), parameters={'amp_c': 0.7, 'freq_c': 500, 'waveform': 'sine',
        #                                                'mod_index': 10}),
        #     SynthModularCell(index=(0, 2), parameters={'factor': 0}),
        #     SynthModularCell(index=(0, 3), parameters={'filter_freq': 15000, 'filter_type': 'low_pass'}),
        #     SynthModularCell(index=(0, 4), parameters={'attack_t': 0.25, 'decay_t': 0.25, 'sustain_t': 0.25,
        #                                                'sustain_level': 0.3, 'release_t': 0.25})
        # ]
        synth_obj = SynthModular(sample_rate=sample_rate,
                                 signal_duration_sec=1.0,
                                 num_sounds=1)
        if synth_cfg.preset == 'BASIC_FLOW':
            synth_cfg.preset = synth_modular_presets.BASIC_FLOW
        elif synth_cfg.preset == 'FM':
            synth_cfg.preset = synth_modular_presets.FM
        synth_obj.apply_architecture(synth_cfg.preset)
        synth_obj.generate_random_params(synth_cfg=synth_cfg,
                                         num_sounds=1)
        # synth_obj.update_cells(update_params)
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
        dataset.append(params_dict)

        # dataset = pd.concat([pd.DataFrame(l) for l in dataset], axis=1).T

        audio_path = dataset_dir_path.joinpath("wav_files", f"{file_name}.wav")

        audio = torch.squeeze(audio)
        audio = audio.detach().cpu().numpy()

        scipy.io.wavfile.write(audio_path, sample_rate, audio)
        print(f"Generated {file_name}")

    dataframe = pd.DataFrame(dataset)
    parameters_path = dataset_dir_path.joinpath("dataset.pkl")
    dataframe.to_pickle(str(parameters_path))


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter, description='Train AI Synth')
    parser.add_argument('-g', '--gpu_index', help='index of gpu (if exist, torch indexing) -1 for cpu',
                        type=int, default=0)
    parser.add_argument('-t', '--train', action='store_true', default=True)

    args = parser.parse_args()
    synth_cfg = SynthConfig()
    dataset_cfg = DatasetConfig()
    cfg = Config()
    create_dataset(args.train, dataset_cfg, synth_cfg, sample_rate=cfg.sample_rate)

