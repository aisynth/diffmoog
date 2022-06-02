import os
import pandas as pd
import scipy.io.wavfile
import torch
from model import helper
from config import SynthConfig, DatasetConfig, Config
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from synth.synth_architecture import SynthModular

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

    synth_obj = SynthModular(synth_cfg=synth_cfg,
                             sample_rate=cfg.sample_rate,
                             signal_duration_sec=cfg.signal_duration_sec,
                             num_sounds_=dataset_cfg.dataset_size,
                             device=device,
                             preset=synth_cfg.preset)

    num_batches = dataset_cfg.dataset_size // dataset_cfg.batch_size
    for batch_idx in range(num_batches):

        synth_obj.generate_random_params(synth_cfg=synth_cfg,
                                         num_sounds_=dataset_cfg.batch_size)

        synth_obj.generate_signal(num_sounds_=dataset_cfg.batch_size)

        audio = synth_obj.signal

        for j in range(dataset_cfg.batch_size):

            params_dict = {}
            for layer in range(synth_cfg.num_layers):
                for channel in range(synth_cfg.num_channels):
                    cell = synth_obj.architecture[channel][layer]
                    if cell.operation is not None:
                        operation = cell.operation
                    else:
                        operation = 'None'
                    if cell.parameters is not None:
                        if isinstance(list(cell.parameters.values())[0], float):
                            parameters = {k: v for k, v in cell.parameters.items()}
                        else:
                            parameters = {k: v[j] for k, v in cell.parameters.items()}
                    else:
                        parameters = 'None'
                    params_dict[cell.index] = {'operation': operation, 'parameters': parameters}
            dataset_parameters.append(params_dict)

            sample_idx = (dataset_cfg.batch_size * batch_idx) + j
            file_name = f"sound_{sample_idx}"
            audio_path = os.path.join(wav_files_dir, f"{file_name}.wav")
            if audio.dim() > 1:
                c_audio = audio[j]
            else:
                c_audio = audio
            c_audio = torch.squeeze(c_audio)
            c_audio = c_audio.detach().cpu().numpy()

            scipy.io.wavfile.write(audio_path, cfg.sample_rate, c_audio)
            print(f"Generated {file_name}")

    parameters_dataframe = pd.DataFrame(dataset_parameters)
    parameters_dataframe.to_pickle(str(parameters_pickle_path))
    parameters_dataframe.to_csv(parameters_csv_path)


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter, description='Train AI Synth')
    parser.add_argument('-g', '--gpu_index', help='index of gpu (if exist, torch indexing) -1 for cpu',
                        type=int, default=-1)
    parser.add_argument('-t', '--train', action='store_true', default=False)
    parser.add_argument('-n', '--name', required=True, help='name of dataset')
    args = parser.parse_args()

    cfg = Config()
    synth_cfg = SynthConfig()
    dataset_cfg = DatasetConfig(args.name)

    device = helper.get_device(args.gpu_index)
    create_dataset(train=args.train, dataset_cfg=dataset_cfg, synth_cfg=synth_cfg, cfg=cfg, device=device)


