import pandas as pd
import os
import scipy.io.wavfile
import torch
import helper
from synth_architecture import SynthBasicFlow, SynthOscOnly, SynthModular, SynthModularCell, BASIC_FLOW
from config import DATASET_SIZE, DATASET_TYPE, DATASET_MODE, OS, SYNTH_TYPE, ONLY_OSC_DATASET
from synth_config import OSC_FREQ_LIST, NUM_LAYERS, NUM_CHANNELS

"""
Create a dataset by randomizing synthesizer parameters and generating sound.

DATASET_MODE may be WAV or MEL_SPEC:
    WAV mode creates audio files
    MEL_SPEC mode creates tensors with the audio files converted using mel spectrogram and amplitude to dB 
    transformations 
    both creates also a csv with the synth parameters.
    
DATASET_TYPE may be TRAIN or TEST. Both create the same dataset, but for different purpose and are saved in different 
locations

Configurations settings are inside config file.
"""
if __name__ == "__main__":
    dataset = []
    dataset_type = DATASET_TYPE
    print(f"Creating dataset \n mode = {DATASET_MODE}, size = {DATASET_SIZE} \n")

    path_parent = os.path.dirname(os.getcwd())
    if dataset_type == 'TRAIN':
        if OS == 'WINDOWS':
            dataset_dir_path = path_parent + f"\\dataset\\train\\"
        elif OS == 'LINUX':
            dataset_dir_path = path_parent + f"/ai_synth/dataset/train/"
    if dataset_type == 'TEST':
        if OS == 'WINDOWS':
            dataset_dir_path = path_parent + f"\\ai_synth\\dataset\\test\\"
        elif OS == 'LINUX':
            dataset_dir_path = path_parent + f"/ai_synth/dataset/test/"

    for i in range(DATASET_SIZE):
        file_name = f"sound_{i}"
        if SYNTH_TYPE == 'SYNTH_BASIC':
            synth_obj = SynthBasicFlow(file_name)
        elif SYNTH_TYPE == 'OSC_ONLY':
            if ONLY_OSC_DATASET:
                synth_obj = SynthOscOnly(file_name, parameters_dict={'osc1_freq': OSC_FREQ_LIST[i]})
            else:
                synth_obj = SynthOscOnly(file_name, parameters_dict=None)
        elif SYNTH_TYPE == 'MODULAR':
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
            synth_obj = SynthModular()
            synth_obj.apply_architecture(BASIC_FLOW)
            synth_obj.generate_random_parmas(num_sounds=1)
            # synth_obj.update_cells(update_params)
            synth_obj.generate_signal()
        else:
            raise ValueError("Provided SYNTH_TYPE is not recognized")

        audio = synth_obj.signal
        if SYNTH_TYPE == 'SYNTH_BASIC' or SYNTH_TYPE == 'OSC_ONLY':
            parameters = synth_obj.params_dict
            dataset.append(parameters)
        elif SYNTH_TYPE == 'MODULAR':
            params_dict = {}
            for layer in range(NUM_LAYERS):
                for channel in range(NUM_CHANNELS):
                    cell = synth_obj.architecture[channel][layer]
                    if cell.operation is not None:
                        params_dict[cell.index] = [cell.operation, cell.parameters]
            dataset.append(params_dict)

            # dataset = pd.concat([pd.DataFrame(l) for l in dataset], axis=1).T

        if DATASET_MODE == 'WAV':
            if OS == 'WINDOWS':
                audio_path = dataset_dir_path + "wav_files\\" + f"{file_name}.wav"
            elif OS == 'LINUX':
                audio_path = dataset_dir_path + "wav_files/" + f"{file_name}.wav"

            audio = audio.detach().cpu().numpy()
            if SYNTH_TYPE == 'OSC_ONLY':
                audio = audio.T

            scipy.io.wavfile.write(audio_path, 44100, audio)
            print(f"Generated {file_name}")

        elif DATASET_MODE == 'MEL_SPEC':
            audio_mel_spec = helper.mel_spectrogram_transform(audio)
            audio_log_mel_spec = helper.amplitude_to_db_transform(audio_mel_spec)
            audio_log_mel_spec = torch.unsqueeze(audio_log_mel_spec, dim=0)
            if OS == 'WINDOWS':
                audio_mel_spec_path = dataset_dir_path + f"audio_mel_spec_files\\{file_name}.pt"
            elif OS == 'LINUX':
                audio_mel_spec_path = dataset_dir_path + f"audio_mel_spec_files/{file_name}.pt"
            torch.save(audio_log_mel_spec, audio_mel_spec_path)
            print(f"Generated {file_name}")

    dataframe = pd.DataFrame(dataset)
    parameters_path = dataset_dir_path + "dataset.csv"

    dataframe.to_csv(parameters_path)

