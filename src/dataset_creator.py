import pandas as pd
import os
import scipy.io.wavfile
import torch
import helper
from synth_architecture import SynthBasicFlow, SynthOscOnly
from config import DATASET_SIZE, DATASET_TYPE, DATASET_MODE, OS, SYNTH_TYPE, ONLY_OSC_DATASET
from synth_config import OSC_FREQ_LIST

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
        else:
            raise ValueError("Provided SYNTH_TYPE is not recognized")

        audio = synth_obj.signal
        parameters = synth_obj.params_dict

        dataset.append(parameters)

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

