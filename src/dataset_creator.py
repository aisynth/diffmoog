import pandas as pd
import os
import scipy.io.wavfile
import torch

import helper
from sound_generator import SynthBasicFlow
from config import DATASET_SIZE, DATASET_MODE, OS

DATASET_TYPE = 'TEST'
DATASET_SIZE = 20 
"""
Create a dataset by randomizing synthesizer parameters and generating sound.

DATASET_MODE may be WAV or MEL_SPEC:
WAV mode creates audio files
MEL_SPEC mode creates tensors with the audio files converted using mel spectrogram and amplitude to dB transformations 
both creates also a csv with the synth parameters.

Configurations settings are inside config file.
"""
if __name__ == "__main__":
    dataset = []
    print(f"Creating dataset \n mode = {DATASET_MODE}, size = {DATASET_SIZE} \n")
    for i in range(DATASET_SIZE):
        generated_audio_and_params = SynthBasicFlow(file_name=file_name)
        synth_obj = SynthBasicFlow(file_name)

        audio = synth_obj.signal
        parameters = synth_obj.params_dict

        dataset.append(parameters)

        path_parent = os.path.dirname(os.getcwd())
        if dataset_type == 'TRAIN':
            if OS == 'WINDOWS':
                audio_path = path_parent + f"\\dataset\\train\\wav_files\\{file_name}"
            elif OS == 'LINUX':
                audio_path = path_parent + f"/dataset/train/wav_files/{file_name}"
        if dataset_type == 'TEST':
            if OS == 'WINDOWS':
                audio_path = path_parent + f"\\dataset\\test\\wav_files\\{file_name}"
            elif OS == 'LINUX':
                audio_path = path_parent + f"/dataset/test/wav_files/{file_name}"
                
    if DATASET_MODE == 'WAV':
        audio_path = cwd + f"/dataset/wav_files/{file_name}.wav"
        audio = audio.detach().cpu().numpy()
        scipy.io.wavfile.write(audio_path, 44100, audio[0][0][0])
        print(f"Generated {file_name}")
    elif DATASET_MODE == 'MEL_SPEC':
        audio_mel_spec = helper.mel_spectrogram_transform(audio)
        audio_log_mel_spec = helper.amplitude_to_db_transform(audio_mel_spec)
        audio_log_mel_spec = torch.unsqueeze(audio_log_mel_spec, dim=0)
        audio_mel_spec_path = cwd + f"/dataset/audio_mel_spec_files/{file_name}.pt"
        torch.save(audio_log_mel_spec, audio_mel_spec_path)
        print(f"Generated {file_name}")        
    
    

    dataframe = pd.DataFrame(dataset)

    path_parent = os.path.dirname(os.getcwd())
    if dataset_type == 'TRAIN':
        if OS == 'WINDOWS':
            parameters_path = path_parent + "\\dataset\\train\\dataset.csv"
        elif OS == 'LINUX':
            parameters_path = path_parent + "/dataset/train/dataset.csv"
    elif dataset_type == 'TEST':
        if OS == 'WINDOWS':
            parameters_path = path_parent + "\\dataset\\test\\dataset.csv"
        elif OS == 'LINUX':
            parameters_path = path_parent + "/dataset/test/dataset.csv"

    dataframe.to_csv(parameters_path)


if __name__ == "__main__":
    create_dataset(DATASET_TYPE, DATASET_SIZE)
