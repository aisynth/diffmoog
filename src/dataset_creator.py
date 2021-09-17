import pandas as pd
import os
import scipy.io.wavfile
import torch

import helper
from sound_generator import SynthBasicFlow
from config import DATASET_SIZE, DATASET_MODE

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
        file_name = f"sound_{i}"
        synth_obj = SynthBasicFlow(file_name)

        audio = synth_obj.signal
        parameters = synth_obj.params_dict

        dataset.append(parameters)

        cwd = os.getcwd()
        if DATASET_MODE == 'WAV':
            audio_path = cwd + f"/dataset/wav_files/{file_name}.wav"
            audio = audio.detach().cpu().numpy()
            scipy.io.wavfile.write(audio_path, 44100, audio)
            print(f"Generated {file_name}")

        elif DATASET_MODE == 'MEL_SPEC':
            audio_mel_spec = helper.mel_spectrogram_transform(audio)
            audio_log_mel_spec = helper.amplitude_to_db_transform(audio_mel_spec)
            audio_log_mel_spec = torch.unsqueeze(audio_log_mel_spec, dim=0)
            audio_mel_spec_path = cwd + f"/dataset/audio_mel_spec_files/{file_name}.pt"
            torch.save(audio_log_mel_spec, audio_mel_spec_path)
            print(f"Generated {file_name}")

    dataframe = pd.DataFrame(dataset)

    parameters_path = cwd + "/dataset/dataset.csv"
    dataframe.to_csv(parameters_path)
