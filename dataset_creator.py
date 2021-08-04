import pandas as pd
import os
import scipy.io.wavfile
from sound_generator import SynthBasicFlow

DATASET_SIZE = 5
dataset = []
for i in range(DATASET_SIZE):
    file_name = f"sound_{i}.wav"
    generated_audio_and_params = SynthBasicFlow(file_name)

    audio = generated_audio_and_params.audio
    parameters = generated_audio_and_params.synth_params_dict

    dataset.append(parameters)

    cwd = os.getcwd()
    audio_path = cwd + f"/dataset/wav_files/{file_name}"
    audio = audio.detach().cpu().numpy()
    scipy.io.wavfile.write(audio_path, 44100, audio)

dataframe = pd.DataFrame(dataset)

cwd = os.getcwd()
parameters_path = cwd + "/dataset/dataset.csv"
dataframe.to_csv(parameters_path)
