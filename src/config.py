import os

# change this to either WINDOWS or LINUX
OS = 'WINDOWS'

# Definitions
PI = 3.141592653589793
TWO_PI = 2 * PI

# Dataset configs
DATASET_SIZE = 100
path_parent = os.path.dirname(os.getcwd())
if OS == 'WINDOWS':
    PARAMETERS_FILE = path_parent + "\\dataset\\dataset.csv"
    AUDIO_DIR = path_parent + "\\dataset\\wav_files"
elif OS == 'LINUX':
    PARAMETERS_FILE = path_parent + "\\dataset/dataset.csv"
    AUDIO_DIR = path_parent + "\\dataset/wav_files"
# DATASET_MODE = 'MEL_SPEC'  # WAV or MEL_SPEC
DATASET_MODE = 'WAV'  # WAV or MEL_SPEC

# Model configs
BATCH_SIZE = 128
EPOCHS = 1000
LEARNING_RATE = 0.001

LOSS_MODE = 'SPECTROGRAM_ONLY'  # SPECTROGRAM_ONLY or FULL (Spectrogram + parameters)
# LOSS_MODE = 'FULL'  # SPECTROGRAM_ONLY or FULL (Spectrogram + parameters)
REGRESSION_LOSS_FACTOR = 1e-1
SPECTROGRAM_LOSS_FACTOR = 1e-3

# Synth configs
SAMPLE_RATE = 44100
SIGNAL_DURATION_SEC = 1.0

# Debug
DEBUG_MODE = False
PRINT_TRAIN_STATS = True

