import os

# change this to either WINDOWS or LINUX
OS = 'LINUX'

# Definitions
PI = 3.141592653589793
TWO_PI = 2 * PI

# Synth architecture. OSC_ONLY or SYNTH_BASIC
SYNTH_TYPE = 'OSC_ONLY'

# Dataset configs
DATASET_SIZE = 10000
DATASET_TYPE = 'TRAIN'  # TRAIN or TEST
DATASET_MODE = 'WAV'  # WAV or MEL_SPEC
path_parent = os.path.dirname(os.getcwd())
if OS == 'WINDOWS':
    TRAIN_PARAMETERS_FILE = path_parent + "\\ai_synth\\dataset\\train\\dataset.csv"
    TRAIN_AUDIO_DIR = path_parent + "\\ai_synth\\dataset\\train\\wav_files"
    TEST_PARAMETERS_FILE = path_parent + "\\ai_synth\\dataset\\test\\dataset.csv"
    TEST_AUDIO_DIR = path_parent + "\\ai_synth\\dataset\\test\\wav_files"
elif OS == 'LINUX':
    TRAIN_PARAMETERS_FILE = path_parent + "/ai_synth/dataset/train/dataset.csv"
    TRAIN_AUDIO_DIR = path_parent + "/ai_synth/dataset/train/wav_files"
    TEST_PARAMETERS_FILE = path_parent + "/ai_synth/dataset/test/dataset.csv"
    TEST_AUDIO_DIR = path_parent + "/ai_synth/dataset/test/wav_files"

# Model configs
BATCH_SIZE = 256
EPOCHS = 10
LEARNING_RATE = 0.001
LOSS_TYPE = 'MSE' # MSE or LSD (log spectral distance)
if OS == 'WINDOWS':
    SAVE_MODEL_PATH = "..\\trained_models\\trained_synth_net.pth"
    LOAD_MODEL_PATH = "..\\trained_models\\synth_net_epoch2.pth"
elif OS == 'LINUX':
    SAVE_MODEL_PATH = "../trained_models/trained_synth_net.pth"
    LOAD_MODEL_PATH = "../trained_models/synth_net_epoch2.pth"

LOSS_MODE = 'SPECTROGRAM_ONLY'  # SPECTROGRAM_ONLY or FULL (Spectrogram + parameters)
# LOSS_MODE = 'FULL'  # SPECTROGRAM_ONLY or FULL (Spectrogram + parameters)
REGRESSION_LOSS_FACTOR = 1e-1
SPECTROGRAM_LOSS_FACTOR = 1e-5

# Synth configs
SAMPLE_RATE = 44100
SIGNAL_DURATION_SEC = 1.0

# Debug
DEBUG_MODE = False
PLOT_SPEC = False
PRINT_TRAIN_STATS = True

