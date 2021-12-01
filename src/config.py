import os

# change this to either WINDOWS or LINUX
OS = 'WINDOWS'

# Definitions
PI = 3.141592653589793
TWO_PI = 2 * PI
NUM_OF_OSC_FREQUENCIES = 49

# Synth configs
SAMPLE_RATE = 44100
SIGNAL_DURATION_SEC = 1.0

# Synth architecture. OSC_ONLY or SYNTH_BASIC
SYNTH_TYPE = 'OSC_ONLY'

# Dataset configs
ONLY_OSC_DATASET = True
if ONLY_OSC_DATASET:
    DATASET_SIZE = NUM_OF_OSC_FREQUENCIES
    NUM_EPOCHS_TO_PRINT_STATS = 20

else:
    DATASET_SIZE = 50000
DATASET_TYPE = 'TRAIN'  # TRAIN or TEST
DATASET_MODE = 'WAV'  # WAV or MEL_SPEC
path_parent = os.path.dirname(os.getcwd())
if OS == 'WINDOWS':
    TRAIN_PARAMETERS_FILE = path_parent + "\\dataset\\train\\dataset.csv"
    TRAIN_AUDIO_DIR = path_parent + "\\dataset\\train\\wav_files"
    TEST_PARAMETERS_FILE = path_parent + "\\dataset\\test\\dataset.csv"
    TEST_AUDIO_DIR = path_parent + "\\dataset\\test\\wav_files"
elif OS == 'LINUX':
    TRAIN_PARAMETERS_FILE = path_parent + "/ai_synth/dataset/train/dataset.csv"
    TRAIN_AUDIO_DIR = path_parent + "/ai_synth/dataset/train/wav_files"
    TEST_PARAMETERS_FILE = path_parent + "/ai_synth/dataset/test/dataset.csv"
    TEST_AUDIO_DIR = path_parent + "/ai_synth/dataset/test/wav_files"

# Model configs
CNN_NETWORK = 'BIG'  # 'BIG' or 'SMALL' - one of 2 possible network architectures
BATCH_SIZE = 256
EPOCHS = 2000
LEARNING_RATE = 0.001

LOSS_MODE = 'SPECTROGRAM_ONLY'  # SPECTROGRAM_ONLY, PARAMETERS_ONLY or FULL (Spectrogram + parameters)
SPECTROGRAM_LOSS_TYPE = 'MSE'  # MSE or LSD (Log Spectral Distance) or KL (Kullback-Leibler)
FREQ_PARAM_LOSS_TYPE = 'MSE'  # MSE or CE (Cross Entropy)

" The model can output the oscillator frequency as:                                 "
"   1. LOGITS (size is num of frequencies, for cross entropy loss)                  "
"   2. WEIGHTED - inner product of <probabilities, original frequencies>. size is 1 "
"   3. SINGLE - linear layer outputs single neuron. size is 1                       "
MODEL_FREQUENCY_OUTPUT = 'SINGLE'
if FREQ_PARAM_LOSS_TYPE == 'CE':
    MODEL_FREQUENCY_OUTPUT = 'LOGITS'

TRANSFORM = 'MEL_SPECTROGRAM'  # MEL_SPECTROGRAM or SPECTROGRAM - to be used in the data loader and at the synth output

USE_LOADED_MODEL = False
# USE_LOADED_MODEL = False
if OS == 'WINDOWS':
    SAVE_MODEL_PATH = "..\\trained_models\\trained_synth_net.pth"
    LOAD_MODEL_PATH = path_parent + "\\trained_models\\synth_net_epoch301.pth"
elif OS == 'LINUX':
    SAVE_MODEL_PATH = "../trained_models/trained_synth_net.pth"
    LOAD_MODEL_PATH = "../trained_models/synth_net_epoch2.pth"

REGRESSION_LOSS_FACTOR = 1e-1
SPECTROGRAM_LOSS_FACTOR = 1e-5
FREQ_MSE_LOSS_FACTOR = 1e-3

# Debug
DEBUG_MODE = False
PLOT_SPEC = False
PRINT_TRAIN_STATS = True
PRINT_ACCURACY_STATS = False

LOG_SPECTROGRAM_MSE_LOSS = False

if LOG_SPECTROGRAM_MSE_LOSS:
    SPECTROGRAM_LOSS_FACTOR = 1000
