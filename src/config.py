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

" Mode - define a common configuration for the whole system     "
"   0 -                     Use custom configurations           "
"   Any other number -      Use predefined configuration preset "
MODE = 6

# Dataset configs
ONLY_OSC_DATASET = True
if ONLY_OSC_DATASET:
    DATASET_SIZE = NUM_OF_OSC_FREQUENCIES
    NUM_EPOCHS_TO_PRINT_STATS = 1
    NUM_EPOCHS_TO_SAVE_MODEL = 100

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
EPOCHS = 50000
LEARNING_RATE = 0.00001

REINFORCEMENT_EPSILON = 0.15

# Synth architecture. OSC_ONLY or SYNTH_BASIC
SYNTH_TYPE = 'OSC_ONLY'

" The architecture of the system, that defines the data flow and the loss functions:                    "
"   1. SPECTROGRAM_ONLY (input -> CNN -> parameters -> Synth -> output; Loss over spectrograms)         "
"   2. PARAMETERS_ONLY (input -> CNN -> parameters; Loss over parameters)                               "
"   3. FULL - (input -> CNN -> parameters -> Synth -> output; Loss over spectrograms AND parameters)    "
"   4. SPEC_NO_SYNTH (input -> CNN -> parameters); Output inner product <probabilities, spectrograms>;   "
"      Loss over spectrograms)                                                                          "
"   5. REINFORCE - (input -> CNN -> parameters); Loss is computed to maximize rewards for correct       "
"       classification. Using the classical REINFORCE algorithm                                         "
ARCHITECTURE = 'SPECTROGRAM_ONLY'  # SPECTROGRAM_ONLY, PARAMETERS_ONLY, SPEC_NO_SYNTH or FULL (Spectrogram + parameters)
SPECTROGRAM_LOSS_TYPE = 'MSE'  # MSE or LSD (Log Spectral Distance) or KL (Kullback-Leibler)
FREQ_PARAM_LOSS_TYPE = 'MSE'  # MSE or CE (Cross Entropy)

" The model can output the oscillator frequency as:                                 "
"   1. LOGITS (size is num of frequencies, for cross entropy loss)                  "
"   2. PROBS (same as LOGITS, but softmax is applied)                               "
"   3. WEIGHTED - inner product of <probabilities, original frequencies>. size is 1 "
"   4. SINGLE - linear layer outputs single neuron. size is 1                       "
MODEL_FREQUENCY_OUTPUT = 'SINGLE'
if FREQ_PARAM_LOSS_TYPE == 'CE':
    MODEL_FREQUENCY_OUTPUT = 'LOGITS'

TRANSFORM = 'MEL_SPECTROGRAM'  # MEL_SPECTROGRAM or SPECTROGRAM - to be used in the data loader and at the synth output

REINFORCE_REWARD_SPEC_MSE_THRESHOLD = 6

USE_LOADED_MODEL = False
# USE_LOADED_MODEL = False
if OS == 'WINDOWS':
    SAVE_MODEL_PATH = "..\\trained_models\\trained_synth_net.pt"
    LOAD_MODEL_PATH = path_parent + "\\trained_models\\synth_net_epoch401.pt"
elif OS == 'LINUX':
    SAVE_MODEL_PATH = "../trained_models/trained_synth_net.pth"
    LOAD_MODEL_PATH = "../trained_models/synth_net_epoch2.pth"

REGRESSION_LOSS_FACTOR = 1e-1
SPECTROGRAM_LOSS_FACTOR = 1e-5
FREQ_MSE_LOSS_FACTOR = 1e-3
FREQ_REINFORCE_LOSS_FACTOR = 1e5

# Debug
DEBUG_MODE = False
PLOT_SPEC = False
PRINT_TRAIN_STATS = True
PRINT_ACCURACY_STATS = False
PRINT_PER_ACCURACY_STATS_MULTIPLE_EPOCHS = False

LOG_SPECTROGRAM_MSE_LOSS = False

if LOG_SPECTROGRAM_MSE_LOSS:
    SPECTROGRAM_LOSS_FACTOR = 1000


if MODE == 1:
    SYNTH_TYPE = 'OSC_ONLY'
    ARCHITECTURE = 'PARAMETERS_ONLY'
    MODEL_FREQUENCY_OUTPUT = 'SINGLE'
    FREQ_PARAM_LOSS_TYPE = 'MSE'
elif MODE == 2:
    SYNTH_TYPE = 'OSC_ONLY'
    ARCHITECTURE = 'PARAMETERS_ONLY'
    FREQ_PARAM_LOSS_TYPE = 'CE'
    MODEL_FREQUENCY_OUTPUT = 'LOGITS'
elif MODE == 3:
    SYNTH_TYPE = 'OSC_ONLY'
    ARCHITECTURE = 'PARAMETERS_ONLY'
    MODEL_FREQUENCY_OUTPUT = 'WEIGHTED'
    FREQ_PARAM_LOSS_TYPE = 'MSE'
elif MODE == 4:
    SYNTH_TYPE = 'OSC_ONLY'
    ARCHITECTURE = 'SPECTROGRAM_ONLY'
    SPECTROGRAM_LOSS_TYPE = 'MSE'
    MODEL_FREQUENCY_OUTPUT = 'SINGLE'
elif MODE == 5:
    SYNTH_TYPE = 'OSC_ONLY'
    ARCHITECTURE = 'SPEC_NO_SYNTH'
    SPECTROGRAM_LOSS_TYPE = 'MSE'
    MODEL_FREQUENCY_OUTPUT = 'WEIGHTED'
elif MODE == 6:
    SYNTH_TYPE = 'OSC_ONLY'
    ARCHITECTURE = 'REINFORCE'
    SPECTROGRAM_LOSS_TYPE = 'MSE'
    MODEL_FREQUENCY_OUTPUT = 'PROBS'

