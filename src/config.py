# Definitions
PI = 3.141592653589793
TWO_PI = 2 * PI

# Dataset configs
DATASET_SIZE = 8
# DATASET_MODE = 'MEL_SPEC'  # WAV or MEL_SPEC
DATASET_MODE = 'WAV'  # WAV or MEL_SPEC
PARAMETERS_FILE = "dataset/dataset.csv"

# Model configs
BATCH_SIZE = 8
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
