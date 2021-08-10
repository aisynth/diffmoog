PI = 3.141592653589793
TWO_PI = 2 * PI
SAMPLE_RATE = 44100
SIGNAL_DURATION_SEC = 1.0
DATASET_SIZE = 5
PARAMETERS_FILE = "dataset/dataset.csv"
AUDIO_DIR = "dataset/wav_files"
NUM_OF_SYNTH_PARAMS = 21

BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001
WAVE_TYPE_DIC = {"sine": 0, "square": 1, "triangle": 2, "sawtooth": 3}
FILTER_TYPE_DIC = {"low_pass": 0, "high_pass": 1, "band_pass": 2}

SEMITONES_MAX_OFFSET = 24

# build a list of possible frequencies
middle_c_freq = 261.6255653005985
semitones_list = [*range(-SEMITONES_MAX_OFFSET, SEMITONES_MAX_OFFSET + 1)]
OSC_FREQ_LIST = [middle_c_freq * (2 ** (1 / 12)) ** x for x in semitones_list]
OSC_FREQ_DIC = {round(key, 4): value for value, key in enumerate(OSC_FREQ_LIST)}

# PARAMETERS_DIC = {'file_name',
#                    'osc1_amp', 'osc1_freq', 'osc1_wave', 'osc1_mod_index',
#                    'lfo1_freq', 'lfo1_phase', 'lfo1_wave',
#                    'osc2_amp', 'osc2_freq', 'osc2_wave', 'osc2_mod_index',
#                    'lfo2_freq', 'lfo2_phase', 'lfo2_wave',
#                    'filter_type', 'filter_freq',
#                    'attack_t', 'decay_t', 'sustain_t', 'release_t', 'sustain_level'}
# QUANTIZED_DIC = {"osc1_freq": []}
