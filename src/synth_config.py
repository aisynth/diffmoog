from config import SYNTH_TYPE

SAMPLE_RATE = 44100
SIGNAL_DURATION_SEC = 1.0

if SYNTH_TYPE == 'OSC_ONLY':
    CLASSIFICATION_PARAM_LIST = ['osc1_freq']
    REGRESSION_PARAM_LIST = []
    PARAM_LIST = [CLASSIFICATION_PARAM_LIST, REGRESSION_PARAM_LIST]

elif SYNTH_TYPE == 'SYNTH_BASIC':
    CLASSIFICATION_PARAM_LIST = \
        ['osc1_freq', 'osc1_wave', 'osc2_freq', 'osc2_wave', 'filter_type']
    REGRESSION_PARAM_LIST = \
        ['osc1_amp', 'osc1_mod_index', 'lfo1_freq',
         'osc2_amp', 'osc2_mod_index', 'lfo2_freq',
         'filter_freq', 'attack_t', 'decay_t', 'sustain_t', 'release_t', 'sustain_level']
    PARAM_LIST = [CLASSIFICATION_PARAM_LIST, REGRESSION_PARAM_LIST]

else:
    raise ValueError("Provided SYNTH_TYPE is not recognized")

WAVE_TYPE_DIC = {"sine": 0,
                 "square": 1,
                 "sawtooth": 2}

WAVE_TYPE_DIC_INV = {v: k for k, v in WAVE_TYPE_DIC.items()}

FILTER_TYPE_DIC = {"low_pass": 0,
                   "high_pass": 1}
FILTER_TYPE_DIC_INV = {v: k for k, v in FILTER_TYPE_DIC.items()}

# build a list of possible frequencies
SEMITONES_MAX_OFFSET = 24
MIDDLE_C_FREQ = 261.6255653005985
semitones_list = [*range(-SEMITONES_MAX_OFFSET, SEMITONES_MAX_OFFSET + 1)]
OSC_FREQ_LIST = [MIDDLE_C_FREQ * (2 ** (1 / 12)) ** x for x in semitones_list]
# OSC_FREQ_LIST = OSC_FREQ_LIST1[39:]
OSC_FREQ_DIC = {round(key, 4): value for value, key in enumerate(OSC_FREQ_LIST)}
OSC_FREQ_DIC_INV = {v: k for k, v in OSC_FREQ_DIC.items()}

MAX_AMP = 1
MAX_MOD_INDEX = 100
MAX_LFO_FREQ = 20
MIN_FILTER_FREQ = 20
MAX_FILTER_FREQ = 20000
