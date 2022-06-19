from config import SynthConfig, Config

synth_cfg = SynthConfig()


BASIC_FLOW_LFO_DECOUPLING = [
    {'index': (0, 0), 'operation': 'lfo_sine', 'default_connection': True, 'synth_config': synth_cfg},
    {'index': (0, 1), 'operation': 'fm', 'default_connection': True, 'synth_config': synth_cfg},
    {'index': (1, 0), 'operation': 'lfo_non_sine', 'default_connection': True, 'synth_config': synth_cfg},
    {'index': (1, 1), 'operation': 'fm', 'input_list': [[1, 0]], 'output': [0, 2], 'synth_config': synth_cfg},
    {'index': (1, 2), 'operation': None, 'input_list': None, 'synth_config': synth_cfg},
    {'index': (2, 2), 'operation': None, 'input_list': None, 'synth_config': synth_cfg},
    {'index': (0, 2), 'operation': 'mix', 'input_list': [[0, 1], [1, 1]], 'synth_config': synth_cfg},
    {'index': (0, 3), 'operation': 'filter', 'default_connection': True, 'synth_config': synth_cfg},
    {'index': (0, 4), 'operation': 'env_adsr', 'default_connection': True, 'synth_config': synth_cfg}
]

# BASIC_FLOW = [
#     {'index': (0, 0), 'operation': 'lfo', 'default_connection': True, 'synth_config': synth_cfg},
#     {'index': (0, 1), 'operation': 'fm_sine', 'default_connection': True, 'synth_config': synth_cfg},
#     {'index': (1, 0), 'operation': 'lfo', 'default_connection': True, 'synth_config': synth_cfg},
#     {'index': (1, 1), 'operation': 'fm_square', 'input_list': [[1, 0]], 'output': [0, 2], 'synth_config': synth_cfg},
#     {'index': (2, 0), 'operation': 'lfo', 'default_connection': True, 'synth_config': synth_cfg},
#     {'index': (2, 1), 'operation': 'fm_saw', 'input_list': [[2, 0]], 'output': [0, 2], 'synth_config': synth_cfg},
#     {'index': (1, 2), 'operation': None, 'input_list': None, 'synth_config': synth_cfg},
#     {'index': (0, 2), 'operation': 'mix', 'input_list': [[0, 1], [1, 1], [2, 1]], 'synth_config': synth_cfg},
#     {'index': (0, 3), 'operation': 'filter', 'default_connection': True, 'synth_config': synth_cfg},
#     {'index': (0, 4), 'operation': 'amplitude_shape', 'default_connection': True, 'synth_config': synth_cfg}
# ]

BASIC_FLOW = [
    {'index': (0, 0), 'operation': 'lfo', 'default_connection': True, 'synth_config': synth_cfg},
    {'index': (0, 1), 'operation': 'fm_sine', 'default_connection': True, 'synth_config': synth_cfg},
    {'index': (1, 0), 'operation': 'lfo', 'default_connection': True, 'synth_config': synth_cfg},
    {'index': (1, 1), 'operation': 'fm_square', 'input_list': [[1, 0]], 'output': [0, 2], 'synth_config': synth_cfg},
    {'index': (2, 0), 'operation': 'lfo', 'default_connection': True, 'synth_config': synth_cfg},
    {'index': (2, 1), 'operation': 'fm_saw', 'input_list': [[2, 0]], 'output': [0, 2], 'synth_config': synth_cfg},
    {'index': (1, 2), 'operation': None, 'input_list': None, 'synth_config': synth_cfg},
    {'index': (2, 2), 'operation': None, 'input_list': None, 'synth_config': synth_cfg},
    {'index': (0, 2), 'operation': 'mix', 'input_list': [[0, 1], [1, 1], [2, 1]], 'synth_config': synth_cfg},
    {'index': (0, 3), 'operation': 'amplitude_shape', 'default_connection': True, 'synth_config': synth_cfg},
    {'index': (0, 4), 'operation': 'lowpass_filter', 'default_connection': True, 'synth_config': synth_cfg}
]


BASIC_FLOW_NO_ADSR = [
    {'index': (0, 0), 'operation': 'lfo_sine', 'default_connection': True, 'synth_config': synth_cfg},
    {'index': (0, 1), 'operation': 'fm', 'default_connection': True, 'synth_config': synth_cfg},
    {'index': (1, 0), 'operation': 'lfo_non_sine', 'default_connection': True, 'synth_config': synth_cfg},
    {'index': (1, 1), 'operation': 'fm', 'input_list': [[1, 0]], 'output': [0, 2], 'synth_config': synth_cfg},
    {'index': (1, 2), 'operation': None, 'input_list': None, 'synth_config': synth_cfg},
    {'index': (0, 2), 'operation': 'mix', 'input_list': [[0, 1], [1, 1]], 'synth_config': synth_cfg},
    {'index': (0, 3), 'operation': 'filter', 'default_connection': True, 'synth_config': synth_cfg},
]

BASIC_FLOW_NO_FILTER = [
    {'index': (0, 0), 'operation': 'lfo_sine', 'default_connection': True, 'synth_config': synth_cfg},
    {'index': (0, 1), 'operation': 'fm', 'default_connection': True, 'synth_config': synth_cfg},
    {'index': (1, 0), 'operation': 'lfo_non_sine', 'default_connection': True, 'synth_config': synth_cfg},
    {'index': (1, 1), 'operation': 'fm', 'input_list': [[1, 0]], 'output': [0, 2], 'synth_config': synth_cfg},
    {'index': (1, 2), 'operation': None, 'input_list': None, 'synth_config': synth_cfg},
    {'index': (0, 2), 'operation': 'mix', 'input_list': [[0, 1], [1, 1]], 'synth_config': synth_cfg},
    {'index': (0, 3), 'operation': 'amplitude_shape', 'default_connection': True, 'synth_config': synth_cfg},
]

BASIC_FLOW_NO_ADSR_NO_FILTER = [
    {'index': (0, 0), 'operation': 'lfo_sine', 'default_connection': True, 'synth_config': synth_cfg},
    {'index': (0, 1), 'operation': 'fm', 'default_connection': True, 'synth_config': synth_cfg},
    {'index': (1, 0), 'operation': 'lfo_non_sine', 'default_connection': True, 'synth_config': synth_cfg},
    {'index': (1, 1), 'operation': 'fm', 'input_list': [[1, 0]], 'output': [0, 2], 'synth_config': synth_cfg},
    {'index': (1, 2), 'operation': None, 'input_list': None, 'synth_config': synth_cfg},
    {'index': (0, 2), 'operation': 'mix', 'input_list': [[0, 1], [1, 1]], 'synth_config': synth_cfg},
]

SINE_LFO = [
    {'index': (0, 0), 'operation': 'lfo_sine', 'default_connection': True, 'synth_config': synth_cfg},
]

NON_SINE_LFO = [
    {'index': (1, 0), 'operation': 'lfo_non_sine', 'default_connection': True, 'synth_config': synth_cfg},
]

DOUBLE_FM_ONLY = [
    {'index': (0, 1), 'operation': 'fm', 'default_connection': True, 'synth_config': synth_cfg},
    {'index': (1, 1), 'operation': 'fm', 'input_list': [], 'output': [0, 2], 'synth_config': synth_cfg},
    {'index': (1, 2), 'operation': None, 'input_list': None, 'synth_config': synth_cfg},
    {'index': (0, 2), 'operation': 'mix', 'input_list': [[0, 1], [1, 1]], 'synth_config': synth_cfg},
]

LFO = [
    {'index': (0, 0), 'operation': 'lfo', 'default_connection': True, 'synth_config': synth_cfg}
]

OSC = [
    {'index': (0, 0), 'operation': 'osc', 'default_connection': True, 'synth_config': synth_cfg}
]

FM = [
    {'index': (0, 0), 'operation': 'lfo', 'default_connection': True, 'synth_config': synth_cfg},
    {'index': (0, 1), 'operation': 'fm', 'default_connection': True, 'synth_config': synth_cfg}
]

FM_ONLY = [
    {'index': (0, 1), 'operation': 'fm', 'default_connection': True, 'synth_config': synth_cfg}
]

FM_FILTER = [
    {'index': (0, 0), 'operation': 'lfo', 'default_connection': True, 'synth_config': synth_cfg},
    {'index': (0, 1), 'operation': 'fm', 'default_connection': True, 'synth_config': synth_cfg},
    {'index': (0, 2), 'operation': 'filter', 'default_connection': True, 'synth_config': synth_cfg}
]

FILTER_ONLY = [
    {'index': (0, 2), 'operation': 'filter', 'default_connection': True, 'synth_config': synth_cfg}
]

# LFO_FILTER = [
#     {'index': (0, 0), 'operation': 'osc', 'default_connection': True, 'synth_config': synth_cfg},
#     {'index': (0, 1), 'operation': 'filter', 'default_connection': True, 'synth_config': synth_cfg}
# ]

FM_FILTER_ADSR = [
    {'index': (0, 0), 'operation': 'lfo', 'default_connection': True, 'synth_config': synth_cfg},
    {'index': (0, 1), 'operation': 'fm', 'default_connection': True, 'synth_config': synth_cfg},
    {'index': (0, 2), 'operation': 'filter', 'default_connection': True, 'synth_config': synth_cfg},
    {'index': (0, 3), 'operation': 'env_adsr', 'default_connection': True, 'synth_config': synth_cfg}
]

DOUBLE_LFO = [
    {'index': (0, 0), 'operation': 'lfo_sine', 'default_connection': True, 'synth_config': synth_cfg},
    {'index': (1, 0), 'operation': 'lfo_non_sine', 'output': [0, 1], 'synth_config': synth_cfg},
    {'index': (0, 1), 'operation': 'mix', 'input_list': [[0, 0], [1, 0]], 'output': [0, 2], 'synth_config': synth_cfg},
    {'index': (1, 1), 'operation': None, 'input_list': None, 'synth_config': synth_cfg}
]

OSC_AMPLITUDE_SHAPER = [
    {'index': (0, 0), 'operation': 'osc', 'default_connection': True, 'synth_config': synth_cfg},
    {'index': (0, 1), 'operation': 'amplitude_shape', 'default_connection': True, 'synth_config': synth_cfg}
]

OSC_FILTER_SHAPER = [
    {'index': (0, 0), 'operation': 'osc', 'default_connection': True, 'synth_config': synth_cfg},
    {'index': (0, 1), 'operation': 'filter_shape', 'default_connection': True, 'synth_config': synth_cfg}
]

synth_presets_dict = {'BASIC_FLOW': BASIC_FLOW, 'LFO': LFO, 'OSC': OSC, 'FM': FM,
                      'BASIC_FLOW_LFO_DECOUPLING': BASIC_FLOW_LFO_DECOUPLING,
                      'BASIC_FLOW_NO_ADSR': BASIC_FLOW_NO_ADSR,
                      'BASIC_FLOW_NO_ADSR_NO_FILTER': BASIC_FLOW_NO_ADSR_NO_FILTER,
                      'FM_FILTER_ADSR': FM_FILTER_ADSR,
                      'FM_FILTER': FM_FILTER,
                      'DOUBLE_LFO': DOUBLE_LFO,
                      'FM_ONLY': FM_ONLY,
                      'FILTER_ONLY': FILTER_ONLY,
                      'OSC_AMPLITUDE_SHAPER': OSC_AMPLITUDE_SHAPER,
                      'OSC_FILTER_SHAPER': OSC_FILTER_SHAPER,
                      'BASIC_FLOW_NO_FILTER': BASIC_FLOW_NO_FILTER,
                      'DOUBLE_FM_ONLY': DOUBLE_FM_ONLY,
                      'NON_SINE_LFO': NON_SINE_LFO,
                      'SINE_LFO': SINE_LFO}
