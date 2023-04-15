# Basic (single channel) presets
OSC_FILTER = [
    {'index': (0, 0), 'operation': 'osc', 'default_connection': True},
    {'index': (0, 1), 'operation': 'lowpass_filter', 'default_connection': True},
]

OSC_ADSR = [
    {'index': (0, 0), 'operation': 'osc', 'default_connection': True},
    {'index': (0, 1), 'operation': 'env_adsr', 'default_connection': True},
]

OSC_TREMOLO = [
    {'index': (0, 0), 'operation': 'osc', 'default_connection': True},
    {'index': (0, 1), 'operation': 'tremolo', 'default_connection': True}
]


FM_SINE = [
    {'index': (0, 0), 'operation': 'lfo'},
    {'index': (0, 1), 'operation': 'fm_sine', 'control_input': [(0, 0)], 'default_connection': True}
]


FM_FILTER_ADSR = [
    {'index': (0, 0), 'operation': 'lfo'},
    {'index': (0, 1), 'operation': 'fm', 'control_input': [(0, 0)], 'default_connection': True},
    {'index': (0, 2), 'operation': 'filter', 'default_connection': True},
    {'index': (0, 3), 'operation': 'env_adsr', 'default_connection': True}
]

# Complex multi-channel presets
BASIC_FLOW = [
    {'index': (0, 0), 'operation': 'lfo'},
    {'index': (0, 1), 'operation': 'fm_sine', 'control_input': [(0, 0)], 'outputs': [(0, 2)]},
    {'index': (1, 0), 'operation': 'lfo'},
    {'index': (1, 1), 'operation': 'fm_square', 'control_input': [(1, 0)], 'outputs': [(0, 2)]},
    {'index': (2, 0), 'operation': 'lfo'},
    {'index': (2, 1), 'operation': 'fm_saw', 'control_input': [(2, 0)], 'outputs': [(0, 2)]},
    {'index': (0, 2), 'operation': 'mix', 'audio_input': [(0, 1), (1, 1), (2, 1)], 'default_connection': True},
    {'index': (0, 3), 'operation': 'env_adsr', 'default_connection': True},
    {'index': (0, 4), 'operation': 'lowpass_filter', 'default_connection': True}
]

REDUCED_BASIC_FLOW = [
    {'index': (0, 0), 'operation': 'fm_saw', 'outputs': [(0, 1)]},
    {'index': (1, 0), 'operation': 'fm_square', 'outputs': [(0, 1)]},
    {'index': (2, 0), 'operation': 'fm_sine', 'outputs': [(0, 1)]},
    {'index': (0, 1), 'operation': 'mix', 'audio_input': [(0, 0), (1, 0), (2, 0)], 'default_connection': True},
    {'index': (0, 2), 'operation': 'env_adsr', 'default_connection': True},
    {'index': (0, 3), 'operation': 'lowpass_filter', 'default_connection': True},
]

# Modular preset
MODULAR = [
    {'index': (0, 0), 'operation': 'lfo_sine', 'outputs': [(0, 6), (1, 1)], 'switch_outputs': True,
     'active_prob': 0.75},
    {'index': (1, 1), 'operation': 'fm', 'control_input': [(0, 0)], 'outputs': [(0, 2), (1, 2), (2, 2)],
     'switch_outputs': True, 'allow_multiple': False, 'active_prob': 0.75},
    {'index': (0, 2), 'operation': 'fm_sine', 'control_input': [(1, 1)], 'outputs': [(0, 3)]},
    {'index': (1, 2), 'operation': 'fm_saw', 'control_input': [(1, 1)], 'outputs': [(0, 3)]},
    {'index': (2, 2), 'operation': 'fm_square', 'control_input': [(1, 1)], 'outputs': [(0, 3)]},
    {'index': (0, 3), 'operation': 'mix', 'audio_input': [(0, 2), (1, 2), (2, 2)], 'default_connection': True},
    {'index': (0, 4), 'operation': 'env_adsr', 'default_connection': True},
    {'index': (0, 5), 'operation': 'lowpass_filter', 'default_connection': True},
    {'index': (0, 6), 'operation': 'tremolo', 'control_input': [(0, 0)], 'default_connection': True}
]

synth_presets_dict = {'BASIC_FLOW': BASIC_FLOW,
                      'FM_FILTER_ADSR': FM_FILTER_ADSR,
                      'MODULAR': MODULAR,
                      'REDUCED_BASIC_FLOW': REDUCED_BASIC_FLOW,
                      'OSC_FILTER': OSC_FILTER,
                      'OSC_ADSR': OSC_ADSR,
                      'OSC_TREMOLO': OSC_TREMOLO,
                      'FM_SINE': FM_SINE,
                      }
