from config import SynthConfig, Config

synth_cfg = SynthConfig()


BASIC_FLOW = [
    {'index': (0, 0), 'operation': 'lfo', 'default_connection': True, 'synth_config': synth_cfg},
    {'index': (0, 1), 'operation': 'fm', 'default_connection': True, 'synth_config': synth_cfg},
    {'index': (1, 0), 'operation': 'lfo', 'default_connection': True, 'synth_config': synth_cfg},
    {'index': (1, 1), 'operation': 'fm', 'input_list': [[1, 0]], 'output': [0, 2], 'synth_config': synth_cfg},
    {'index': (1, 2), 'operation': None, 'input_list': None, 'synth_config': synth_cfg},
    {'index': (0, 2), 'operation': 'mix', 'input_list': [[0, 1], [1, 1]], 'synth_config': synth_cfg},
    {'index': (0, 3), 'operation': 'filter', 'default_connection': True, 'synth_config': synth_cfg},
    {'index': (0, 4), 'operation': 'env_adsr', 'default_connection': True, 'synth_config': synth_cfg}
]

BASIC_FLOW_NO_ADSR = [
    {'index': (0, 0), 'operation': 'lfo', 'default_connection': True, 'synth_config': synth_cfg},
    {'index': (0, 1), 'operation': 'fm', 'default_connection': True, 'synth_config': synth_cfg},
    {'index': (1, 0), 'operation': 'lfo', 'default_connection': True, 'synth_config': synth_cfg},
    {'index': (1, 1), 'operation': 'fm', 'input_list': [[1, 0]], 'output': [0, 2], 'synth_config': synth_cfg},
    {'index': (1, 2), 'operation': None, 'input_list': None, 'synth_config': synth_cfg},
    {'index': (0, 2), 'operation': 'mix', 'input_list': [[0, 1], [1, 1]], 'synth_config': synth_cfg},
    {'index': (0, 3), 'operation': 'filter', 'default_connection': True, 'synth_config': synth_cfg},
]

BASIC_FLOW_NO_ADSR_NO_FILTER = [
    {'index': (0, 0), 'operation': 'lfo', 'default_connection': True, 'synth_config': synth_cfg},
    {'index': (0, 1), 'operation': 'fm', 'default_connection': True, 'synth_config': synth_cfg},
    {'index': (1, 0), 'operation': 'lfo', 'default_connection': True, 'synth_config': synth_cfg},
    {'index': (1, 1), 'operation': 'fm', 'input_list': [[1, 0]], 'output': [0, 2], 'synth_config': synth_cfg},
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

FM_FILTER = [
    {'index': (0, 0), 'operation': 'lfo', 'default_connection': True, 'synth_config': synth_cfg},
    {'index': (0, 1), 'operation': 'fm', 'default_connection': True, 'synth_config': synth_cfg},
    {'index': (0, 2), 'operation': 'filter', 'default_connection': True, 'synth_config': synth_cfg}
]

synth_presets_dict = {'BASIC_FLOW': BASIC_FLOW, 'LFO': LFO, 'OSC': OSC, 'FM': FM,
                      'BASIC_FLOW_NO_ADSR': BASIC_FLOW_NO_ADSR,
                      'BASIC_FLOW_NO_ADSR_NO_FILTER': BASIC_FLOW_NO_ADSR_NO_FILTER,
                      'FM_FILTER': FM_FILTER}
