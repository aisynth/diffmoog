from config import SynthConfig, Config

synth_cfg = SynthConfig()

BASIC_FLOW = [
    {'index': (0, 0), 'operation': 'osc', 'default_connection': True, 'synth_config': synth_cfg},
    {'index': (0, 1), 'operation': 'fm', 'default_connection': True, 'synth_config': synth_cfg},
    {'index': (1, 0), 'operation': 'osc', 'default_connection': True, 'synth_config': synth_cfg},
    {'index': (1, 1), 'operation': 'fm', 'input_list': [[1, 0]], 'output': [0, 2], 'synth_config': synth_cfg},
    {'index': (1, 2), 'operation': None, 'input_list': None, 'synth_config': synth_cfg},
    {'index': (0, 2), 'operation': 'mix', 'input_list': [[0, 1], [1, 1]], 'synth_config': synth_cfg},
    {'index': (0, 3), 'operation': 'filter', 'default_connection': True, 'synth_config': synth_cfg},
    {'index': (0, 4), 'operation': 'env_adsr', 'default_connection': True, 'synth_config': synth_cfg}
]

OSC = [
    {'index': (0, 0), 'operation': 'osc', 'default_connection': True, 'synth_config': synth_cfg}
]

FM = [
    {'index': (0, 0), 'operation': 'osc', 'default_connection': True, 'synth_config': synth_cfg},
    {'index': (0, 1), 'operation': 'fm', 'default_connection': True, 'synth_config': synth_cfg}
]