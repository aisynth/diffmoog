import numpy as np
from dataclasses import dataclass


@dataclass
class SynthConstants:

    wave_type_dict = {"sine": 0,
                      "square": 1,
                      "sawtooth": 2}

    filter_type_dict = {"low_pass": 0,
                        "high_pass": 1}

    semitones_max_offset: int = 24
    middle_c_freq: float = 261.6255653005985
    min_amp: float = 0.05
    max_amp: float = 1
    min_mod_index: float = 0.01
    max_mod_index: float = 0.3
    min_lfo_freq: float = 0.5
    max_lfo_freq: float = 20
    min_filter_freq: float = 100
    max_filter_freq: float = 8000
    min_resonance_val: float = 0.01
    max_resonance_val: float = 10
    min_amount_tremolo: float = 0.05
    max_amount_tremolo: float = 1

    sample_rate: int = 16000
    fixed_note_off: bool = True

    # non-active operation defaults
    non_active_waveform_default = 'sine'
    non_active_freq_default = 0
    non_active_amp_default = 0
    non_active_mod_index_default = 0
    non_active_tremolo_amount_default = 0

    # When predicting oscillator frequency by regression, the defines are used to normalize the output from the model
    margin: float = 200
    # --------------------------------------
    # -----------Modular Synth--------------
    # --------------------------------------
    # Modular Synth attributes:

    # Seed for random parameters generator
    seed = 2345124

    # Modular synth possible modules from synth_modules.py
    modular_synth_operations = ['osc', 'fm', 'lfo', 'mix', 'filter', 'env_adsr', 'fm_lfo', 'lfo_sine', 'lfo_non_sine',
                                'fm_sine', 'fm_square', 'fm_saw', 'lowpass_filter']

    modular_synth_params = {'osc': ['amp', 'freq', 'waveform'],
                            'lfo_sine': ['amp', 'active', 'freq'],
                            'lfo_non_sine': ['amp', 'freq', 'waveform'],
                            'lfo': ['amp', 'freq', 'waveform'],
                            'fm_lfo': ['active', 'fm_active', 'freq_c', 'waveform', 'mod_index', 'amp_c'],
                            'fm': ['freq_c', 'waveform', 'mod_index'],
                            'fm_sine': ['active', 'fm_active', 'amp_c', 'freq_c', 'mod_index'],
                            'fm_square': ['active', 'fm_active', 'amp_c', 'freq_c', 'mod_index'],
                            'fm_saw': ['active', 'fm_active', 'amp_c', 'freq_c', 'mod_index'],
                            'mix': [],
                            'filter': ['filter_freq', 'filter_type'],
                            'lowpass_filter': ['filter_freq', 'resonance'],
                            'env_adsr': ['attack_t', 'decay_t', 'sustain_t', 'sustain_level', 'release_t'],
                            'amplitude_shape': ['envelope', 'attack_t', 'decay_t', 'sustain_t', 'sustain_level',
                                                'release_t'],
                            'tremolo': ['amount', 'active', 'fm_active']}

    def __post_init__(self):
        self.wave_type_dic_inv = {v: k for k, v in self.wave_type_dict.items()}
        self.filter_type_dic_inv = {v: k for k, v in self.filter_type_dict.items()}

        # build a list of possible frequencies
        self.semitones_list = [*range(-self.semitones_max_offset, self.semitones_max_offset + 1)]
        self.osc_freq_list = [self.middle_c_freq * (2 ** (1 / 12)) ** x for x in self.semitones_list]
        self.osc_freq_dic = {round(key, 4): value for value, key in enumerate(self.osc_freq_list)}
        self.osc_freq_dic_inv = {v: k for k, v in self.osc_freq_dic.items()}
        self.oscillator_freq = self.osc_freq_list[-1] + self.margin

        self.all_params_presets = {
            'lfo': {'freq': np.asarray([0.5] + [k+1 for k in range(int(self.max_lfo_freq))])},
            'fm': {'freq_c': np.asarray(self.osc_freq_list),
                   'mod_index': np.linspace(0, self.max_mod_index, 16)},
            'filter': {'filter_freq': np.asarray([100*1.4**k for k in range(14)])}
        }
        
        self.sampling_configurations = self._create_sampling_config()
        self.param_configs = self._create_op_types_dict()
        
    def _create_sampling_config(self):
        sampling_configurations = {
            'uniform_amp': {'type': 'uniform',
                            'values': (self.min_amp, self.max_amp),
                            'non_active_default': self.non_active_amp_default},
            'constant_amp': {'type': 'choice',
                             'values': (1,)},
            'osc_freq': {'type': 'choice',
                         'values': self.osc_freq_list,
                         'non_active_default': self.non_active_freq_default},
            'waveform': {'type': 'choice',
                         'values': list(self.wave_type_dict),
                         'non_active_default': self.non_active_waveform_default},
            'non_sine_waveform': {'type': 'choice',
                                  'values': [k for k in self.wave_type_dict.keys() if k != 'sine'],
                                  'non_active_default': self.non_active_waveform_default},
            'lfo_freq': {'type': 'uniform',
                         'values': (self.min_lfo_freq, self.max_lfo_freq),
                         'non_active_default': self.non_active_freq_default},
            'fm_freq': {'type': 'freq_c',
                        'non_active_default': self.non_active_freq_default},
            'mod_index': {'type': 'uniform',
                          'values': (self.min_mod_index, self.max_mod_index),
                          'non_active_default': self.non_active_mod_index_default,
                          'activity_signal': 'fm_active'},
            'filter_freq': {'type': 'uniform',
                            'values': (self.min_filter_freq, self.max_filter_freq)},
            'filter_type': {'type': 'choice',
                            'values': list(self.filter_type_dict)},
            'resonance': {'type': 'uniform',
                          'values': (self.min_resonance_val, self.max_resonance_val)},
            'amount': {'type': 'uniform',
                       'values': (self.min_amount_tremolo, self.max_amount_tremolo),
                       'non_active_default': self.non_active_tremolo_amount_default}
        }
        
        return sampling_configurations

    def _create_op_types_dict(self):
        sampling_configurations = self.sampling_configurations
        op_types = {
            'osc': {'amp': sampling_configurations['uniform_amp'], 'freq': sampling_configurations['osc_freq'],
                    'waveform': sampling_configurations['waveform']},
            'lfo_sine': {'freq': sampling_configurations['lfo_freq'], 'amp': sampling_configurations['constant_amp']},
            'lfo_non_sine': {'freq': sampling_configurations['lfo_freq'], 'amp': sampling_configurations['constant_amp'],
                             'waveform': sampling_configurations['non_sine_waveform']},
            'lfo': {'freq': sampling_configurations['lfo_freq'], 'waveform': sampling_configurations['waveform'],
                    'amp': sampling_configurations['constant_amp']},
            'fm_lfo': {'freq_c': sampling_configurations['lfo_freq'], 'waveform': sampling_configurations['waveform'],
                       'mod_index': sampling_configurations['mod_index'],
                       'amp_c': sampling_configurations['constant_amp']},
            'fm': {'freq_c': sampling_configurations['fm_freq'], 'waveform': sampling_configurations['waveform'],
                   'mod_index': sampling_configurations['mod_index']},
            'fm_sine': {'amp_c': sampling_configurations['uniform_amp'], 'freq_c': sampling_configurations['fm_freq'],
                        'mod_index': sampling_configurations['mod_index']},
            'fm_square': {'amp_c': sampling_configurations['uniform_amp'], 'freq_c': sampling_configurations['fm_freq'],
                          'mod_index': sampling_configurations['mod_index']},
            'fm_saw': {'amp_c': sampling_configurations['uniform_amp'], 'freq_c': sampling_configurations['fm_freq'],
                       'mod_index': sampling_configurations['mod_index']},
            'mix': {},
            'filter': {'filter_freq': sampling_configurations['filter_freq'],
                       'filter_type': sampling_configurations['filter_type']},
            'lowpass_filter': {'filter_freq': sampling_configurations['filter_freq'],
                               'resonance': sampling_configurations['resonance']},
            'env_adsr': {'attack_t', 'decay_t', 'sustain_t', 'sustain_level', 'release_t'},
            'amplitude_shape': {'envelope', 'attack_t', 'decay_t', 'sustain_t', 'sustain_level',
                                'release_t'},
            'tremolo': {'amount': sampling_configurations['amount']}
        }

        return op_types


synth_structure = SynthConstants()