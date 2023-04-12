import math

import numpy as np
import torch

from synth.synth_constants import SynthConstants


class Normalizer:
    """ normalize/de-normalise regression parameters"""

    def __init__(self, note_off_time: float, signal_duration: int, synth_structure: SynthConstants, clamp_adsr=True,
                 clip=False):

        self.signal_duration = signal_duration
        self.clamp_adsr = clamp_adsr
        self.synth_structure = synth_structure

        self.mod_index_normalizer = MinMaxNormaliser(target_min_val=0,
                                                     target_max_val=1,
                                                     original_min_val=synth_structure.min_mod_index,
                                                     original_max_val=synth_structure.max_mod_index,
                                                     clip=clip)

        self.fm_lfo_mod_index_normalizer = MinMaxNormaliser(target_min_val=0,
                                                            target_max_val=1,
                                                            original_min_val=synth_structure.min_fm_lfo_mod_index,
                                                            original_max_val=synth_structure.max_fm_lfo_mod_index,
                                                            clip=clip)

        self.lfo_freq_normalizer = MinMaxNormaliser(target_min_val=0,
                                                    target_max_val=1,
                                                    original_min_val=synth_structure.min_lfo_freq,
                                                    original_max_val=synth_structure.max_lfo_freq,
                                                    clip=clip)

        self.lfo_phase_normalizer = MinMaxNormaliser(target_min_val=0,
                                                     target_max_val=1,
                                                     original_min_val=0,
                                                     original_max_val=math.pi)

        self.adsr_normalizer = MinMaxNormaliser(target_min_val=0,
                                                target_max_val=1,
                                                original_min_val=0,
                                                original_max_val=note_off_time)

        self.filter_freq_normalizer = MinMaxNormaliser(target_min_val=0,
                                                       target_max_val=1,
                                                       original_min_val=synth_structure.min_filter_freq,
                                                       original_max_val=synth_structure.max_filter_freq,
                                                       clip=clip)

        self.oscillator_freq_normalizer = MinMaxNormaliser(target_min_val=0,
                                                           target_max_val=1,
                                                           original_min_val=0,
                                                           original_max_val=synth_structure.oscillator_freq)

    def normalize(self, parameters_dict: dict):
        normalized_params_dict = {}
        for key, val in parameters_dict.items():
            operation = val['operation'] if isinstance(val['operation'], str) else val['operation'][0]
            params = val['params'] if 'params' in val else val['parameters']

            if operation in ["None", 'mix']:
                continue

            normalized_params_dict[key] = {'operation': operation, 'parameters': {}}
            for param_name, param_val in params.items():
                if (operation in ['osc', 'saw_square_osc'] and param_name in ['freq']) or \
                        (param_name == 'freq_c' and 'lfo' not in operation):
                    normalized_params_dict[key]['parameters'][param_name] = \
                        self.oscillator_freq_normalizer.normalise(params[param_name])
                elif 'lfo' in operation and param_name in ['freq', 'freq_c']:
                    normalized_params_dict[key]['parameters'][param_name] = \
                        self.lfo_freq_normalizer.normalise(params[param_name])
                elif param_name in ['mod_index']:
                    normalized_params_dict[key]['parameters'][param_name] = \
                        self.mod_index_normalizer.normalise(params[param_name])
                elif param_name in ['filter_freq']:
                    normalized_params_dict[key]['parameters'][param_name] = \
                        self.filter_freq_normalizer.normalise(params[param_name])
                elif operation == 'env_adsr' and param_name in ['attack_t', 'decay_t', 'sustain_t']:
                    normalized_params_dict[key]['parameters'][param_name] = \
                        self.adsr_normalizer.normalise(params[param_name])
                else:
                    normalized_params_dict[key]['parameters'][param_name] = params[param_name]

        return normalized_params_dict

    def denormalize(self, parameters_dict: dict):

        denormalized_params_dict = {}
        for key, val in parameters_dict.items():
            operation = val['operation']
            params = val['params'] if 'params' in val else val['parameters']

            if operation == "None":
                continue

            denormalized_params_dict[key] = {'operation': operation, 'parameters': {}}
            for param_name, param_val in params.items():
                if (operation in ['osc', 'saw_square_osc'] and param_name in ['freq']) or \
                        (param_name == 'freq_c' and 'lfo' not in operation):
                    denormalized_params_dict[key]['parameters'][param_name] = \
                        self.oscillator_freq_normalizer.denormalise(params[param_name])
                elif 'lfo' in operation and param_name in ['freq', 'freq_c']:
                    denormalized_params_dict[key]['parameters'][param_name] = \
                        self.lfo_freq_normalizer.denormalise(params[param_name])
                elif param_name in ['mod_index']:
                    denormalized_params_dict[key]['parameters'][param_name] = \
                        self.mod_index_normalizer.denormalise(params[param_name])
                elif param_name in ['filter_freq']:
                    denormalized_params_dict[key]['parameters'][param_name] = \
                        self.filter_freq_normalizer.denormalise(params[param_name])
                elif operation in ['env_adsr', 'lowpass_filter_adsr'] and param_name in ['attack_t', 'decay_t', 'sustain_t']:
                    denormalized_params_dict[key]['parameters'][param_name] = \
                        self.adsr_normalizer.denormalise(params[param_name])
                else:
                    denormalized_params_dict[key]['parameters'][param_name] = params[param_name]

            if operation in ['env_adsr', 'lowpass_filter_adsr'] and self.clamp_adsr:
                denormalized_params_dict[key]['parameters'] = \
                    self._clamp_adsr_params(denormalized_params_dict[key]['parameters'], operation)

        return denormalized_params_dict

    def _clamp_adsr_params(self, params: dict, operation: 'str'):
        """look for adsr operations to send to clamping function"""

        clamped_attack, clamped_decay, clamped_sustain, clamped_release = \
            self._clamp_adsr_superposition(params['attack_t'],
                                           params['decay_t'],
                                           params['sustain_t'],
                                           params['release_t'])
        if operation == 'env_adsr':
            ret_params = {'attack_t': clamped_attack,
                          'decay_t': clamped_decay,
                          'sustain_t': clamped_sustain,
                          'sustain_level': torch.clamp(params['sustain_level'], min=0, max=self.synth_structure.max_amp),
                          'release_t': clamped_release}

        elif operation == 'lowpass_filter_adsr':
            ret_params = {'attack_t': clamped_attack,
                          'decay_t': clamped_decay,
                          'sustain_t': clamped_sustain,
                          'sustain_level': torch.clamp(params['sustain_level'], min=0, max=self.synth_structure.max_amp),
                          'release_t': clamped_release,
                          'intensity': params['intensity'],
                          'filter_freq': params['filter_freq']}


        return ret_params

    def _clamp_adsr_superposition(self, attack_t, decay_t, sustain_t, release_t):
        """This function clamps the superposition of adsr times, so it does not exceed signal length"""

        adsr_length_in_sec = attack_t + decay_t + sustain_t + release_t

        adsr_clamp_indices = torch.nonzero(adsr_length_in_sec >= self.signal_duration, as_tuple=True)[0]

        normalized_attack_list = []
        normalized_decay_list = []
        normalized_sustain_list = []
        normalized_release_list = []

        for i in range(adsr_length_in_sec.shape[0]):
            if i in adsr_clamp_indices.tolist():
                # add small number to normalization to prevent numerical issue where the sum exceeds 1
                normalization_value = adsr_length_in_sec[i] + 1e-3
                normalized_attack = attack_t[i] / normalization_value
                normalized_decay = decay_t[i] / normalization_value
                normalized_sustain = sustain_t[i] / normalization_value
                normalized_release = release_t[i] / normalization_value

            else:
                normalized_attack = attack_t[i]
                normalized_decay = decay_t[i]
                normalized_sustain = sustain_t[i]
                normalized_release = release_t[i]

            normalized_attack_list.append(normalized_attack)
            normalized_decay_list.append(normalized_decay)
            normalized_sustain_list.append(normalized_sustain)
            normalized_release_list.append(normalized_release)

        normalized_attack_tensor = torch.stack(normalized_attack_list)
        normalized_decay_tensor = torch.stack(normalized_decay_list)
        normalized_sustain_tensor = torch.stack(normalized_sustain_list)
        normalized_release_tensor = torch.stack(normalized_release_list)

        return normalized_attack_tensor, normalized_decay_tensor, normalized_sustain_tensor, normalized_release_tensor


class MinMaxNormaliser:
    """MinMaxNormaliser applies min max normalisation to a tensor"""

    def __init__(self, target_min_val, target_max_val, original_min_val, original_max_val, clip=False):
        self.target_min_val = target_min_val
        self.target_max_val = target_max_val
        self.original_max_val = original_max_val
        self.original_min_val = original_min_val
        self.clip = clip

    def normalise(self, array):
        """
        From full range to (0-1)
        """
        norm_array = (array - self.original_min_val) / (self.original_max_val - self.original_min_val)
        if self.clip:
            norm_array[norm_array < 0] = 0
        return norm_array

    def denormalise(self, norm_array):
        """
        From (0-1) range to full range
        """
        array = (norm_array - self.target_min_val) / (self.target_max_val - self.target_min_val)
        array = array * (self.original_max_val - self.original_min_val) + self.original_min_val
        return array


class LogNormaliser:
    """LogNormaliser applies log normalisation to a tensor"""

    def __init__(self):
        pass

    def normalise(self, array):
        # add small value to prevent -inf for log(0)
        norm_array = torch.log(array + 1e-10)
        return norm_array

    def denormalise(self, norm_array):
        array = torch.exp(norm_array) - 1e-10
        return array