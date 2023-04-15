import random
from typing import List
from itertools import chain, combinations

import numpy as np

from synth.synth_architecture import SynthModularCell
from synth.synth_constants import SynthConstants


class ParametersSampler:

    def __init__(self, synth_structure: SynthConstants):
        self.synth_structure = synth_structure

    def generate_activations_and_chains(self, synth_matrix: List[List[SynthModularCell]], signal_len: float,
                                        note_off_time: float, num_sounds_=1):

        rng = np.random.default_rng()

        n_channels = len(synth_matrix)
        n_layers = len(synth_matrix[0])

        # Propagate activations through layers
        output_params_dict = {}
        for layer_idx in range(n_layers):
            for channel_idx in range(n_channels):

                cell_params = {}
                cell = synth_matrix[channel_idx][layer_idx]

                operation = cell.operation
                if operation is None or operation.lower() in ['none', 'mix']:
                    continue

                op_params = self.synth_structure.modular_synth_params[operation]

                if cell.control_input is not None:
                    assert len(cell.control_input) == 1
                    control_input_cell = synth_matrix[cell.control_input[0][0]][cell.control_input[0][1]]
                    if control_input_cell.switch_outputs is not None:
                        has_control_input = [tuple(cell.index) in control_cell_output for control_cell_output
                                             in control_input_cell.parameters['output']]
                    else:
                        has_control_input = [True for _ in range(num_sounds_)]
                    cell_params['fm_active'] = has_control_input
                else:
                    has_control_input = [False for _ in range(num_sounds_)]
                    if 'fm_active' in op_params:
                        cell_params['fm_active'] = has_control_input

                if 'active' in op_params:
                    active_prob = cell.active_prob if cell.active_prob is not None else 1.0
                    random_activeness = np.random.choice([True, False], size=num_sounds_,
                                                         p=[active_prob, 1 - active_prob])

                    is_active = np.logical_or(has_control_input, random_activeness)
                    cell_params['active'] = is_active
                else:
                    is_active = [True for _ in range(num_sounds_)]

                if cell.switch_outputs is not None:
                    if cell.allow_multiple_outputs:
                        outputs_powerset = self.powerset(cell.outputs)
                        selected_outputs = rng.choice(outputs_powerset, size=num_sounds_, axis=0).tolist()
                    else:
                        selected_outputs = rng.choice(cell.outputs, size=num_sounds_, axis=0).tolist()
                        selected_outputs = [[tuple(x)] for x in selected_outputs]

                    selected_outputs = [selected_outputs[k] if act else [(-1, -1)] for k, act in enumerate(is_active)]
                    cell_params['output'] = selected_outputs

                if operation in ['env_adsr', 'amplitude_shape', 'lowpass_filter_adsr']:
                    sampled_params = self._generate_random_adsr_values(signal_len, note_off_time,
                                                                       num_sounds_=num_sounds_)
                    if operation == 'lowpass_filter_adsr':
                        sampled_non_adsr_params = self._sample_parameters(operation, cell_params.get('active', None),
                                                                 cell_params.get('fm_active', None), num_sounds_)
                        sampled_params.update(sampled_non_adsr_params)

                else:
                    n_input_sounds = len(cell_params.get("audio_input", [1]))
                    sampled_params = self._sample_parameters(operation, cell_params.get('active', None),
                                                             cell_params.get('fm_active', None), num_sounds_,
                                                             n_input_sounds)

                cell_params.update(sampled_params)

                cell.parameters = cell_params
                output_params_dict[cell.index] = {'operation': operation, 'parameters': cell_params}

        return output_params_dict

    def _sample_parameters(self, op_name: str, is_active, is_fm_active, batch_size: int = 1, n_input_sounds=1):

        synth_structure = self.synth_structure
        sampling_config = synth_structure.param_configs[op_name]

        params_dict = {}
        for param_name, param_config in sampling_config.items():

            # Sample according to param type and possible values
            if param_config['type'] == 'freq_c':
                sampled_values = self._sample_c_freq(batch_size)
            elif param_config['type'] == 'uniform':
                sampled_values = np.random.uniform(low=param_config['values'][0], high=param_config['values'][1],
                                                   size=batch_size)
            elif param_config['type'] == 'unit_uniform':
                if n_input_sounds == 1:
                    sampled_values = np.random.uniform(low=param_config['values'][0], high=param_config['values'][1],
                                                       size=batch_size)
                else:
                    sampled_values = np.random.uniform(low=param_config['values'][0], high=param_config['values'][1],
                                                   size=(n_input_sounds, batch_size))
                    sampled_values = sampled_values / np.sum(sampled_values, axis=0) * param_config['sum']
            elif param_config['type'] == 'choice':
                sampled_values = random.choices(param_config['values'], k=batch_size)
            else:
                raise ValueError(f'Unrecognized parameter type {param_config["type"]}')

            # Apply activity defaults
            if param_config.get('activity_signal', None) == 'fm_active':
                activity_signal = is_fm_active
            else:
                activity_signal = is_active

            if activity_signal is not None and param_config.get('non_active_default', None) is not None:
                sampled_values = np.array([val if activity_signal[k] else param_config['non_active_default']
                                  for k, val in enumerate(sampled_values)])

            params_dict[param_name] = sampled_values

        return params_dict

    def _generate_random_adsr_values(self, signal_duration: float, note_off_time: float, num_sounds_=1):

        synth_structure = self.synth_structure

        attack_t = np.random.random(size=num_sounds_)
        decay_t = np.random.random(size=num_sounds_)
        sustain_t = np.random.random(size=num_sounds_)
        release_t = np.random.random(size=num_sounds_)

        if synth_structure.fixed_note_off:
            adsr_sum = attack_t + decay_t + sustain_t
            ads_time = note_off_time
            release_t = release_t * (signal_duration - note_off_time)
        else:
            adsr_sum = attack_t + decay_t + sustain_t + release_t
            ads_time = signal_duration
            release_t = (release_t / adsr_sum) * ads_time

        attack_t = (attack_t / adsr_sum) * ads_time
        decay_t = (decay_t / adsr_sum) * ads_time
        sustain_t = (sustain_t / adsr_sum) * ads_time

        # fixing a numerical issue in case the ADSR times exceeds signal length
        adsr_aggregated_time = attack_t + decay_t + sustain_t + release_t
        overflow_indices = [idx for idx, val in enumerate(adsr_aggregated_time) if
                            val > signal_duration]
        attack_t[overflow_indices] -= 1e-6
        decay_t[overflow_indices] -= 1e-6
        sustain_t[overflow_indices] -= 1e-6
        release_t[overflow_indices] -= 1e-6

        sustain_level = np.random.random(size=num_sounds_)

        adsr_params = {'attack_t': attack_t, 'decay_t': decay_t, 'sustain_t': sustain_t,
                       'sustain_level': sustain_level, 'release_t': release_t}

        return adsr_params

    def _sample_c_freq(self, num_sounds_: int):

        synth_structure = self.synth_structure

        osc_freq_list = np.asarray(synth_structure.osc_freq_list)

        base_freqs = np.random.uniform(low=synth_structure.osc_freq_list[0],
                                       high=synth_structure.osc_freq_list[-1],
                                       size=num_sounds_)

        idx = np.searchsorted(synth_structure.osc_freq_list, base_freqs, side="left")
        idx = idx - (np.abs(base_freqs - osc_freq_list[idx - 1]) < np.abs(base_freqs - osc_freq_list[idx]))
        return osc_freq_list[idx]

    @staticmethod
    def powerset(iterable):
        "powerset([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
        s = list(iterable)
        powerset = chain.from_iterable(combinations(s, r) for r in range(1, len(s) + 1))
        return list(powerset)
