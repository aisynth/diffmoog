import random
from typing import List

import numpy as np

from synth.synth_architecture import SynthModularCell
from synth.synth_constants import SynthConstants


def generate_random_params(self, synth_matrix: List[List[SynthModularCell]], synth_structure: SynthConstants,
                           num_sounds_=1):
    for layer in range(synth_structure.num_layers):
        for channel in range(synth_structure.num_channels):
            cell = self.synth_matrix[channel][layer]
            operation = cell.operation
            parameters = cell.parameters
            if parameters is None:
                continue

            activity_flags = parameters['active']
            if 'fm_active' in parameters.keys():
                fm_activity_flags = parameters['fm_active']
            operation_params = {}
            if operation == 'osc':
                operation_params = {'amp': np.random.random_sample(size=num_sounds_),
                                    'freq': np.asarray(random.choices(synth_structure.osc_freq_list, k=num_sounds_)),
                                    'waveform': random.choices(list(synth_structure.wave_type_dict), k=num_sounds_)}

            elif operation == 'lfo':
                operation_params = {'freq': np.random.uniform(low=0, high=synth_structure.max_lfo_freq, size=num_sounds_),
                                    'waveform': random.choices([k for k in synth_structure.wave_type_dict.keys()],
                                                               k=num_sounds_)}

            elif operation == 'lfo_non_sine':
                operation_params = {'freq': np.random.uniform(low=0, high=synth_structure.max_lfo_freq, size=num_sounds_),
                                    'waveform': random.choices(
                                        [k for k in synth_structure.wave_type_dict.keys() if k != 'sine'],
                                        k=num_sounds_)}

            elif operation == 'lfo_sine':
                freq = np.random.uniform(low=synth_structure.min_lfo_freq,
                                         high=synth_structure.max_lfo_freq,
                                         size=num_sounds_)

                operation_params['freq'] = [freq[k] if activity_flags[k] else synth_structure.non_active_freq_default
                                            for k in range(num_sounds_)]

            elif operation == 'fm':
                operation_params = {'freq_c': self._sample_c_freq(synth_structure, num_sounds_),
                                    'waveform': random.choices(list(synth_structure.wave_type_dict), k=num_sounds_),
                                    'mod_index': np.random.uniform(low=0, high=synth_structure.max_mod_index,
                                                                   size=num_sounds_)}

            elif operation == 'fm_lfo':
                freq_c = np.random.uniform(low=synth_structure.min_lfo_freq,
                                           high=synth_structure.max_lfo_freq,
                                           size=num_sounds_)
                waveform = random.choices(list(synth_structure.wave_type_dict), k=num_sounds_)
                mod_index = np.random.uniform(low=synth_structure.min_mod_index,
                                              high=synth_structure.max_mod_index,
                                              size=num_sounds_)

                operation_params['freq_c'] = np.asarray \
                    ([freq_c[k] if activity_flags[k] else synth_structure.non_active_freq_default
                                                         for k in range(num_sounds_)])
                operation_params['waveform'] = [waveform[k] if activity_flags[k]
                                                else synth_structure.non_active_waveform_default
                                                for k in range(num_sounds_)]
                operation_params['mod_index'] = np.asarray([mod_index[k] if fm_activity_flags[k]
                                                            else synth_structure.non_active_mod_index_default
                                                            for k in range(num_sounds_)])

            elif operation in ['fm_sine', 'fm_square', 'fm_saw']:
                amp_c = np.random.uniform(low=synth_structure.min_amp, high=synth_structure.max_amp, size=num_sounds_)
                freq_c = self._sample_c_freq(synth_structure, num_sounds_)
                mod_index = np.random.uniform(low=0, high=synth_structure.max_mod_index, size=num_sounds_)
                operation_params['amp_c'] = np.asarray \
                    ([amp_c[k] if activity_flags[k] else synth_structure.non_active_amp_default
                                                        for k in range(num_sounds_)])
                operation_params['freq_c'] = np.asarray \
                    ([freq_c[k] if activity_flags[k] else synth_structure.non_active_freq_default
                                                         for k in range(num_sounds_)])
                operation_params['mod_index'] = np.asarray([mod_index[k] if fm_activity_flags[k]
                                                            else synth_structure.non_active_mod_index_default
                                                            for k in range(num_sounds_)])

            elif operation == 'mix':
                operation_params = None
                continue

            elif operation == 'amplitude_shape':
                attack_t, decay_t, sustain_t, sustain_level, release_t = \
                    self.generate_random_adsr_values(num_sounds_=num_sounds_)

                operation_params = {'attack_t': attack_t,
                                    'decay_t': decay_t,
                                    'sustain_t': sustain_t,
                                    'sustain_level': sustain_level,
                                    'release_t': release_t,
                                    'envelope': torch.full([num_sounds_], -1)}

                # attack_t, decay_t, sustain_t, sustain_level, release_t = \
                #     self.generate_random_adsr_values(num_sounds_=num_sounds_)

                # envelope = self.make_envelope_shape(attack_t,
                #                                     decay_t,
                #                                     sustain_t,
                #                                     sustain_level,
                #                                     release_t,
                #                                     num_sounds_)
                # params = {'envelope': envelope}

            elif operation == 'filter':
                operation_params = {'filter_type': random.choices(list(synth_structure.filter_type_dict), k=num_sounds_),
                                    'filter_freq': np.random.uniform(low=synth_structure.min_filter_freq,
                                                                     high=synth_structure.max_filter_freq,
                                                                     size=num_sounds_)}

            elif operation == 'lowpass_filter':
                operation_params = {'filter_freq': np.random.uniform(low=synth_structure.min_filter_freq,
                                                                     high=synth_structure.max_filter_freq,
                                                                     size=num_sounds_),
                                    'resonance': np.random.uniform(low=synth_structure.min_resonance_val,
                                                                   high=synth_structure.max_resonance_val,
                                                                   size=num_sounds_)}
            elif operation == 'env_adsr':
                attack_t, decay_t, sustain_t, sustain_level, release_t = \
                    self.generate_random_adsr_values(num_sounds_=num_sounds_)

                operation_params = {'attack_t': attack_t,
                                    'decay_t': decay_t,
                                    'sustain_t': sustain_t,
                                    'sustain_level': sustain_level,
                                    'release_t': release_t}

            elif operation == 'tremolo':
                amount = np.random.uniform(low=synth_structure.min_amount_tremolo,
                                           high=synth_structure.max_amount_tremolo,
                                           size=num_sounds_)

                operation_params['amount'] = np.asarray([
                    amount[k] if activity_flags[k] else synth_structure.non_active_tremolo_amount_default for k in
                    range(num_sounds_)])

            elif operation is None:
                operation_params = None
                #
                # if operation_params is not None:
                #     for key, val in operation_params.items():
                #         if isinstance(val, numpy.ndarray):
                #             operation_params[key] = val.tolist()

                if num_sounds_ == 1:
                    for key, value in operation_params.items():
                        operation_params[key] = value[0]

            cell.parameters.update(operation_params)


def generate_activations_and_chains(self, num_sounds_=1, train: bool=False):
    # todo: generalize code to go over all cells
    rng = np.random.default_rng()

    # lfo-sine
    lfo_sine_cell = self.synth_matrix[0][0]
    operation = lfo_sine_cell.operation
    audio_input = lfo_sine_cell.audio_input
    lfo_sine_outputs = lfo_sine_cell.outputs

    lfo_sine_output = rng.choice(lfo_sine_outputs, size=num_sounds_, axis=0).tolist()
    if train:
        probs = [0.75, 0.25]
    else:
        probs = [0.25, 0.75]
    lfo_sine_params = {'active': np.random.choice([True, False], size=num_sounds_, p=probs)}
    lfo_sine_params['output'] = [lfo_sine_output[k] if lfo_sine_params['active'][k] else [-1, -1] for k in
                                 range(num_sounds_)]
    lfo_sine_cell.parameters = lfo_sine_params

    # tremolo
    tremolo_cell = self.synth_matrix[0][6]
    tremolo_params = {
        'active': [True if (lfo_sine_params['active'][k] and lfo_sine_params['output'][k] == [0, 6]) else False for
                   k in range(num_sounds_)]}
    tremolo_cell.parameters = tremolo_params

    # fm_lfo
    fm_lfo_cell = self.synth_matrix[1][1]
    fm_lfo_outputs = fm_lfo_cell.outputs

    fm_lfo_random_activeness = np.random.choice([True, False], size=num_sounds_, p=[0.75, 0.25])
    fm_lfo_output = rng.choice(fm_lfo_outputs, size=num_sounds_, axis=0).tolist()

    fm_lfo_params = {'fm_active': [True if (lfo_sine_params['active'][k] and lfo_sine_params['output'][k] == [1, 1])
                                   else False for k in range(num_sounds_)]}
    fm_lfo_params['active'] = [True if fm_lfo_params['fm_active'][k] or fm_lfo_random_activeness[k] else False for k
                               in range(num_sounds_)]
    fm_lfo_params['output'] = [fm_lfo_output[k] if fm_lfo_params['active'][k] else [-1, -1] for k in range(num_sounds_)]
    fm_lfo_cell.parameters = fm_lfo_params

    oscillator_options = [['sine'], ['saw'], ['square'], ['sine', 'saw'], ['sine', 'square'], ['saw', 'square'],
                          ['sine', 'saw', 'square']]
    oscillator_activeness = rng.choice(oscillator_options, size=num_sounds_, axis=0).tolist()

    # sine oscillator
    sine_cell = self.synth_matrix[0][2]
    sine_params = {
        'fm_active': [True if (fm_lfo_params['active'][k] and fm_lfo_params['output'][k] == [0, 2]) else False for
                      k in range(num_sounds_)]}
    sine_params['active'] = [True if sine_params['fm_active'][k] or 'sine' in oscillator_activeness[k] else False
                             for k in range(num_sounds_)]

    sine_cell.parameters = sine_params

    # saw oscillator
    saw_cell = self.synth_matrix[1][2]
    saw_params = {
        'fm_active': [True if (fm_lfo_params['active'][k] and fm_lfo_params['output'][k] == [1, 2]) else False for
                      k in range(num_sounds_)]}
    saw_params['active'] = [True if saw_params['fm_active'][k] or 'saw' in oscillator_activeness[k] else False
                            for k in range(num_sounds_)]

    saw_cell.parameters = saw_params

    # square oscillator
    square_cell = self.synth_matrix[2][2]
    square_params = {
        'fm_active': [True if (fm_lfo_params['active'][k] and fm_lfo_params['output'][k] == [2, 2]) else False for
                      k in range(num_sounds_)]}
    square_params['active'] = [True if square_params['fm_active'][k] or 'square' in oscillator_activeness[k]
                               else
                               False
                               for k
                               in range(num_sounds_)]

    square_cell.parameters = square_params

    mix_cell = self.synth_matrix[0][3]
    mix_params = {'active': np.random.choice([True], size=num_sounds_)}
    mix_cell.parameters = mix_params

    adsr_cell = self.synth_matrix[0][4]
    adsr_params = {'active': np.random.choice([True], size=num_sounds_)}
    adsr_cell.parameters = adsr_params

    filter_cell = self.synth_matrix[0][5]
    filters_params = {'active': np.random.choice([True], size=num_sounds_)}
    filter_cell.parameters = filters_params

def generate_random_adsr_values(self, num_sounds_=1):
        attack_t = np.random.random(size=num_sounds_)
        decay_t = np.random.random(size=num_sounds_)
        sustain_t = np.random.random(size=num_sounds_)
        release_t = np.random.random(size=num_sounds_)

        if self.synth_cfg.fixed_note_off:
            adsr_sum = attack_t + decay_t + sustain_t

            ads_time = self.synth_cfg.note_off_time
            release_t = release_t * (self.signal_duration_sec - self.synth_cfg.note_off_time)
        else:
            adsr_sum = attack_t + decay_t + sustain_t + release_t
            ads_time = self.signal_duration_sec
            release_t = (release_t / adsr_sum) * ads_time

        attack_t = (attack_t / adsr_sum) * ads_time
        decay_t = (decay_t / adsr_sum) * ads_time
        sustain_t = (sustain_t / adsr_sum) * ads_time

        # fixing a numerical issue in case the ADSR times exceeds signal length
        adsr_aggregated_time = attack_t + decay_t + sustain_t + release_t
        overflow_indices = [idx for idx, val in enumerate(adsr_aggregated_time) if
                            val > self.signal_duration_sec]
        attack_t[overflow_indices] -= 1e-6
        decay_t[overflow_indices] -= 1e-6
        sustain_t[overflow_indices] -= 1e-6
        release_t[overflow_indices] -= 1e-6

        sustain_level = np.random.random(size=num_sounds_)

        return attack_t, decay_t, sustain_t, sustain_level, release_t


def _sample_c_freq(synth_cfg: SynthConfig, num_sounds_: int):

    osc_freq_list = np.asarray(synth_cfg.osc_freq_list)

    base_freqs = np.random.uniform(low=synth_cfg.osc_freq_list[0],
                                   high=synth_cfg.osc_freq_list[-1],
                                   size=num_sounds_)

    idx = np.searchsorted(synth_cfg.osc_freq_list, base_freqs, side="left")
    idx = idx - (np.abs(base_freqs - osc_freq_list[idx - 1]) < np.abs(base_freqs - osc_freq_list[idx]))
    return osc_freq_list[idx]