import math

import numpy as np
import torch

from synth.synth_constants import SynthConstants
from typing import Dict


class Normalizer:
    """ normalize/de-normalise regression parameters
    The class also takes care of inherent constraints in the parameters (see post_process_inherent_constraints)
    """

    def __init__(self, note_off_time: float, signal_duration: int, synth_structure: SynthConstants, clamp_adsr=True,
                 clip=False):

        self.signal_duration = signal_duration
        self.clamp_adsr = clamp_adsr
        self.synth_structure = synth_structure
        self.note_off_time = torch.tensor(note_off_time)

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

        self.release_t_normalizer = MinMaxNormaliser(target_min_val=0,
                                                     target_max_val=1,
                                                     original_min_val=0,
                                                     original_max_val=signal_duration - note_off_time)

        self.filter_freq_normalizer = MinMaxNormaliser(target_min_val=0,
                                                       target_max_val=1,
                                                       original_min_val=synth_structure.min_filter_freq,
                                                       original_max_val=synth_structure.max_filter_freq,
                                                       clip=clip)

        self.filter_freq_log_normalizer = LogMinMaxNormaliser(target_min_val=0,
                                                              target_max_val=1,
                                                              original_min_val=synth_structure.min_filter_freq,
                                                              original_max_val=synth_structure.max_filter_freq,
                                                              clip=clip)

        self.oscillator_freq_normalizer = MinMaxNormaliser(target_min_val=0,
                                                           target_max_val=1,
                                                           original_min_val=0,
                                                           original_max_val=synth_structure.max_oscillator_freq)

        self.oscillator_freq_log_normalizer = LogMinMaxNormaliser(target_min_val=0,
                                                                  target_max_val=1,
                                                                  original_min_val=synth_structure.min_oscillator_freq,
                                                                  original_max_val=synth_structure.max_oscillator_freq,
                                                                  clip=clip)

        self.adsr_normalizers = {
            'release_t': self.release_t_normalizer,
            'attack_t': self.adsr_normalizer,
            'decay_t': self.adsr_normalizer,
            'sustain_t': self.adsr_normalizer,
        }

    def normalize(self, parameters_dict: dict):
        normalized_params_dict = {}
        for key, val in parameters_dict.items():
            operation = val['operation'] if isinstance(val['operation'], str) else val['operation'][0]
            params = val['params'] if 'params' in val else val['parameters']

            if operation in ["None", 'mix']:
                continue

            normalized_params_dict[key] = {'operation': operation, 'parameters': {}}
            for param_name, param_val in params.items():
                if ((operation in ['osc', 'saw_square_osc', 'osc_sine', 'osc_saw', 'osc_square',
                                   'osc_sine_no_activeness', 'osc_square_no_activeness', 'osc_saw_no_activeness']
                     and param_name in ['freq']) or (param_name == 'freq_c' and 'lfo' not in operation)):
                    normalized_params_dict[key]['parameters'][param_name] = \
                        self.oscillator_freq_log_normalizer.normalise(params[param_name])
                elif ((operation in ['osc_sine_no_activeness_cont_freq',
                                     'osc_square_no_activeness_cont_freq',
                                     'osc_saw_no_activeness_cont_freq']
                       and param_name in ['freq'])):
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
                elif operation in ['env_adsr', 'lowpass_filter_adsr'] and param_name in self.adsr_normalizers:
                    normalized_params_dict[key]['parameters'][param_name] = \
                        self.adsr_normalizers[param_name].normalise(params[param_name])
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
                if ((operation in ['osc', 'saw_square_osc', 'osc_saw', 'osc_square',
                                   'osc_sine_no_activeness', 'osc_square_no_activeness', 'osc_saw_no_activeness']
                     and param_name in ['freq']) or (param_name == 'freq_c' and 'lfo' not in operation)):
                    denormalized_params_dict[key]['parameters'][param_name] = \
                        self.oscillator_freq_log_normalizer.denormalise(params[param_name])
                elif ((operation in ['osc_sine_no_activeness_cont_freq',
                                     'osc_square_no_activeness_cont_freq',
                                     'osc_saw_no_activeness_cont_freq']
                       and param_name in ['freq'])):
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
                elif operation in ['env_adsr', 'lowpass_filter_adsr'] and param_name in self.adsr_normalizers:
                    denormalized_params_dict[key]['parameters'][param_name] = \
                        self.adsr_normalizers[param_name].denormalise(params[param_name])
                else:
                    denormalized_params_dict[key]['parameters'][param_name] = params[param_name]

            if operation in ['env_adsr', 'lowpass_filter_adsr'] and self.clamp_adsr:
                denormalized_params_dict[key]['parameters'] = \
                    self._clamp_adsr_params(denormalized_params_dict[key]['parameters'],
                                            operation,
                                            is_data_normalized=False)

        return denormalized_params_dict

    def post_process_inherent_constraints(self, model_output: dict):
        """
        Post-process the predicted parameters to satisfy the inherent constraints within the dataset.
        For example:
            - The attack, decay, and sustain timings summed up should not exceed 1 when normalized
            - Non-active modules should have zeroed out (or other default) parameters
        This function may be further implemented to more cases.
        Args:
            model_output (dict): The predicted parameters in the unit range.

        Returns:
            post_processed_model_output: The post-processed parameters.
        """
        model_output = self._force_default_parameters_if_non_active(model_output)
        post_processed_model_output = self._clamp_normalized_ads_params(model_output)
        return post_processed_model_output

    def _clamp_adsr_params(self, params: dict, operation: 'str', is_data_normalized=False):
        """
        Clamp the attack, decay, and sustain parameters to satisfy the inherent constraints within the dataset.
        Args:
            params (dict): The predicted parameters
            operation (str): The operation of the module
            is_data_normalized (bool): Whether the data is normalized (unit range) or not (full range in the dataset)
        """

        if is_data_normalized:
            time_limit = torch.tensor(1.0)
            sustain_level_max = torch.tensor(1.0)
        else:
            time_limit = self.note_off_time
            sustain_level_max = self.synth_structure.max_amp

        clamped_attack, clamped_decay, clamped_sustain = \
            self._clamp_ads_superposition(params['attack_t'],
                                          params['decay_t'],
                                          params['sustain_t'],
                                          time_limit=time_limit)

        clamped_sustain_level = torch.clamp(params['sustain_level'], min=0, max=sustain_level_max)

        ret_params = {}
        if operation == 'env_adsr':
            ret_params = {'attack_t': clamped_attack,
                          'decay_t': clamped_decay,
                          'sustain_t': clamped_sustain,
                          'sustain_level': clamped_sustain_level,
                          'release_t': params['release_t']}

        elif operation == 'lowpass_filter_adsr':
            ret_params = {'attack_t': clamped_attack,
                          'decay_t': clamped_decay,
                          'sustain_t': clamped_sustain,
                          'sustain_level': clamped_sustain_level,
                          'release_t': params['release_t'],
                          'intensity': params['intensity'],
                          'filter_freq': params['filter_freq']}

        return ret_params

    def _clamp_ads_superposition(self, attack_t: torch.Tensor, decay_t: torch.Tensor, sustain_t: torch.Tensor,
                                 time_limit: torch.Tensor):
        """This function clamps the superposition of attack decay sustain times, so it does not exceed time limit"""

        ads_length = attack_t + decay_t + sustain_t

        # Create a mask where ads_length exceeds the time limit
        exceed_mask = ads_length > time_limit
        normalization_values = torch.where(exceed_mask, ads_length + 1e-8, torch.ones_like(ads_length))

        # Normalize the attack, decay, and sustain values, but only change the exceeding ones.
        normalized_attack = torch.where(exceed_mask, attack_t * time_limit / normalization_values, attack_t)
        normalized_decay = torch.where(exceed_mask, decay_t * time_limit / normalization_values, decay_t)
        normalized_sustain = torch.where(exceed_mask, sustain_t * time_limit / normalization_values, sustain_t)

        return normalized_attack, normalized_decay, normalized_sustain

    def _clamp_normalized_ads_params(self, model_output: dict):
        """
        Normalize Attack, Decay, and Sustain timings, so they sum up to 1.

        Parameters:
        - model_output: Dictionary of the model output. The keys are the cell indexes of the modules,
         and the values are dictionaries of the operation and parameters.

        Returns:
        - attack_pred_normalized: Tensor of normalized Attack timings.
        - decay_pred_normalized: Tensor of normalized Decay timings.
        - sustain_pred_normalized: Tensor of normalized Sustain timings.
        """
        for key, val in model_output.items():
            operation = val['operation']

            if operation == "None":
                continue

            if operation in ['env_adsr', 'lowpass_filter_adsr'] and self.clamp_adsr:
                model_output[key]['parameters'] = \
                    self._clamp_adsr_params(model_output[key]['parameters'],
                                            operation,
                                            is_data_normalized=True)

        return model_output

    def _force_default_parameters_if_non_active(self, parameters_dict: dict):
        # currently not implemented

        # This commented code is for the case of constraining parameters in case of activeness parameter is present.
        # The idea is to preprocess the output such that module parameters with active=0
        # will be forced in some way to be 0. Without this, the model may learn that active=0 while the other parameters
        # are not 0, which doesn't make sense.
        # For example,for an oscillator if active=0, then the amplitude and frequency should be 0 as well. This shall
        # be inlined with the default values of the parameters in the case of active=0 when creating the dataset.
        #
        #     module_has_activeness_param = False
        #     if 'active' in synth_constants.modular_synth_params[operation]:
        #         module_has_activeness_param = True
        #         param_head = self.heads_module_dict[self.get_key(index, operation, 'active')]
        #         activeness_logits = param_head(latent)
        #         activeness_probs = self.softmax(activeness_logits)
        #         probability_active = activeness_probs[:, 1]
        #         output_dict[index]['parameters']['active'] = activeness_logits
        #
        #     for param in synth_constants.modular_synth_params[operation]:
        #         if param == 'active':
        #             continue
        #
        #         param_head = self.heads_module_dict[self.get_key(index, operation, param)]
        #         model_output = param_head(latent)
        #
        #         if param in ['waveform']:
        #             final_model_output = self.softmax(model_output)
        #         elif param not in ['active', 'fm_active', 'filter_type']:
        #             final_model_output = self.sigmoid(model_output)
        #         else:
        #             final_model_output = model_output
        #
        #         # todo: additional constraints for other sound modules with activeness parameter may be needed
        #         #  (for example if fm_active=False we shall do mod_index=0)
        #         if param in ['amp', 'freq'] and module_has_activeness_param:
        #             final_model_output = final_model_output * probability_active.unsqueeze(1)
        #
        #         output_dict[index]['parameters'][param] = final_model_output
        #
        return parameters_dict


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


class LogMinMaxNormaliser:
    """LogMinMaxNormaliser applies logarithmic min max normalisation to a tensor"""

    def __init__(self, target_min_val, target_max_val, original_min_val, original_max_val, clip=False, epsilon=1e-10):
        self.target_min_val = target_min_val
        self.target_max_val = target_max_val
        self.original_max_val = original_max_val
        self.original_min_val = original_min_val
        self.clip = clip

        # Ensure original_min_val is not zero or very close to zero.
        if self.original_min_val < epsilon:
            self.original_min_val += epsilon

        # Pre-compute the logarithmic values to make the transformations faster.
        self.log_original_min_val = torch.log(torch.tensor(original_min_val))
        self.log_original_max_val = torch.log(torch.tensor(original_max_val))

    def normalise(self, tensor, epsilon=1e-10):
        """
        From original frequency range to (0-1) using a logarithmic scale.
        """
        # Ensure that no values in the tensor are zero.
        tensor = torch.clamp(tensor, min=epsilon)

        log_tensor = torch.log(tensor)
        norm_tensor = (log_tensor - self.log_original_min_val) / (self.log_original_max_val - self.log_original_min_val)

        if self.clip:
            norm_tensor = torch.clamp(norm_tensor, 0, 1)

        return norm_tensor

    def denormalise(self, norm_tensor):
        """
        From (0-1) range to original frequency range using an exponential scale.
        """
        tensor = (norm_tensor - self.target_min_val) / (self.target_max_val - self.target_min_val)
        tensor = torch.exp(tensor * (self.log_original_max_val - self.log_original_min_val) + self.log_original_min_val)
        denormalized_values = torch.clamp(tensor, min=0, max=self.original_max_val)

        return denormalized_values


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
