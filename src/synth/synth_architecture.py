from typing import Tuple

import matplotlib.pyplot as plt
import numpy
from synth import synth_modular_presets
import random
# import simpleaudio as sa
import numpy as np
import torch
from torch import nn
from model import helper
from synth.synth_constants import SynthConstants
from synth.synth_modules import SynthModules, make_envelope_shape, get_synth_module

from model.gumble_softmax import gumbel_softmax


class SynthModularCell:

    def __init__(self,
                 index: tuple,
                 audio_input=None,
                 control_input=None,
                 operation=None,
                 parameters=None,
                 signal=None,
                 outputs=None,
                 default_connection=False,
                 synth_structure: SynthConstants = None,
                 device: str = 'cuda:0'):

        self.check_inputs(index, audio_input, outputs, operation, parameters, synth_structure)

        self.index = index

        self.module = get_synth_module(operation, device, synth_structure)

        if default_connection:
            self.audio_input = None
            self.control_input = None
            self.outputs = None

        else:
            self.audio_input = audio_input
            self.control_input = control_input
            self.outputs = outputs

        self.operation = operation
        self.parameters = parameters
        self.signal = signal

    @staticmethod
    def check_inputs(index, audio_input, outputs, operation, parameters, synth_structure: SynthConstants):
        channel = index[0]
        layer = index[1]

        if audio_input is not None:
            if type(audio_input) is not list:
                ValueError("Illegal input_list")
            for input_ in audio_input:
                if len(input_) != 2:
                    ValueError("Illegal input index")
                input_layer = input_[1]

                if input_layer >= layer:
                    ValueError("Illegal input chain")

        if outputs is not None:
            if type(outputs) is not list:
                ValueError("Illegal input_list - not a list")
            for output_ in outputs:
                if len(outputs) != 2:
                    ValueError("Illegal output index")
                output_layer = output_[1]

                if output_layer <= layer:
                    ValueError("Illegal output chain. Output must be chained to a layer > cell.layer")

        if operation is not None:
            if operation not in synth_structure.modular_synth_operations:
                ValueError("Illegal operation")

            if parameters is not None:
                for key in parameters:
                    if key not in synth_structure.modular_synth_params[operation]:
                        ValueError("Illegal parameter for the provided operation")

    def generate_signal(self, input_signal, modulator_signal, params, sample_rate, signal_duration, batch_size):
        self.signal = self.module.generate_sound(input_signal, modulator_signal, params, sample_rate, signal_duration,
                                                 batch_size)


class SynthModular:
    def __init__(self, preset_name: str,
                 sample_rate=44100,
                 signal_duration_sec=1.0,
                 device='cuda:0'):

        self.synth_constants = SynthConstants()

        self.sample_rate = sample_rate
        self.signal_duration_sec = signal_duration_sec

        self.device = device

        preset, (n_channels, n_layers) = self.parse_preset(preset_name)

        self.synth_matrix = None
        self.apply_architecture(preset, n_channels, n_layers)

    def apply_architecture(self, preset: dict, n_channels: int, n_layers: int):
        for c in range(n_channels):
            for l in range(n_layers):
                cell = preset.get((c, l), {})
                self.synth_matrix[c][l] = SynthModularCell(**cell, device=self.device,
                                                           synth_structure=self.synth_constants)

    def generate_signal(self, batch_size=1):
        synth_module = SynthModules(num_sounds=1,
                                    sample_rate=self.sample_rate,
                                    signal_duration_sec=self.signal_duration_sec,
                                    device=self.device)

        output_signals = {}
        for channel in self.synth_matrix:
            for cell in channel:
                operation = cell.operation

                if operation is None:
                    if cell.audio_input is None:
                        signal = None
                    elif len(cell.audio_input) == 1:
                        input_cell_index = cell.audio_input[0]
                        input_cell = self.synth_matrix[input_cell_index[0]][input_cell_index[1]]
                        signal = input_cell.signal
                    else:
                        signal = None
                        AttributeError("Illegal cell input for operation==None")
                    cell.signal = signal

                cell.generate_signal(signal, modulator_signal, cell.parameters, self.sample_rate,
                                     self.signal_duration_sec, batch_size)

                elif operation == 'osc':
                    cell.signal = synth_module.oscillator(amp=cell.parameters['amp'],
                                                          freq=cell.parameters['freq'],
                                                          phase=0,
                                                          waveform=cell.parameters['waveform'],
                                                          num_sounds=batch_size)
                    # cell.signal = synth_module.batch_oscillator(amp=1,
                    #                                             freq=440,
                    #                                             phase=0,
                    #                                             waveform='square')
                elif operation in ['lfo', 'lfo_non_sine']:
                    cell.signal = synth_module.batch_oscillator(amp=1.0,
                                                                freq=cell.parameters['freq'],
                                                                phase=0,
                                                                waveform=cell.parameters['waveform'])
                elif operation == 'lfo_sine':
                    active_signal = self._process_active_signal(cell.parameters['active']).to(self.device)

                    #todo: standartize input
                    if torch.is_tensor(cell.parameters['freq']):
                        activated_freq = active_signal.squeeze() * cell.parameters['freq'].squeeze()
                    else:
                        activated_freq = active_signal.squeeze() * torch.tensor(cell.parameters['freq'],
                                                                              device=self.device).squeeze()

                    cell.signal = synth_module.batch_oscillator(amp=active_signal.float().squeeze(),
                                                                freq=activated_freq,
                                                                phase=0,
                                                                waveform='sine')

                elif operation in ['fm', 'fm_lfo', 'fm_sine', 'fm_square', 'fm_saw']:
                    fm_active_signal = self._process_active_signal(cell.parameters['fm_active']).to(self.device)
                    active_signal = self._process_active_signal(cell.parameters['active']).to(self.device)

                    # todo: standartize input
                    if torch.is_tensor(cell.parameters['freq_c']):
                        activated_freq_c = active_signal.squeeze() * cell.parameters['freq_c'].squeeze()
                    else:
                        activated_freq_c = active_signal.squeeze() * torch.tensor(cell.parameters['freq_c'],
                                                                              device=self.device).squeeze()

                    if torch.is_tensor(cell.parameters['mod_index']):
                        activated_mod_index = fm_active_signal.squeeze() * cell.parameters['mod_index'].squeeze()
                    else:
                        activated_mod_index = fm_active_signal.squeeze() * torch.tensor(cell.parameters['mod_index'],
                                                                              device=self.device).squeeze()
                    ##############

                    control_input_cell_index = cell.control_input[0]
                    control_input_cell = self.synth_matrix[control_input_cell_index[0]][control_input_cell_index[1]]
                    control_input_cell_signal = control_input_cell.signal
                    modulator = fm_active_signal * control_input_cell_signal

                    if operation in ['fm', 'fm_lfo']:
                        cell.signal = synth_module.batch_oscillator_fm(amp_c=1.0,
                                                                       freq_c=activated_freq_c,
                                                                       waveform=cell.parameters['waveform'],
                                                                       mod_index=activated_mod_index,
                                                                       modulator=modulator)
                    else:
                        if operation == 'fm_sine':
                            waveform = 'sine'
                        elif operation == 'fm_square':
                            waveform = 'square'
                        elif operation == 'fm_saw':
                            waveform = 'sawtooth'
                        else:
                            raise ValueError("Unsupported waveform")
                        cell.signal = \
                            synth_module.batch_specific_waveform_oscillator_fm(amp_c=cell.parameters['amp_c'],
                                                                               freq_c=activated_freq_c,
                                                                               waveform=waveform,
                                                                               mod_index=activated_mod_index,
                                                                               modulator=modulator)

                elif operation == 'mix':
                    signal = 0
                    num_inputs = len(cell.audio_input)
                    for input_num in range(num_inputs):
                        input_index = cell.audio_input[input_num]
                        input_sound = self.synth_matrix[input_index[0]][input_index[1]].signal
                        signal += input_sound

                    cell.signal = signal / num_inputs

                elif operation == 'amplitude_shape':
                    if len(cell.input_list) == 1:
                        input_cell_index = cell.input_list[0]
                        input_cell = self.synth_matrix[input_cell_index[0]][input_cell_index[1]]
                        input_signal = input_cell.signal
                    else:
                        input_signal = 0
                        AttributeError("Illegal cell input")

                    attack_t = cell.parameters['attack_t']
                    decay_t = cell.parameters['decay_t']
                    sustain_t = cell.parameters['sustain_t']
                    sustain_level = cell.parameters['sustain_level']
                    release_t = cell.parameters['release_t']
                    envelope = cell.parameters['envelope']

                    if envelope.dim() == 1:
                        compute_envelope = True
                    else:
                        compute_envelope = False

                    if compute_envelope:
                        envelope_shape = make_envelope_shape(attack_t,
                                                             decay_t,
                                                             sustain_t,
                                                             sustain_level,
                                                             release_t,
                                                             self.signal_duration_sec,
                                                             self.sample_rate,
                                                             self.device,
                                                             batch_size)
                    else:
                        envelope_shape = envelope

                    # envelope_shape = cell.parameters['envelope']
                    # plt.plot(envelope_shape[1].detach().numpy())
                    # plt.show()
                    cell.signal = synth_module.amplitude_envelope(input_signal, envelope_shape)

                elif operation == 'filter_shape':
                    if len(cell.input_list) == 1:
                        input_cell_index = cell.input_list[0]
                        input_cell = self.synth_matrix[input_cell_index[0]][input_cell_index[1]]
                        input_signal = input_cell.signal
                    else:
                        input_signal = 0
                        AttributeError("Illegal cell input")
                    # todo: change line below to wanted behavior
                    # envelope_shape = torch.linspace(1, 0, 16000).to(self.device)
                    envelope_shape = torch.ones(16000).to(self.device)
                    cell.signal = synth_module.filter_envelope(input_signal, envelope_shape)

                elif operation == 'filter':
                    if len(cell.input_list) == 1:
                        input_cell_index = cell.input_list[0]
                        input_cell = self.synth_matrix[input_cell_index[0]][input_cell_index[1]]
                        input_signal = input_cell.signal
                    else:
                        input_signal = 0
                        AttributeError("Illegal cell input")
                    cell.signal = synth_module.batch_filter(input_signal,
                                                            filter_freq=cell.parameters['filter_freq'],
                                                            filter_type=cell.parameters['filter_type'])

                elif operation == 'lowpass_filter':
                    if len(cell.audio_input) == 1:
                        input_cell_index = cell.audio_input[0]
                        input_cell = self.synth_matrix[input_cell_index[0]][input_cell_index[1]]
                        input_signal = input_cell.signal
                    else:
                        input_signal = 0
                        AttributeError("Illegal cell input")
                    cell.signal = synth_module.lowpass_batch_filter(input_signal,
                                                                    filter_freq=cell.parameters['filter_freq'],
                                                                    resonance=cell.parameters['resonance'])

                elif operation == 'env_adsr':
                    if len(cell.audio_input) == 1:
                        input_cell_index = cell.audio_input[0]
                        input_cell = self.synth_matrix[input_cell_index[0]][input_cell_index[1]]
                        input_signal = input_cell.signal
                    else:
                        input_signal = 0
                        AttributeError("Illegal cell input")

                    # cell.signal = synth_module.adsr_envelope(input_signal,
                    #                                          attack_t=cell.parameters['attack_t'],
                    #                                          decay_t=cell.parameters['decay_t'],
                    #                                          sustain_t=cell.parameters['sustain_t'],
                    #                                          sustain_level=cell.parameters['sustain_level'],
                    #                                          release_t=cell.parameters['release_t'],
                    #                                          num_sounds=num_sounds_)
                    cell.signal = synth_module.batch_adsr_envelope(input_signal,
                                                                   attack_t=cell.parameters['attack_t'],
                                                                   decay_t=cell.parameters['decay_t'],
                                                                   sustain_t=cell.parameters['sustain_t'],
                                                                   sustain_level=cell.parameters['sustain_level'],
                                                                   release_t=cell.parameters['release_t'])

                elif operation == 'tremolo':
                    if len(cell.audio_input) == 1:
                        input_cell_index = cell.audio_input[0]
                        input_cell = self.synth_matrix[input_cell_index[0]][input_cell_index[1]]
                        input_signal = input_cell.signal
                    else:
                        input_signal = 0
                        AttributeError("Illegal cell input")

                    if len(cell.control_input) == 1:
                        control_input_cell_index = cell.control_input[0]
                        control_input_cell = self.synth_matrix[control_input_cell_index[0]][control_input_cell_index[1]]
                        control_input_cell_signal = control_input_cell.signal
                        active_signal = self._process_active_signal(cell.parameters['active']).to(self.device)
                        modulator = active_signal * control_input_cell_signal
                    else:
                        modulator = 0
                        AttributeError("Illegal cell control input")

                    cell.signal = synth_module.tremolo_by_modulator_signal(input_signal=input_signal,
                                                                           modulator_signal=modulator,
                                                                           amount=cell.parameters['amount'].squeeze())

                output_signals[f"({channel}, {layer})"] = cell.signal

        # Final signal summing from all channels in the last layer
        final_signal = torch.zeros((int(self.sample_rate * self.signal_duration_sec)),
                                   requires_grad=True,
                                   device=self.device)
        num_active_channels = 0
        for channel in range(self.num_channels):
            signal = self.synth_matrix[channel][self.num_layers - 1].signal
            if signal is not None:
                final_signal = final_signal + signal
                num_active_channels += 1
        if num_active_channels > 0:
            final_signal = final_signal / num_active_channels
        else:
            final_signal = None

        return final_signal, output_signals

    def _process_active_signal(self, active_vector):

        if not isinstance(active_vector, torch.Tensor):
            active_vector = torch.tensor(active_vector)

        if active_vector.dtype == torch.bool:
            ret_active_vector = active_vector.long().unsqueeze(1)
            return ret_active_vector

        active_vector_gumble = gumbel_softmax(active_vector, hard=True, device=self.device)
        ret_active_vector = active_vector_gumble[:, :1]
        return ret_active_vector

    def update_cells_from_dict(self, params_dict):
        for cell_index, cell_params in params_dict.items():
            self.update_cell_parameters(index=cell_index, parameters=cell_params['parameters'])

    def update_cell_parameters(self, index: tuple, parameters: dict):
        cell = self.synth_matrix[index[0]][index[1]]
        if parameters is not None and isinstance(parameters, dict):
            for key in parameters:
                if key not in self.synth_constants.modular_synth_params[cell.operation]:
                    raise ValueError("Illegal parameter for the provided operation.")
            cell.parameters = parameters
        else:
            raise ValueError("Unsupported or empty parameters provided for cell {index} update.")

    @staticmethod
    def parse_preset(preset_name: str) -> (dict, Tuple[int, int]):

        # Load preset and convert to dictionary of cell_index: cell_parameters
        preset_list = synth_modular_presets.synth_presets_dict.get(preset_name, None)
        if preset_list is None:
            raise ValueError("Unknown PRESET")

        preset_dict = {}
        n_layers, n_channels = 0, 0
        for cell_desc in preset_list:
            cell_idx = cell_desc['index']
            preset_dict[cell_idx] = cell_desc

            # Deduce preset channel and layer numbers
            n_channels = max(n_channels, cell_idx[0] + 1)
            n_layers = max(n_layers, cell_idx[1] + 1)

        return preset_dict, (n_channels, n_layers)

    def collect_params(self):
        params_dict = {}
        for channel in self.synth_matrix:
            for cell in channel:
                if cell.operation is not None:
                    operation = cell.operation
                else:
                    operation = 'None'
                if cell.parameters is not None:
                    parameters = cell.parameters
                else:
                    parameters = 'None'
                params_dict[cell.index] = {'operation': operation, 'params': parameters}
        return params_dict

    def reset_signal(self):
        for channel in self.synth_matrix:
            for cell in channel:
                cell.signal = torch.zeros((1, int(self.sample_rate * self.signal_duration_sec)), requires_grad=True)

