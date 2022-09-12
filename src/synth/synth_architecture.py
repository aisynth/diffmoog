from typing import Tuple, Dict

import torch

from synth.synth_constants import SynthConstants
from synth.synth_modules import get_synth_module
from synth.synth_presets import synth_presets_dict

from utils.types import TensorLike


class SynthModularCell:

    def __init__(self,
                 index: tuple,
                 audio_input=None,
                 control_input=None,
                 operation=None,
                 parameters=None,
                 signal=None,
                 outputs=None,
                 switch_outputs=None,
                 active_prob=None,
                 default_connection=False,
                 synth_structure: SynthConstants = None,
                 device: str = 'cuda:0'):

        self.check_inputs(index, audio_input, outputs, switch_outputs, operation, parameters, synth_structure)

        self.index = index

        self.module = get_synth_module(operation, device, synth_structure)

        if default_connection:
            self.audio_input = None
            self.control_input = None
            self.outputs = None
            self.switch_outputs = None
        else:
            self.audio_input = audio_input
            self.control_input = control_input
            self.outputs = outputs
            self.switch_outputs = switch_outputs

        self.operation = operation
        self.parameters = parameters
        self.signal = signal
        self.active_prob = active_prob

    @staticmethod
    def check_inputs(index, audio_input, outputs, switch_outputs, operation, parameters,
                     synth_structure: SynthConstants):
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
        signal = self.module.process_sound(input_signal, modulator_signal, params, sample_rate, signal_duration,
                                           batch_size)

        if signal is not None and torch.any(torch.isnan(signal)):
            raise RuntimeError("Signal contains NaN")

        self.signal = signal


class SynthModular(torch.nn.Module):
    def __init__(self, preset_name: str, synth_structure: SynthConstants, device='cuda:0'):

        super().__init__()

        self.synth_structure = synth_structure
        self.sample_rate = synth_structure.sample_rate

        self.device = device

        preset, (n_channels, n_layers) = self._parse_preset(preset_name)

        self.n_channels = n_channels
        self.n_layers = n_layers
        self.synth_matrix = None
        self.apply_architecture(preset, n_channels, n_layers)

    def apply_architecture(self, preset: dict, n_channels: int, n_layers: int):
        self.synth_matrix = [[None for _ in range(n_layers)] for _ in range(n_channels)]
        for channel_idx in range(n_channels):
            for layer_idx in range(n_layers):
                cell = preset.get((channel_idx, layer_idx), {'index': (channel_idx, layer_idx)})
                self.synth_matrix[channel_idx][layer_idx] = SynthModularCell(**cell, device=self.device,
                                                                             synth_structure=self.synth_structure)

    def generate_signal(self, signal_duration: float, batch_size: int = 1) -> (TensorLike, Dict[str, TensorLike]):
        output_signals = {}
        for layer in range(self.n_layers):
            for channel in range(self.n_channels):
                cell = self.synth_matrix[channel][layer]
                audio_inputs, control_input = self._get_cell_inputs(cell)
                if audio_inputs is not None and len(audio_inputs) == 1:
                    audio_inputs = audio_inputs[0]

                cell.generate_signal(audio_inputs, control_input, cell.parameters, self.sample_rate,
                                     signal_duration, batch_size)

                output_signals[f"({cell.index[0]}, {cell.index[1]})"] = cell.signal

        final_signal = self.get_final_signal()

        return final_signal, output_signals

    def _get_cell_inputs(self, cell: SynthModularCell) -> (TensorLike, TensorLike):

        audio_inputs = []
        if cell.audio_input is not None:
            for input_cell_index in cell.audio_input:
                input_cell = self.synth_matrix[input_cell_index[0]][input_cell_index[1]]
                audio_inputs.append(input_cell.signal)
            audio_inputs = torch.stack(audio_inputs)
        else:
            audio_inputs = None

        if cell.control_input is not None:
            control_input_index = cell.control_input[0]
            control_input_cell = self.synth_matrix[control_input_index[0]][control_input_index[1]]
            control_input = control_input_cell.signal
        else:
            control_input = None

        return audio_inputs, control_input

    def update_cells_from_dict(self, params_dict: dict):
        for cell_index, cell_params in params_dict.items():
            self._update_cell_parameters(index=cell_index, parameters=cell_params['parameters'])

    def _update_cell_parameters(self, index: Tuple[int, int], parameters: dict):
        cell = self.synth_matrix[index[0]][index[1]]
        if parameters is not None and (isinstance(parameters, dict) or isinstance(parameters, list)):
            for key in parameters:
                if key not in ['output'] and key not in self.synth_structure.modular_synth_params[cell.operation]:
                    raise ValueError("Illegal parameter for the provided operation.")
            cell.parameters = parameters
        else:
            raise ValueError("Unsupported or empty parameters provided for cell {index} update.")

    def collect_params(self) -> dict:
        params_dict = {}
        for channel in self.synth_matrix:
            for cell in channel:
                operation = cell.operation if cell.operation is not None else 'None'
                parameters = cell.parameters if cell.parameters is not None else 'None'
                params_dict[cell.index] = {'operation': operation, 'parameters': parameters}
        return params_dict

    def get_final_signal(self) -> TensorLike:

        # Final signal summing from all channels in the last layer
        final_signal = 0
        num_active_channels = 0
        for channel in self.synth_matrix:
            signal = channel[-1].signal
            if signal is not None:
                final_signal = final_signal + signal
                num_active_channels += 1

        if num_active_channels > 0:
            final_signal = final_signal / num_active_channels
        else:
            final_signal = None

        return final_signal

    def reset_signal(self):
        for channel in self.synth_matrix:
            for cell in channel:
                cell.signal = 0

    @staticmethod
    def _parse_preset(preset_name: str) -> (dict, Tuple[int, int]):

        # Load preset and convert to dictionary of cell_index: cell_parameters
        preset_list = synth_presets_dict.get(preset_name, None)
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
