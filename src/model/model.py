from typing import Sequence

import torch
from torch import nn
from torchvision.models import resnet18, resnet34
from synth.synth_presets import synth_presets_dict

from synth.synth_constants import synth_structure

LATENT_SPACE_SIZE = 128


class DecoderOnlyNetwork(nn.Module):
    def __init__(self, preset: str, device):
        self.preset = synth_presets_dict.get(preset, None)
        if self.preset is None:
            ValueError("Unknown self.cfg.PRESET")

        super().__init__()

        self.device = device
        self.parameters_dict = nn.ModuleDict()

    def apply_params(self, init_params):

        for cell in self.preset:
            index = cell.get('index')
            operation = cell.get('operation')
            init_values = init_params[index]['parameters']
            if operation == 'osc':
                self.parameters_dict[self.get_key(index, operation, 'amp')] = \
                    SimpleWeightLayer(torch.tensor(init_values['amp'], dtype=torch.float, device=self.device,
                                                   requires_grad=True), do_sigmoid=True)
                self.parameters_dict[self.get_key(index, operation, 'freq')] = \
                    SimpleWeightLayer(torch.tensor(init_values['freq'], dtype=torch.float, device=self.device,
                                                   requires_grad=True), do_sigmoid=True)
                self.parameters_dict[self.get_key(index, operation, 'waveform')] = \
                    SimpleWeightLayer(torch.rand(len(synth_structure.wave_type_dict), device=self.device,
                                                 requires_grad=True), do_softmax=True)

            if operation == 'lfo':
                self.parameters_dict[self.get_key(index, operation, 'freq')] = \
                    SimpleWeightLayer(torch.tensor(init_values['freq'], dtype=torch.float, device=self.device,
                                                   requires_grad=True), do_sigmoid=True)

            elif operation == 'fm':
                for fm_param in ['amp_c', 'freq_c', 'mod_index']:
                    self.parameters_dict[self.get_key(index, operation, fm_param)] = \
                        SimpleWeightLayer(torch.tensor(init_values[fm_param], dtype=torch.float, device=self.device,
                                                       requires_grad=True), do_sigmoid=True)
                self.parameters_dict[self.get_key(index, operation, 'waveform')] = \
                    SimpleWeightLayer(torch.rand(len(self.synth_cfg.wave_type_dict), device=self.device,
                                                 requires_grad=True), do_softmax=True)

            elif operation == 'filter':
                self.parameters_dict[self.get_key(index, operation, 'filter_type')] = \
                    SimpleWeightLayer(torch.rand(len(self.synth_cfg.filter_type_dict), device=self.device,
                                                 requires_grad=True), do_softmax=True)
                self.parameters_dict[self.get_key(index, operation, 'filter_freq')] = \
                    SimpleWeightLayer(torch.tensor(init_values['filter_freq'], dtype=torch.float, device=self.device,
                                                   requires_grad=True), do_sigmoid=True)

            elif operation == 'env_adsr':
                for adsr_param in ['attack_t', 'decay_t', 'sustain_t', 'release_t', 'sustain_level']:
                    self.parameters_dict[self.get_key(index, operation, adsr_param)] = \
                        SimpleWeightLayer(torch.tensor(init_values[adsr_param], dtype=torch.float, device=self.device,
                                                       requires_grad=True), do_sigmoid=True)

    def freeze_params(self, params_to_freeze: dict):
        for cell_index, cell_params in params_to_freeze.items():
            for param_to_freeze, param_val in cell_params['parameters'].items():
                layer = self.parameters_dict[self.get_key(cell_index, cell_params['operation'], param_to_freeze)]
                layer.freeze(param_val)

    def forward(self):
        output_dic = {}
        for cell in self.preset:
            index = cell.get('index')
            operation = cell.get('operation')

            if operation == 'osc':
                op_params = ['amp', 'freq', 'waveform']
                param_vals = {k: self.parameters_dict[self.get_key(index, operation, k)]() for k in op_params}
                output_dic[index] = {'operation': operation,
                                     'parameters': param_vals}

            if operation == 'lfo':
                output_dic[index] = {'operation': operation,
                                     'params': {'freq': self.parameters_dict[self.get_key(index, operation, 'freq')]()
                                                }}

            elif operation == 'fm':
                op_params = ['amp_c', 'freq_c', 'mod_index', 'waveform']
                param_vals = {k: self.parameters_dict[self.get_key(index, operation, k)]() for k in op_params}
                output_dic[index] = {'operation': operation,
                                     'parameters': param_vals}

            elif operation == 'filter':
                op_params = ['filter_type', 'filter_freq']
                param_vals = {k: self.parameters_dict[self.get_key(index, operation, k)]() for k in op_params}
                output_dic[index] = {'operation': operation,
                                     'parameters': param_vals}

            elif operation == 'env_adsr':
                op_params = ['attack_t', 'decay_t', 'sustain_t', 'release_t', 'sustain_level']
                param_vals = {k: self.parameters_dict[self.get_key(index, operation, k)]() for k in op_params}
                output_dic[index] = {'operation': operation,
                                     'parameters': param_vals}

        return output_dic

    @staticmethod
    def get_key(index: tuple, operation: str, parameter: str) -> str:
        return f'{index}' + '_' + operation + '_' + parameter


class SynthNetwork(nn.Module):

    def __init__(self, preset: str, device, backbone='resnet'):
        super().__init__()

        self.preset = synth_presets_dict.get(preset, None)
        if self.preset is None:
            ValueError("Unknown self.cfg.PRESET")

        self.device = device

        if backbone in ['lstm', 'gru']:
            self.backbone = RNNBackbone(backbone)
        elif backbone == 'resnet':
            # self.backbone = resnet18(weights=None)
            #todo: ask almog why weights is configured
            self.backbone = resnet18()
            self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            num_ftrs = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(num_ftrs, LATENT_SPACE_SIZE)

        self.heads_module_dict = nn.ModuleDict({})
        self.make_heads_from_preset()

        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    @staticmethod
    def get_key(index: tuple, operation: str, parameter: str) -> str:
        return f'{index}' + '_' + operation + '_' + parameter

    def make_heads_from_preset(self):
        for cell in self.preset:
            index = cell.get('index')
            operation = cell.get('operation')

            if operation in ['None', 'mix'] or operation is None:
                continue

            op_params = synth_structure.modular_synth_params[operation]
            for param in op_params:
                if param == 'waveform':
                    param_head = MLPBlock([LATENT_SPACE_SIZE, LATENT_SPACE_SIZE // 2, 10,
                                           len(synth_structure.wave_type_dict)])
                elif param == 'filter_type':
                    param_head = MLPBlock([LATENT_SPACE_SIZE, LATENT_SPACE_SIZE // 2, 10,
                                           len(synth_structure.filter_type_dict)])
                elif param in ['active', 'fm_active']:
                    param_head = MLPBlock([LATENT_SPACE_SIZE, LATENT_SPACE_SIZE // 2, 10, 2])
                else:
                    param_head = MLPBlock([LATENT_SPACE_SIZE, LATENT_SPACE_SIZE // 2, 10, 1])

                self.heads_module_dict[self.get_key(index, operation, param)] = param_head

    def forward(self, x):
        latent = self.backbone(x)

        # Apply different heads to predict each synth parameter
        output_dict = {}
        for cell in self.preset:
            index = cell.get('index')
            operation = cell.get('operation')

            if operation in ['None', 'mix'] or operation is None:
                continue

            output_dict[index] = {'operation': operation,
                                  'parameters': {}}

            for param in synth_structure.modular_synth_params[operation]:

                param_head = self.heads_module_dict[self.get_key(index, operation, param)]
                model_output = param_head(latent)

                if param in ['waveform', 'filter_type']:
                    final_model_output = self.softmax(model_output)
                elif param not in ['active', 'fm_active']:
                    final_model_output = self.sigmoid(model_output)
                else:
                    final_model_output = model_output

                output_dict[index]['parameters'][param] = final_model_output

        return output_dict


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv_op = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

    def forward(self, x):
        return self.conv_op(x)


class MLPBlock(nn.Module):

    def __init__(self, layer_sizes: Sequence[int]):
        super(MLPBlock, self).__init__()

        layers = []
        for i in range(len(layer_sizes) - 1):
            linear = nn.Linear(layer_sizes[i], layer_sizes[i + 1], bias=False)
            relu = nn.ReLU()
            layers.extend([linear, relu])

        self.mlp = nn.Sequential(*layers[:-1])

    def forward(self, x):
        return self.mlp(x)


class SimpleWeightLayer(nn.Module):

    def __init__(self, w: torch.Tensor, requires_grad=True, do_sigmoid=False, do_softmax=False):
        super().__init__()

        self.do_softmax = do_softmax
        self.do_sigmoid = do_sigmoid

        if w.ndim == 1:
            w = w.unsqueeze(0)

        self.weight = nn.Parameter(w, requires_grad=requires_grad)

        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self):

        x = self.weight
        if self.do_sigmoid:
            x = self.sigmoid(x)
        if self.do_softmax:
            x = self.softmax(x)

        return x

    def freeze(self, val):

        device = self.weight.device

        val = torch.tensor(val, device=device)

        if val.ndim == 1:
            val = val.unsqueeze(0)

        self.weight = nn.Parameter(val, requires_grad=False)
        self.do_softmax = False
        self.do_sigmoid = False


class RNNBackbone(nn.Module):

    def __init__(self, rnn_type: str = 'lstm', input_size: int = 128, hidden_size: int = 128, output_size: int = 128,
                 agg_mean: bool = False):

        super().__init__()

        self.agg_mean = agg_mean
        self.rnn_type = rnn_type

        if rnn_type.lower() == 'lstm':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers=4, batch_first=True, bias=False)
        elif rnn_type.lower() == 'gru':
            self.rnn = nn.GRU(input_size, hidden_size, 1, batch_first=True)
        else:
            raise ValueError(f"{rnn_type} RNN not supported")

        self.fc = nn.Sequential(nn.Linear(hidden_size, output_size, bias=False),
                                nn.ReLU())

    def forward(self, x):

        x = torch.transpose(x.squeeze(), 2, 1)

        output, hidden = self.rnn(x)
        if self.rnn_type == 'lstm':
            hidden = hidden[0]

        rnn_pred = output[:, -1, :]
        fc_output = self.fc(rnn_pred)

        return fc_output
