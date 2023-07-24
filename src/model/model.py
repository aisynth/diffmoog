from typing import Sequence

import numpy as np
import torch
from torch import nn
from torchvision.models import resnet18, resnet34
from synth.synth_chains import synth_chains_dict

from synth.synth_constants import synth_constants
from model.loss.spectral_loss_presets import loss_presets
from utils.train_utils import process_categorical_variable

LATENT_SPACE_SIZE = 128


class DecoderNetwork(nn.Module):
    def __init__(self, chain: str, device):
        self.chain = synth_chains_dict.get(chain, None)
        if self.chain is None:
            ValueError("Unknown self.cfg.CHAIN")

        super().__init__()

        self.device = device
        self.parameters_dict = nn.ModuleDict()

    def apply_params(self, init_params, batch_size=1):

        for cell in self.chain:

            index = cell.get('index')
            operation = cell.get('operation')

            if operation is None or index not in init_params:
                continue

            init_values = init_params[index]['parameters']
            op_params = synth_constants.modular_synth_params[operation]

            assert set(op_params) == set(init_values.keys()) - {'output'},\
                f"DecoderNetwork got mismatching parameter values in apply_params for operation {operation}: \n " \
                f"Expected: {set(op_params)}\n Received: {set(init_values.keys())}"

            for param_name in op_params:
                if param_name in ['waveform', 'filter_type']:
                    val_idx_dict = synth_constants.wave_type_dict if param_name == 'waveform' else \
                        synth_constants.filter_type_dict
                    init_val = process_categorical_variable(init_values[param_name], val_idx_dict, batch_size,
                                                            return_one_hot=True, factor=1000)
                elif param_name in ['active', 'fm_active']:
                    map = lambda x: 1.0 if x else -1.0
                    init_val = process_categorical_variable(init_values[param_name], map, batch_size,
                                                            return_one_hot=False)
                else:
                    init_val = init_values[param_name]

                if isinstance(init_val, torch.Tensor):
                    init_val = init_val.clone().detach().cpu().numpy()
                elif isinstance(init_val, list):
                    init_val = np.asarray(init_val)

                param_head = SimpleWeightLayer(torch.tensor(init_val, device=self.device, requires_grad=True),
                                               do_softmax=False, do_sigmoid=True)
                self.parameters_dict[self.get_key(index, operation, param_name)] = param_head

    def apply_params_partial(self, params_to_apply):

        for index, params in params_to_apply.items():
            operation = params['operation']
            param_vals = params['parameters']

            for param_name, param_val in param_vals.items():
                self.parameters_dict[self.get_key(index, operation, param_name)].update_val(param_val)

    def freeze_params(self, params_to_freeze: dict):
        for cell_index, cell_params in params_to_freeze.items():
            operation = cell_params['operation']
            parameters = cell_params['parameters']
            for param in parameters:
                key = self.get_key(cell_index, operation, param)
                layer = self.parameters_dict[key]
                param_val = self.parameters_dict[key].weight.data
                layer.freeze(param_val)

    def forward(self):

        output_dic = {}
        for cell in self.chain:

            index = cell.get('index')
            operation = cell.get('operation')

            if operation is None:
                continue

            op_params = synth_constants.modular_synth_params[operation]
            param_vals = {k: self.parameters_dict[self.get_key(index, operation, k)]() for k in op_params}
            output_dic[index] = {'operation': operation,
                                 'parameters': param_vals}

        return output_dic

    @staticmethod
    def get_key(index: tuple, operation: str, parameter: str) -> str:
        return f'{index}' + '_' + operation + '_' + parameter


class SynthNetwork(nn.Module):

    def __init__(self, cfg, synth_chain: str, loss_preset: str, device, backbone='resnet'):
        super().__init__()

        self.chain = synth_chains_dict.get(synth_chain, None)
        self.loss_preset = loss_presets[loss_preset]
        if cfg.synth.use_multi_spec_input == True:
            in_channels = len(self.loss_preset['fft_sizes'])
        else:
            in_channels = 1
        if self.chain is None:
            ValueError("Unknown self.cfg.CHAIN")

        self.device = device

        if backbone in ['lstm', 'gru']:
            self.backbone = RNNBackbone(backbone)
        elif backbone == 'resnet':
            # self.backbone = resnet18(weights=None)
            # todo: ask almog why weights is configured
            self.backbone = resnet18()
            self.backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            num_ftrs = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(num_ftrs, LATENT_SPACE_SIZE)

        self.heads_module_dict = nn.ModuleDict({})
        self.make_heads_from_chain()

        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    @staticmethod
    def get_key(index: tuple, operation: str, parameter: str) -> str:
        return f'{index}' + '_' + operation + '_' + parameter

    def make_heads_from_chain(self):
        for cell in self.chain:
            index = cell.get('index')
            operation = cell.get('operation')

            if operation in ['None', 'mix'] or operation is None:
                continue

            op_params = synth_constants.modular_synth_params[operation]
            for param in op_params:
                if param == 'waveform':
                    param_head = MLPBlock([LATENT_SPACE_SIZE, LATENT_SPACE_SIZE // 2, 10,
                                           len(synth_constants.wave_type_dict)])
                elif param == 'filter_type':
                    param_head = MLPBlock([LATENT_SPACE_SIZE, LATENT_SPACE_SIZE // 2, 10, 1])
                elif param in ['active', 'fm_active']:
                    param_head = MLPBlock([LATENT_SPACE_SIZE, LATENT_SPACE_SIZE // 2, 10, 1])
                else:
                    param_head = MLPBlock([LATENT_SPACE_SIZE, LATENT_SPACE_SIZE // 2, 10, 1])

                self.heads_module_dict[self.get_key(index, operation, param)] = param_head

    def forward(self, x):
        latent = self.backbone(x)

        # Apply different heads to predict each synth parameter
        output_dict = {}
        for cell in self.chain:
            index = cell.get('index')
            operation = cell.get('operation')

            if operation in ['None', 'mix'] or operation is None:
                continue

            output_dict[index] = {'operation': operation,
                                  'parameters': {}}

            for param in synth_constants.modular_synth_params[operation]:

                param_head = self.heads_module_dict[self.get_key(index, operation, param)]
                model_output = param_head(latent)

                if param in ['waveform']:
                    final_model_output = self.softmax(model_output)
                elif param not in ['active', 'fm_active', 'filter_type']:
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

        if isinstance(val, float):
            val = torch.tensor(val, device=device)

        if val.ndim == 1:
            val = val.unsqueeze(0)

        self.weight = nn.Parameter(val, requires_grad=False)
        self.do_softmax = False
        self.do_sigmoid = False

    def update_val(self, val):
        device = self.weight.device

        val = torch.tensor(val, device=device)

        if val.ndim == 1:
            val = val.unsqueeze(0)

        self.weight = nn.Parameter(val)


class RNNBackbone(nn.Module):

    def __init__(self, rnn_type: str = 'lstm', input_size: int = 128, hidden_size: int = 1024, output_size: int = LATENT_SPACE_SIZE,
                 agg_mean: bool = False):

        super().__init__()

        self.agg_mean = agg_mean
        self.rnn_type = rnn_type

        if rnn_type.lower() == 'lstm':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers=4, batch_first=True, bias=False)
        elif rnn_type.lower() == 'gru':
            self.conv1 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=7, stride=2)
            self.bn1 = nn.BatchNorm1d(num_features=64)

            self.conv2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=7, stride=2)
            self.bn2 = nn.BatchNorm1d(num_features=32)

            self.conv3 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=7, stride=2)
            self.bn3 = nn.BatchNorm1d(num_features=16)

            self.rnn = nn.GRU(input_size=16, hidden_size=hidden_size, num_layers=1, batch_first=True)
        else:
            raise ValueError(f"{rnn_type} RNN not supported")

        self.fc = nn.Sequential(nn.Linear(hidden_size, output_size, bias=False),
                                nn.ReLU())

    def forward(self, x):
        x = x.squeeze()

        if self.rnn_type == 'gru':
            x = self.bn1(self.conv1(x))
            x = self.bn2(self.conv2(x))
            x = self.bn3(self.conv3(x))
            x = x.transpose(1, 2)

        else:  # assuming the only other option is 'lstm'
            x = x.transpose(2, 1)

        output, hidden = self.rnn(x)

        if self.rnn_type == 'lstm':
            hidden = hidden[0]

        rnn_pred = output[:, -1, :]
        fc_output = self.fc(rnn_pred)

        return fc_output
