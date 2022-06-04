from typing import Sequence

import torch
from torch import nn
from torchvision.models import resnet18, resnet34
from synth.synth_modular_presets import synth_presets_dict
from config import SynthConfig, Config

# todo: this is value from Valerio Tutorial. has to check
# LINEAR_IN_CHANNELS = 128 * 5 * 4
SMALL_LINEAR_IN_CHANNELS = 4480
# BIG_LINEAR_IN_CHANNELS = 17920
# BIG_LINEAR_IN_CHANNELS = 32256
# BIG_LINEAR_IN_CHANNELS = 13824
BIG_LINEAR_IN_CHANNELS = 23040
# BIG_LINEAR_IN_CHANNELS = 55296
# BIG_LINEAR_IN_CHANNELS = 60928
# LINEAR_IN_CHANNELS = 8064
LATENT_SPACE_SIZE = 128


class BigSynthNetwork(nn.Module):
    """
    CNN model to extract synth parameters from a batch of signals represented by mel-spectrograms

    :return: output_dic with classification and regression parameters.
                Concerning classification parameters:
                    'osc1_wave', 'osc2_wave' and 'filter freq' are returned as probability vectors.
                    size = (batch_size, #of_possibilities)

                    * Note:
                        In the synth module, in order to prevent if/else statements, the probability vectors will be
                        used to weight all possible wave and filter types in a superposition.
                        This operation is differential, while if/else statements are not


                    'osc1_freq' and 'osc2_freq' are returned as single values, by dot product.
                    size = (batch_size, 1)

                    The dot product is between learnt probabilities and fixed possible frequencies vector.
                    the learnt probabilities weight the possible frequencies.
                    * Note:
                        This technique allows the usage of a predicted frequency from a closed set of possible values in
                        the synth module, while preserving gradients computation.
                        Using argmax to predict a single value is not diffrentiable.
    """

    def __init__(self, synth_cfg: SynthConfig, device):
        super().__init__()
        self.device = device
        self.conv1 = ConvBlock(1, 64)
        self.conv2 = ConvBlock(64, 128)
        self.conv3 = ConvBlock(128, 256)
        self.conv4 = ConvBlock(256, 512)
        self.flatten = nn.Flatten()

        self.preset = synth_presets_dict.get(synth_cfg.preset, None)
        if self.preset is None:
            ValueError("Unknown self.cfg.PRESET")

        self.heads_module_dict = nn.ModuleDict({})
        for cell in self.preset:
            index = cell.get('index')
            operation = cell.get('operation')
            if operation == 'osc':
                amplitude_head = MLPBlock([BIG_LINEAR_IN_CHANNELS, LATENT_SPACE_SIZE, 1])
                frequency_head = MLPBlock([BIG_LINEAR_IN_CHANNELS, LATENT_SPACE_SIZE, 1])
                waveform_head = MLPBlock([BIG_LINEAR_IN_CHANNELS, LATENT_SPACE_SIZE, len(synth_cfg.wave_type_dict)])
                self.heads_module_dict[self.get_key(index, operation, 'amp')] = amplitude_head
                self.heads_module_dict[self.get_key(index, operation, 'freq')] = frequency_head
                self.heads_module_dict[self.get_key(index, operation, 'waveform')] = waveform_head

            if operation == 'lfo':
                amplitude_head = MLPBlock([BIG_LINEAR_IN_CHANNELS, LATENT_SPACE_SIZE, 1])
                frequency_head = MLPBlock([BIG_LINEAR_IN_CHANNELS, LATENT_SPACE_SIZE, 1])
                self.heads_module_dict[self.get_key(index, operation, 'amp')] = amplitude_head
                self.heads_module_dict[self.get_key(index, operation, 'freq')] = frequency_head

            elif operation == 'fm':
                carrier_amplitude_head = MLPBlock([BIG_LINEAR_IN_CHANNELS, LATENT_SPACE_SIZE, 1])
                carrier_frequency_head = MLPBlock([BIG_LINEAR_IN_CHANNELS, LATENT_SPACE_SIZE, 1])
                carrier_waveform_head = MLPBlock([BIG_LINEAR_IN_CHANNELS, LATENT_SPACE_SIZE,
                                                  len(synth_cfg.wave_type_dict)])
                modulation_index_head = MLPBlock([BIG_LINEAR_IN_CHANNELS, LATENT_SPACE_SIZE, 1])
                self.heads_module_dict[self.get_key(index, operation, 'amp_c')] = carrier_amplitude_head
                self.heads_module_dict[self.get_key(index, operation, 'freq_c')] = carrier_frequency_head
                self.heads_module_dict[self.get_key(index, operation, 'waveform')] = carrier_waveform_head
                self.heads_module_dict[self.get_key(index, operation, 'mod_index')] = modulation_index_head

            elif operation == 'filter':
                filter_type_head = MLPBlock([BIG_LINEAR_IN_CHANNELS, LATENT_SPACE_SIZE, len(synth_cfg.wave_type_dict)])
                filter_freq_head = MLPBlock([BIG_LINEAR_IN_CHANNELS, LATENT_SPACE_SIZE, 1])
                self.heads_module_dict[self.get_key(index, operation, 'filter_type')] = filter_type_head
                self.heads_module_dict[self.get_key(index, operation, 'filter_freq')] = filter_freq_head

            elif operation == 'env_adsr':
                attack_t_head = MLPBlock([BIG_LINEAR_IN_CHANNELS, LATENT_SPACE_SIZE, 1])
                decay_t_head = MLPBlock([BIG_LINEAR_IN_CHANNELS, LATENT_SPACE_SIZE, 1])
                sustain_t_head = MLPBlock([BIG_LINEAR_IN_CHANNELS, LATENT_SPACE_SIZE, 1])
                sustain_level_head = MLPBlock([BIG_LINEAR_IN_CHANNELS, LATENT_SPACE_SIZE, 1])
                release_t_head = MLPBlock([BIG_LINEAR_IN_CHANNELS, LATENT_SPACE_SIZE, 1])
                self.heads_module_dict[self.get_key(index, operation, 'attack_t')] = attack_t_head
                self.heads_module_dict[self.get_key(index, operation, 'decay_t')] = decay_t_head
                self.heads_module_dict[self.get_key(index, operation, 'sustain_t')] = sustain_t_head
                self.heads_module_dict[self.get_key(index, operation, 'sustain_level')] = sustain_level_head
                self.heads_module_dict[self.get_key(index, operation, 'release_t')] = release_t_head

        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

        # Initialization
        # nn.init.kaiming_normal_(self.conv1[0].weight, mode='fan_in', nonlinearity='relu')
        # nn.init.kaiming_normal_(self.conv2[0].weight, mode='fan_in', nonlinearity='relu')
        # nn.init.kaiming_normal_(self.conv3[0].weight, mode='fan_in', nonlinearity='relu')
        # nn.init.kaiming_normal_(self.conv4[0].weight, mode='fan_in', nonlinearity='relu')

    @staticmethod
    def get_key(index: tuple, operation: str, parameter: str) -> str:
        return f'{index}' + '_' + operation + '_' + parameter

    def forward(self, input_data):
        """
        :param input_data: Spectrograms batch
        :return: 1st argument: output_dic - dictionary containing all the model predictions
                 2st argument: logits - frequency logits prediction when self.cfg.synth_type == 'OSC_ONLY'
                                None - when self.cfg.synth_type != 'OSC_ONLY'
        """
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        latent = self.flatten(x)

        # Apply different heads to predict each synth parameter
        output_dic = {}
        for cell in self.preset:
            index = cell.get('index')
            operation = cell.get('operation')

            if operation == 'osc':
                amplitude_head = self.heads_module_dict[self.get_key(index, operation, 'amp')]
                frequency_head = self.heads_module_dict[self.get_key(index, operation, 'freq')]
                waveform_head = self.heads_module_dict[self.get_key(index, operation, 'waveform')]

                predicted_amplitude = amplitude_head(latent)
                predicted_amplitude = self.sigmoid(predicted_amplitude)
                predicted_frequency = frequency_head(latent)
                predicted_frequency = self.sigmoid(predicted_frequency)
                waveform_logits = waveform_head(latent)
                waveform_probabilities = self.softmax(waveform_logits)

                output_dic[index] = {'operation': operation,
                                     'params': {'amp': predicted_amplitude,
                                                'freq': predicted_frequency,
                                                'waveform': waveform_probabilities
                                                }}
            if operation == 'lfo':
                amplitude_head = self.heads_module_dict[self.get_key(index, operation, 'amp')]
                frequency_head = self.heads_module_dict[self.get_key(index, operation, 'freq')]

                predicted_amplitude = amplitude_head(latent)
                predicted_amplitude = self.sigmoid(predicted_amplitude)
                predicted_frequency = frequency_head(latent)
                predicted_frequency = self.sigmoid(predicted_frequency)

                output_dic[index] = {'operation': operation,
                                     'params': {'amp': predicted_amplitude,
                                                'freq': predicted_frequency
                                                }}
            elif operation == 'fm':
                carrier_amplitude_head = self.heads_module_dict[self.get_key(index, operation, 'amp_c')]
                carrier_frequency_head = self.heads_module_dict[self.get_key(index, operation, 'freq_c')]
                waveform_head = self.heads_module_dict[self.get_key(index, operation, 'waveform')]
                mod_index_head = self.heads_module_dict[self.get_key(index, operation, 'mod_index')]

                predicted_carrier_amplitude = carrier_amplitude_head(latent)
                predicted_carrier_amplitude = self.sigmoid(predicted_carrier_amplitude)
                predicted_carrier_frequency = carrier_frequency_head(latent)
                predicted_carrier_frequency = self.sigmoid(predicted_carrier_frequency)
                waveform_logits = waveform_head(latent)
                waveform_probabilities = self.softmax(waveform_logits)
                predicted_mod_index = mod_index_head(latent)
                predicted_mod_index = self.sigmoid(predicted_mod_index)

                output_dic[index] = {'operation': operation,
                                     'params': {'amp_c': predicted_carrier_amplitude,
                                                'freq_c': predicted_carrier_frequency,
                                                'waveform': waveform_probabilities,
                                                'mod_index': predicted_mod_index
                                                }}

            elif operation == 'filter':
                filter_type_head = self.heads_module_dict[self.get_key(index, operation, 'filter_type')]
                filter_freq_head = self.heads_module_dict[self.get_key(index, operation, 'filter_freq')]

                filter_type_logits = filter_type_head(latent)
                filter_type_probabilities = self.softmax(filter_type_logits)
                predicted_filter_freq = filter_freq_head(latent)
                predicted_filter_freq = self.sigmoid(predicted_filter_freq)

                output_dic[index] = {'operation': operation,
                                     'params': {'filter_type': filter_type_probabilities,
                                                'filter_freq': predicted_filter_freq
                                                }}

            elif operation == 'env_adsr':
                attack_t_head = self.heads_module_dict[self.get_key(index, operation, 'attack_t')]
                decay_t_head = self.heads_module_dict[self.get_key(index, operation, 'decay_t')]
                sustain_t_head = self.heads_module_dict[self.get_key(index, operation, 'sustain_t')]
                sustain_level_head = self.heads_module_dict[self.get_key(index, operation, 'sustain_level')]
                release_t_head = self.heads_module_dict[self.get_key(index, operation, 'release_t')]

                predicted_attack_t = attack_t_head(latent)
                predicted_attack_t = self.sigmoid(predicted_attack_t)

                predicted_decay_t = decay_t_head(latent)
                predicted_decay_t = self.sigmoid(predicted_decay_t)

                predicted_sustain_t = sustain_t_head(latent)
                predicted_sustain_t = self.sigmoid(predicted_sustain_t)

                predicted_sustain_level = sustain_level_head(latent)
                predicted_sustain_level = self.sigmoid(predicted_sustain_level)

                predicted_release_t = release_t_head(latent)
                predicted_release_t = self.sigmoid(predicted_release_t)

                output_dic[index] = {'operation': operation,
                                     'parameters': {'attack_t': predicted_attack_t,
                                                    'decay_t': predicted_decay_t,
                                                    'sustain_t': predicted_sustain_t,
                                                    'sustain_level': predicted_sustain_level,
                                                    'release_t': predicted_release_t
                                                    }}

        return output_dic


class DecoderOnlyNetwork(nn.Module):
    def __init__(self, synth_cfg: SynthConfig, device):
        self.preset = synth_presets_dict.get(synth_cfg.preset, None)
        if self.preset is None:
            ValueError("Unknown self.cfg.PRESET")

        super().__init__()

        self.synth_cfg = synth_cfg
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
                    SimpleWeightLayer(torch.rand(len(self.synth_cfg.wave_type_dict), device=self.device,
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


class SimpleSynthNetwork(nn.Module):

    def __init__(self, preset: str, synth_cfg: SynthConfig, cfg: Config, device, backbone='resnet'):
        super().__init__()

        self.cfg = cfg
        self.preset = synth_presets_dict.get(preset, None)
        if self.preset is None:
            ValueError("Unknown self.cfg.PRESET")

        self.device = device

        if backbone in ['lstm', 'gru']:
            self.backbone = RNNBackbone(backbone)
        elif backbone == 'resnet':
            self.backbone = resnet18(pretrained=False)
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

            op_params = SynthConfig.modular_synth_params[operation]
            for param in op_params:
                if param == 'waveform':
                    param_head = MLPBlock([LATENT_SPACE_SIZE, LATENT_SPACE_SIZE // 2, 10,
                                           len(SynthConfig.wave_type_dict)])
                elif param == 'filter_type':
                    param_head = MLPBlock([LATENT_SPACE_SIZE, LATENT_SPACE_SIZE // 2, 10,
                                           len(SynthConfig.filter_type_dict)])
                elif param == 'envelope':
                    param_head = MLPBlock([LATENT_SPACE_SIZE, LATENT_SPACE_SIZE * 4, LATENT_SPACE_SIZE * 8,
                                           self.cfg.sample_rate])
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

            for param in SynthConfig.modular_synth_params[operation]:

                param_head = self.heads_module_dict[self.get_key(index, operation, param)]
                model_output = param_head(latent)

                if param in ['waveform', 'filter_type']:
                    model_output = self.softmax(model_output)
                else:
                    model_output = self.sigmoid(model_output)

                output_dict[index]['parameters'][param] = model_output

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
        layers = [nn.Linear(layer_sizes[i], layer_sizes[i + 1], bias=False) for i in range(len(layer_sizes) - 1)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


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
