from torch import nn
import torch
import helper
from torchsummary import summary
from dataclasses import dataclass
from synth import synth_modular_presets
from config import SynthConfig

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
HIDDEN_IN_CHANNELS = 128


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

    def __init__(self, synth_cfg: SynthConfig):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.flatten = nn.Flatten()

        self.module_dict = {}
        if synth_cfg.preset == 'BASIC_FLOW':
            self.preset = synth_modular_presets.BASIC_FLOW
        elif synth_cfg.preset == 'FM':
            self.preset = synth_modular_presets.FM
        else:
            ValueError("Unknown self.cfg.PRESET")

        for cell in self.preset:
            index = cell.index
            operation = cell.operation

            if operation == 'osc':
                amplitude_head = nn.Sequential(
                    nn.Linear(BIG_LINEAR_IN_CHANNELS, HIDDEN_IN_CHANNELS),
                    nn.Linear(HIDDEN_IN_CHANNELS, 1)
                )
                frequency_head = nn.Sequential(
                    nn.Linear(BIG_LINEAR_IN_CHANNELS, HIDDEN_IN_CHANNELS),
                    nn.Linear(HIDDEN_IN_CHANNELS, 1)
                )
                waveform_head = nn.Sequential(
                    nn.Linear(BIG_LINEAR_IN_CHANNELS, HIDDEN_IN_CHANNELS),
                    nn.Linear(HIDDEN_IN_CHANNELS, len(synth_cfg.wave_type_dict))
                )
                self.module_dict[index] = {'operation': operation,
                                           'module_dict': nn.ModuleDict({
                                               'amp': amplitude_head,
                                               'freq': frequency_head,
                                               'waveform': waveform_head}).to(device=helper.get_device())}
            elif operation == 'fm':
                carrier_amplitude_head = nn.Sequential(
                    nn.Linear(BIG_LINEAR_IN_CHANNELS, HIDDEN_IN_CHANNELS),
                    nn.Linear(HIDDEN_IN_CHANNELS, 1)
                )
                carrier_frequency_head = nn.Sequential(
                    nn.Linear(BIG_LINEAR_IN_CHANNELS, HIDDEN_IN_CHANNELS),
                    nn.Linear(HIDDEN_IN_CHANNELS, 1)
                )
                carrier_waveform_head = nn.Sequential(
                    nn.Linear(BIG_LINEAR_IN_CHANNELS, HIDDEN_IN_CHANNELS),
                    nn.Linear(HIDDEN_IN_CHANNELS, len(synth_cfg.wave_type_dict))
                )
                modulation_index_head = nn.Sequential(
                    nn.Linear(BIG_LINEAR_IN_CHANNELS, HIDDEN_IN_CHANNELS),
                    nn.Linear(HIDDEN_IN_CHANNELS, 1)
                )
                self.module_dict[index] = {'operation': operation,
                                           'module_dict': nn.ModuleDict({
                                               'amp_c': carrier_amplitude_head,
                                               'freq_c': carrier_frequency_head,
                                               'waveform': carrier_waveform_head,
                                               'mod_index': modulation_index_head}).to(device=helper.get_device())}
            elif operation == 'filter':
                filter_type_head = nn.Sequential(
                    nn.Linear(BIG_LINEAR_IN_CHANNELS, HIDDEN_IN_CHANNELS),
                    nn.Linear(HIDDEN_IN_CHANNELS, len(synth_cfg.filter_type_dict))
                )
                filter_freq_head = nn.Sequential(
                    nn.Linear(BIG_LINEAR_IN_CHANNELS, HIDDEN_IN_CHANNELS),
                    nn.Linear(HIDDEN_IN_CHANNELS, 1)
                )
                self.module_dict[index] = {'operation': operation,
                                           'module_dict': nn.ModuleDict({
                                               'filter_type': filter_type_head,
                                               'filter_freq': filter_freq_head}).to(device=helper.get_device())}
            elif operation == 'env_adsr':
                attack_t_head = nn.Sequential(
                    nn.Linear(BIG_LINEAR_IN_CHANNELS, HIDDEN_IN_CHANNELS),
                    nn.Linear(HIDDEN_IN_CHANNELS, 1)
                )
                decay_t_head = nn.Sequential(
                    nn.Linear(BIG_LINEAR_IN_CHANNELS, HIDDEN_IN_CHANNELS),
                    nn.Linear(HIDDEN_IN_CHANNELS, 1)
                )
                sustain_t_head = nn.Sequential(
                    nn.Linear(BIG_LINEAR_IN_CHANNELS, HIDDEN_IN_CHANNELS),
                    nn.Linear(HIDDEN_IN_CHANNELS, 1)
                )
                sustain_level_head = nn.Sequential(
                    nn.Linear(BIG_LINEAR_IN_CHANNELS, HIDDEN_IN_CHANNELS),
                    nn.Linear(HIDDEN_IN_CHANNELS, 1)
                )
                release_t_head = nn.Sequential(
                    nn.Linear(BIG_LINEAR_IN_CHANNELS, HIDDEN_IN_CHANNELS),
                    nn.Linear(HIDDEN_IN_CHANNELS, 1)
                )
                self.module_dict[index] = {'operation': operation,
                                           'module_dict': nn.ModuleDict({
                                               'attack_t': attack_t_head,
                                               'decay_t': decay_t_head,
                                               'sustain_t': sustain_t_head,
                                               'sustain_level': sustain_level_head,
                                               'release_t': release_t_head}).to(device=helper.get_device())}

        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

        # Initialization
        nn.init.kaiming_normal_(self.conv1[0].weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2[0].weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv3[0].weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv4[0].weight, mode='fan_in', nonlinearity='relu')
        # nn.init.kaiming_normal_(self.classification_params, mode='fan_in', nonlinearity='relu')
        # nn.init.kaiming_normal_(self.classification_params, mode='fan_in', nonlinearity='relu')

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
            index = cell.index
            operation = cell.operation

            if operation == 'osc':
                amplitude_head = self.module_dict[index]['module_dict']['amp']
                frequency_head = self.module_dict[index]['module_dict']['freq']
                waveform_head = self.module_dict[index]['module_dict']['waveform']
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
            elif operation == 'fm':
                carrier_amplitude_head = self.module_dict[index]['module_dict']['amp_c']
                carrier_frequency_head = self.module_dict[index]['module_dict']['freq_c']
                waveform_head = self.module_dict[index]['module_dict']['waveform']
                mod_index_head = self.module_dict[index]['module_dict']['mod_index']
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
                filter_type_head = self.module_dict[index]['module_dict']['filter_type']
                filter_freq_head = self.module_dict[index]['module_dict']['filter_freq']
                filter_type_logits = filter_type_head(latent)
                filter_type_probabilities = self.softmax(filter_type_logits)
                predicted_filter_freq = filter_freq_head(latent)
                predicted_filter_freq = self.sigmoid(predicted_filter_freq)

                output_dic[index] = {'operation': operation,
                                     'params': {'filter_type': filter_type_probabilities,
                                                'filter_freq': predicted_filter_freq
                                                }}

            elif operation == 'env_adsr':
                attack_t_head = self.module_dict[index]['module_dict']['attack_t']
                decay_t_head = self.module_dict[index]['module_dict']['decay_t']
                sustain_t_head = self.module_dict[index]['module_dict']['sustain_t']
                sustain_level_head = self.module_dict[index]['module_dict']['sustain_level']
                release_t_head = self.module_dict[index]['module_dict']['release_t']

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
                                     'params': {'attack_t': predicted_attack_t,
                                                'decay_t': predicted_decay_t,
                                                'sustain_t': predicted_sustain_t,
                                                'sustain_level': predicted_sustain_level,
                                                'release_t': predicted_release_t
                                                }}

        return output_dic


if __name__ == "__main__":
    synth_net = BigSynthNetwork()
    summary(synth_net.cuda(), (1, 64, 44))
