from torch import nn
import torch
import helper
from torchsummary import summary
from synth.synth_config import REGRESSION_PARAM_LIST
from synth.synth_modules import WAVE_TYPE_DIC, FILTER_TYPE_DIC, OSC_FREQ_LIST
from config import SYNTH_TYPE, ARCHITECTURE, MODEL_FREQUENCY_OUTPUT, PRESET, CNN_NETWORK
from synth.synth_architecture import SynthOscOnly
from synth.synth_config import NUM_CHANNELS, NUM_LAYERS
from synth.synth_modular_presets import BASIC_FLOW, FM

# todo: this is value from Valerio Tutorial. has to check
# LINEAR_IN_CHANNELS = 128 * 5 * 4
SMALL_LINEAR_IN_CHANNELS = 4480
# BIG_LINEAR_IN_CHANNELS = 17920
# BIG_LINEAR_IN_CHANNELS = 32256
# BIG_LINEAR_IN_CHANNELS = 13824
BIG_LINEAR_IN_CHANNELS = 23040
# BIG_LINEAR_IN_CHANNELS = 60928
# LINEAR_IN_CHANNELS = 8064
HIDDEN_IN_CHANNELS = 128

freq_dict = {'osc1_freq': torch.tensor(OSC_FREQ_LIST, requires_grad=False, device=helper.get_device())}
synth_obj = SynthOscOnly(file_name=None, parameters_dict=freq_dict, num_sounds=len(OSC_FREQ_LIST))
SPECTROGRAMS_TENSOR = helper.mel_spectrogram_transform(synth_obj.signal)


class SmallSynthNetwork(nn.Module):
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

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv4 = nn.Sequential(
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
        self.flatten = nn.Flatten()
        if SYNTH_TYPE == 'OSC_ONLY':
            self.linear = nn.Linear(SMALL_LINEAR_IN_CHANNELS, len(OSC_FREQ_LIST))

        elif SYNTH_TYPE == 'SYNTH_BASIC':
            self.classification_params = nn.ModuleDict([
                ['osc1_freq', nn.Linear(SMALL_LINEAR_IN_CHANNELS, len(OSC_FREQ_LIST))],
                ['osc1_wave', nn.Linear(SMALL_LINEAR_IN_CHANNELS, len(WAVE_TYPE_DIC))],
                ['osc2_freq', nn.Linear(SMALL_LINEAR_IN_CHANNELS, len(OSC_FREQ_LIST))],
                ['osc2_wave', nn.Linear(SMALL_LINEAR_IN_CHANNELS, len(WAVE_TYPE_DIC))],
                ['filter_type', nn.Linear(SMALL_LINEAR_IN_CHANNELS, len(FILTER_TYPE_DIC))],
            ])

            self.regression_params = nn.Sequential(
                nn.Linear(SMALL_LINEAR_IN_CHANNELS, HIDDEN_IN_CHANNELS),
                nn.Linear(HIDDEN_IN_CHANNELS, len(REGRESSION_PARAM_LIST))
            )
        else:
            raise ValueError("Provided SYNTH_TYPE is not recognized")

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
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)

        # Apply different heads to predict each synth parameter
        output_dic = {}
        if SYNTH_TYPE == 'OSC_ONLY':
            x = self.linear(x)
            logits = x
            probabilities = self.softmax(logits)
            osc_freq_tensor = torch.tensor(OSC_FREQ_LIST, requires_grad=False, device=helper.get_device())
            output_dic['osc1_freq'] = torch.matmul(probabilities, osc_freq_tensor)

            return output_dic, logits

        if SYNTH_TYPE == 'SYNTH_BASIC':
            for out_name, lin in self.classification_params.items():
                # -----> do not use softmax if using CrossEntropyLoss()
                x = lin(x)
                probabilities = self.softmax(x)
                if out_name == 'osc1_freq' or out_name == 'osc2_freq':
                    osc_freq_tensor = torch.tensor(OSC_FREQ_LIST, requires_grad=False, device=helper.get_device())
                    output_dic[out_name] = torch.matmul(probabilities, osc_freq_tensor)
                else:
                    output_dic[out_name] = probabilities

            x = self.regression_params(x)
            x = self.sigmoid(x)

            for index, param in enumerate(REGRESSION_PARAM_LIST):
                output_dic[param] = x[:, index]

        return output_dic, None


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

    def __init__(self):
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

        if SYNTH_TYPE == 'MODULAR':
            self.module_dict = {}
            if PRESET == 'BASIC_FLOW':
                self.preset = BASIC_FLOW
            elif PRESET == 'FM':
                self.preset = FM
            else:
                ValueError("Unknown preset")

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
                        nn.Linear(HIDDEN_IN_CHANNELS, len(WAVE_TYPE_DIC))
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
                        nn.Linear(HIDDEN_IN_CHANNELS, len(WAVE_TYPE_DIC))
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
                        nn.Linear(HIDDEN_IN_CHANNELS, len(FILTER_TYPE_DIC))
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

        elif SYNTH_TYPE == 'OSC_ONLY':
            if MODEL_FREQUENCY_OUTPUT == 'SINGLE':
                # self.linear = nn.Linear(BIG_LINEAR_IN_CHANNELS, 1)
                self.linear_sequential = nn.Sequential(
                    nn.Linear(BIG_LINEAR_IN_CHANNELS, HIDDEN_IN_CHANNELS),
                    nn.Linear(HIDDEN_IN_CHANNELS, 1)
                )
            else:
                self.linear = nn.Linear(BIG_LINEAR_IN_CHANNELS, len(OSC_FREQ_LIST))

        elif SYNTH_TYPE == 'SYNTH_BASIC':
            if CNN_NETWORK == 'BIG':
                self.classification_params = nn.ModuleDict([
                    ['osc1_freq', nn.Linear(BIG_LINEAR_IN_CHANNELS, len(OSC_FREQ_LIST))],
                    ['osc1_wave', nn.Linear(BIG_LINEAR_IN_CHANNELS, len(WAVE_TYPE_DIC))],
                    ['osc2_freq', nn.Linear(BIG_LINEAR_IN_CHANNELS, len(OSC_FREQ_LIST))],
                    ['osc2_wave', nn.Linear(BIG_LINEAR_IN_CHANNELS, len(WAVE_TYPE_DIC))],
                    ['filter_type', nn.Linear(BIG_LINEAR_IN_CHANNELS, len(FILTER_TYPE_DIC))],
                ])
            elif CNN_NETWORK == 'SMALL':
                self.classification_params = nn.ModuleDict([
                    ['osc1_freq', nn.Linear(SMALL_LINEAR_IN_CHANNELS, len(OSC_FREQ_LIST))],
                    ['osc1_wave', nn.Linear(SMALL_LINEAR_IN_CHANNELS, len(WAVE_TYPE_DIC))],
                    ['osc2_freq', nn.Linear(SMALL_LINEAR_IN_CHANNELS, len(OSC_FREQ_LIST))],
                    ['osc2_wave', nn.Linear(SMALL_LINEAR_IN_CHANNELS, len(WAVE_TYPE_DIC))],
                    ['filter_type', nn.Linear(SMALL_LINEAR_IN_CHANNELS, len(FILTER_TYPE_DIC))],
                ])
        else:
            raise ValueError("Provided SYNTH_TYPE is not recognized")

        if SYNTH_TYPE == 'SYNTH_BASIC':
            self.regression_params = nn.Sequential(
                nn.Linear(BIG_LINEAR_IN_CHANNELS, HIDDEN_IN_CHANNELS),
                nn.Linear(HIDDEN_IN_CHANNELS, len(REGRESSION_PARAM_LIST))
            )

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
                 2st argument: logits - frequency logits prediction when SYNTH_TYPE == 'OSC_ONLY'
                                None - when SYNTH_TYPE != 'OSC_ONLY'
        """
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        latent = self.flatten(x)

        # Apply different heads to predict each synth parameter
        output_dic = {}

        if SYNTH_TYPE == 'MODULAR':
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

        elif SYNTH_TYPE == 'SYNTH_BASIC':
            for out_name, lin in self.classification_params.items():
                # -----> do not use softmax if using CrossEntropyLoss()
                x = lin(latent)
                probabilities = self.softmax(x)
                if out_name == 'osc1_freq' or out_name == 'osc2_freq':
                    osc_freq_tensor = torch.tensor(OSC_FREQ_LIST, requires_grad=False, device=helper.get_device())
                    output_dic[out_name] = torch.matmul(probabilities, osc_freq_tensor)
                else:
                    output_dic[out_name] = probabilities

            x = self.regression_params(latent)
            x = self.sigmoid(x)

            for index, param in enumerate(REGRESSION_PARAM_LIST):
                output_dic[param] = x[:, index]

        elif SYNTH_TYPE == 'OSC_ONLY':
            x = self.linear_sequential(latent)
            logits = x
            probabilities = self.softmax(logits)
            # x = torch.square(x)
            # x = torch.sqrt(x)

            if ARCHITECTURE == 'SPEC_NO_SYNTH':
                # inner product between probabilities and all Spectrograms
                weighted_avg_spectrograms = torch.einsum("ik,klm->ilm", probabilities, SPECTROGRAMS_TENSOR)
                output_dic['osc1_freq'] = weighted_avg_spectrograms

            else:
                if MODEL_FREQUENCY_OUTPUT == 'WEIGHTED':
                    osc_freq_tensor = torch.tensor(OSC_FREQ_LIST, requires_grad=False, device=helper.get_device())
                    output_dic['osc1_freq'] = torch.matmul(probabilities, osc_freq_tensor)
                elif MODEL_FREQUENCY_OUTPUT == 'LOGITS':
                    output_dic['osc1_freq'] = logits
                elif MODEL_FREQUENCY_OUTPUT == 'PROBS':
                    output_dic['osc1_freq'] = probabilities
                elif MODEL_FREQUENCY_OUTPUT == 'SINGLE':
                    x = torch.square(x)
                    x = torch.sqrt(x)
                    output_dic['osc1_freq'] = torch.squeeze(x)
                else:
                    ValueError("MODEL_FREQUENCY_OUTPUT is not known")

            return output_dic, logits

        return output_dic, None


if __name__ == "__main__":
    synth_net = SmallSynthNetwork()
    summary(synth_net.cuda(), (1, 64, 44))
