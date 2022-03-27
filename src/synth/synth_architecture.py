import matplotlib.pyplot as plt
import numpy
from config import SynthConfig, Config
from synth import synth_modular_presets
import random
import simpleaudio as sa
import numpy as np
import torch
from torch import nn
import helper
from synth.synth_modules import SynthModules


class SynthModularCell:

    def __init__(self,
                 index: tuple,
                 input_list=None,
                 operation=None,
                 parameters=None,
                 signal=None,
                 output=None,
                 default_connection=False,
                 num_channels=4,
                 num_layers=5,
                 synth_config: SynthConfig = None):

        self.check_cell(index, input_list, output, operation, parameters, num_channels, num_layers, synth_config)

        self.index = index
        channel = self.index[0]
        layer = self.index[1]

        if layer == 0:
            self.input_list = None
        elif default_connection:
            self.input_list = [[-1, -1]]
            self.input_list[0][0] = channel
            self.input_list[0][1] = layer - 1
        else:
            self.input_list = input_list

        # if last layer
        if layer == num_layers - 1:
            self.output = None
        elif default_connection:
            self.output = [-1, -1]
            self.output[0] = channel
            self.output[1] = layer + 1
        else:
            self.output = output

        self.operation = operation
        self.parameters = parameters
        self.signal = signal

    @staticmethod
    def check_cell(index, input_list, output, operation, parameters, num_channels, num_layers,
                   synth_config: SynthConfig):
        channel = index[0]
        layer = index[1]

        if len(index) != 2 \
                or channel < 0 \
                or channel > num_channels \
                or layer < 0 \
                or layer > num_layers:
            ValueError("Illegal cell index")

        if input_list is not None:
            if type(input_list) is not list:
                ValueError("Illegal input_list")
            for input in input_list:
                if len(input) != 2:
                    ValueError("Illegal input index")
                input_layer = input[1]

                if input_layer >= layer:
                    ValueError("Illegal input chain")

        if output is not None:
            if len(output) != 2:
                ValueError("Illegal input index")
            output_layer = output[1]

            if output_layer <= layer:
                ValueError("Illegal output chain")

        if operation is not None:
            if operation not in synth_config.modular_synth_operations:
                ValueError("Illegal operation")

            if parameters is not None:
                for key in parameters:
                    if key not in synth_config.modular_synth_params[operation]:
                        ValueError("Illegal parameter for the provided operation")


class SynthModular:
    def __init__(self,
                 synth_cfg: SynthConfig = None,
                 sample_rate=44100,
                 signal_duration_sec=1.0,
                 num_sounds=1,
                 device='cuda:0',
                 preset: str = None
                 ):

        self.architecture = [[SynthModularCell(index=(channel, layer), default_connection=True)
                              for layer in range(synth_cfg.num_layers)]
                             for channel in range(synth_cfg.num_channels)]
        self.num_channels = synth_cfg.num_channels
        self.num_layers = synth_cfg.num_layers
        self.sample_rate = sample_rate
        self.signal_duration_sec = signal_duration_sec
        self.num_sound = num_sounds
        self.signal = torch.zeros((1, int(sample_rate * signal_duration_sec)), requires_grad=True)
        self.synth_cfg = synth_cfg
        self.device = device
        self.preset = self.get_preset(preset)

        if preset is not None:
            self.apply_architecture()

    def apply_architecture(self):
        for cell in self.preset:
            self.apply_cell(cell)

    def apply_cell(self, modular_synth_cell):
        index = modular_synth_cell.index
        self.architecture[index[0]][index[1]] = modular_synth_cell

    def update_cells(self, cell_list):
        for cell in cell_list:
            self.update_cell(cell.index, cell.input_list, cell.operation, cell.parameters, cell.signal, cell.output,
                             self.synth_cfg)

    def update_cell(self,
                    index: tuple,
                    input_list=None,
                    operation=None,
                    parameters=None,
                    signal=None,
                    output=None,
                    synth_config: SynthConfig = None):
        cell = self.architecture[index[0]][index[1]]
        if input_list is not None:
            cell.input_list = input_list
            for input in input_list:
                input_cell = self.architecture[input[0]][input[1]]
                input_cell.output = cell.index
        if operation is not None:
            if operation == 'osc' or operation == 'lfo' and len(cell.input_list) != 0:
                AttributeError(f'Operation {operation} does not take input audio')
            elif operation in ['filter', 'env_adsr'] and len(cell.input_list) != 1:
                AttributeError(f'{operation} must have single input')
            elif operation == 'mix' and len(cell.input_list) != 2:
                AttributeError(f'{operation} must have double inputs')
            elif operation in ['fm', 'am'] and (len(cell.input_list) > 2 or cell.input_list is not None):
                AttributeError(f'{operation} must have single or no inputs')

            cell.operation = operation
        if parameters is not None:
            for key in parameters:
                if key not in synth_config.modular_synth_params[cell.operation]:
                    ValueError("Illegal parameter for the provided operation")
            cell.parameters = parameters
        if signal is not None:
            cell.signal = signal
        if output is not None:
            output_cell_index = cell.output
            output_cell = self.architecture[output_cell_index[0]][output_cell_index[1]]
            if output_cell_index in output_cell.input_list:
                output_cell.input_list.remove(output_cell_index)
            cell.output = output

    def update_cell_parameters(self, index: tuple, parameters: dict):
        cell = self.architecture[index[0]][index[1]]
        for key in parameters:
            if key not in [cell.operation]:
                ValueError("Illegal parameter for the provided operation")
        cell.parameters = parameters

    def generate_random_params(self, synth_cfg: SynthConfig = None, num_sounds=1):
        for layer in range(synth_cfg.num_layers):
            for channel in range(synth_cfg.num_channels):
                cell = self.architecture[channel][layer]
                operation = cell.operation

                if operation == 'osc':
                    params = {'amp': np.random.random_sample(size=num_sounds),
                              'freq': random.choices(synth_cfg.osc_freq_list, k=num_sounds),
                              'waveform': random.choices(list(synth_cfg.wave_type_dict), k=num_sounds)}
                elif operation == 'lfo':
                    params = {'amp': np.random.random_sample(size=num_sounds),
                              'freq': np.random.uniform(low=0, high=synth_cfg.max_lfo_freq, size=num_sounds)}
                elif operation == 'fm':
                    params = {'amp_c': np.random.random_sample(size=num_sounds),
                              'freq_c': random.choices(synth_cfg.osc_freq_list, k=num_sounds),
                              'waveform': random.choices(list(synth_cfg.wave_type_dict), k=num_sounds),
                              'mod_index': np.random.uniform(low=0, high=synth_cfg.max_mod_index, size=num_sounds)}
                elif operation == 'mix':
                    params = None
                elif operation == 'filter':
                    params = {'filter_type': random.choices(list(synth_cfg.filter_type_dict), k=num_sounds),
                              'filter_freq': np.random.uniform(low=synth_cfg.min_filter_freq,
                                                               high=synth_cfg.max_filter_freq,
                                                               size=num_sounds)}
                elif operation == 'env_adsr':
                    attack_t = np.random.random_sample(size=num_sounds)
                    decay_t = np.random.random_sample(size=num_sounds)
                    sustain_t = np.random.random_sample(size=num_sounds)
                    release_t = np.random.random_sample(size=num_sounds)
                    adsr_sum = attack_t + decay_t + sustain_t + release_t
                    attack_t = attack_t / adsr_sum
                    decay_t = decay_t / adsr_sum
                    sustain_t = sustain_t / adsr_sum
                    release_t = release_t / adsr_sum

                    # fixing a numerical issue in case the ADSR times exceeds signal length
                    adsr_aggregated_time = attack_t + decay_t + sustain_t + release_t
                    overflow_indices = [idx for idx, val in enumerate(adsr_aggregated_time) if
                                        val > self.signal_duration_sec]
                    attack_t[overflow_indices] -= 1e-6
                    decay_t[overflow_indices] -= 1e-6
                    sustain_t[overflow_indices] -= 1e-6
                    release_t[overflow_indices] -= 1e-6

                    params = {'attack_t': attack_t,
                              'decay_t': decay_t,
                              'sustain_t': sustain_t,
                              'sustain_level': np.random.random_sample(size=num_sounds),
                              'release_t': release_t}

                elif operation is None:
                    params = None

                if params is not None:
                    for key, val in params.items():
                        if isinstance(val, numpy.ndarray):
                            params[key] = val.tolist()

                    if num_sounds == 1:
                        for key, value in params.items():
                            params[key] = value[0]

                cell.parameters = params

    def generate_signal(self, num_sounds=1):
        synth_module = SynthModules(num_sounds=1,
                                    sample_rate=self.sample_rate,
                                    signal_duration_sec=self.signal_duration_sec,
                                    device=self.device)

        for layer in range(self.num_layers):
            for channel in range(self.num_channels):
                cell = self.architecture[channel][layer]
                operation = cell.operation

                if operation is None:
                    if cell.input_list is None:
                        signal = None
                    elif len(cell.input_list) == 1:
                        input_cell_index = cell.input_list[0]
                        input_cell = self.architecture[input_cell_index[0]][input_cell_index[1]]
                        signal = input_cell.signal
                    else:
                        signal = None
                        AttributeError("Illegal cell input for operation==None")
                    cell.signal = signal

                elif operation == 'osc':
                    cell.signal = synth_module.oscillator(amp=cell.parameters['amp'],
                                                          freq=cell.parameters['freq'],
                                                          phase=0,
                                                          waveform=cell.parameters['waveform'],
                                                          num_sounds=num_sounds)
                elif operation == 'lfo':
                    cell.signal = synth_module.batch_oscillator(amp=cell.parameters['amp'],
                                                                freq=cell.parameters['freq'],
                                                                phase=0,
                                                                waveform='sine')

                elif operation == 'fm':
                    if len(cell.input_list) == 1:
                        input_cell_index = cell.input_list[0]
                        input_cell = self.architecture[input_cell_index[0]][input_cell_index[1]]
                        modulator = input_cell.signal
                    elif cell.input_list is None:
                        modulator = 0
                    else:
                        modulator = 0
                        AttributeError("Illegal cell input")

                    cell.signal = synth_module.batch_oscillator_fm(amp_c=cell.parameters['amp_c'],
                                                                   freq_c=cell.parameters['freq_c'],
                                                                   waveform=cell.parameters['waveform'],
                                                                   mod_index=cell.parameters['mod_index'],
                                                                   modulator=modulator)

                elif operation == 'mix':
                    signal = 0
                    num_inputs = len(cell.input_list)
                    for input_num in range(num_inputs):
                        input_index = cell.input_list[input_num]
                        input_sound = self.architecture[input_index[0]][input_index[1]].signal
                        signal += input_sound

                    cell.signal = signal / num_inputs

                elif operation == 'filter':
                    if len(cell.input_list) == 1:
                        input_cell_index = cell.input_list[0]
                        input_cell = self.architecture[input_cell_index[0]][input_cell_index[1]]
                        input_signal = input_cell.signal
                    else:
                        input_signal = 0
                        AttributeError("Illegal cell input")
                    cell.signal = synth_module.filter(input_signal,
                                                      filter_freq=cell.parameters['filter_freq'],
                                                      filter_type=cell.parameters['filter_type'],
                                                      num_sounds=num_sounds)

                elif operation == 'env_adsr':
                    if len(cell.input_list) == 1:
                        input_cell_index = cell.input_list[0]
                        input_cell = self.architecture[input_cell_index[0]][input_cell_index[1]]
                        input_signal = input_cell.signal
                    else:
                        input_signal = 0
                        AttributeError("Illegal cell input")

                    cell.signal = synth_module.adsr_envelope(input_signal,
                                                             attack_t=cell.parameters['attack_t'],
                                                             decay_t=cell.parameters['decay_t'],
                                                             sustain_t=cell.parameters['sustain_t'],
                                                             sustain_level=cell.parameters['sustain_level'],
                                                             release_t=cell.parameters['release_t'],
                                                             num_sounds=num_sounds)

        # Final signal summing from all channels in the last layer
        final_signal = torch.zeros((int(self.sample_rate * self.signal_duration_sec)),
                                   requires_grad=True,
                                   device=self.device)
        num_active_channels = 0
        for channel in range(self.num_channels):
            signal = self.architecture[channel][self.num_layers - 1].signal
            if signal is not None:
                final_signal = final_signal + signal
                num_active_channels += 1
        if num_active_channels > 0:
            final_signal = final_signal / num_active_channels
        else:
            final_signal = None

        self.signal = final_signal
        return final_signal

    def get_preset(self, preset: str):
        if preset == 'BASIC_FLOW':
            preset_list = synth_modular_presets.BASIC_FLOW
        elif preset == 'OSC':
            preset_list = synth_modular_presets.OSC
        elif preset == 'LFO':
            preset_list = synth_modular_presets.LFO
        elif preset == 'FM':
            preset_list = synth_modular_presets.FM

        else:
            preset_list = None
            ValueError("Unknown PRESET")

        preset_as_synth_input = [SynthModularCell(index=cell.get('index'),
                                                  input_list=cell.get('input_list'),
                                                  operation=cell.get('operation'),
                                                  parameters=cell.get('parameters'),
                                                  signal=cell.get('signal'),
                                                  output=cell.get('output'),
                                                  default_connection=cell.get('default_connection'),
                                                  synth_config=cell.get('synth_config'))
                                 for cell in preset_list]

        return preset_as_synth_input


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    # for i in OSC_FREQ_LIST:
    #     a = SynthOscOnly('audio_example', {'osc1_freq': i}, num_sounds=1)
    #     # signal = a.signal.squeeze().cpu().detach().numpy()
    #     # plt.plot(signal)
    #     # plt.show()
    #     play_obj = sa.play_buffer(a.signal.detach().cpu().numpy(),
    #                               num_channels=1,
    #                               bytes_per_sample=4,
    #                               sample_rate=44100)
    #     play_obj.wait_done()
    #
    #     a = helper.mel_spectrogram_transform(a.signal).squeeze()
    #
    #     if PLOT_SPEC:
    #         helper.plot_spectrogram(a.cpu().detach().numpy(),
    #                                 scale='linear',
    #                                 title="MelSpectrogram (dB)",
    #                                 ylabel='mel freq')

    # a = SynthOscOnly('audio_example', num_sounds=10)

    modular_synth_basic_flow = [
        SynthModularCell(index=(0, 0), operation='lfo', default_connection=True),
        SynthModularCell(index=(0, 1), operation='fm', default_connection=True),
        SynthModularCell(index=(1, 0), operation='lfo', default_connection=True),
        SynthModularCell(index=(1, 1), operation='fm', input_list=[[1, 0]], output=[0, 2]),
        SynthModularCell(index=(1, 2), operation=None, input_list=None),
        SynthModularCell(index=(0, 2),
                         operation='mix',
                         input_list=[[0, 1], [1, 1]]),
        SynthModularCell(index=(0, 3), operation='filter', default_connection=True),
        SynthModularCell(index=(0, 4), operation='env_adsr', default_connection=True),
    ]

    update_params = [
        SynthModularCell(index=(0, 0), parameters={'amp': 1, 'freq': 3, 'waveform': 'sine'}),
        SynthModularCell(index=(0, 1), parameters={'amp_c': 0.9, 'freq_c': 220, 'waveform': 'square',
                                                   'mod_index': 10}),
        SynthModularCell(index=(1, 0), parameters={'amp': 1, 'freq': 1, 'waveform': 'sine'}),
        SynthModularCell(index=(1, 1), parameters={'amp_c': 0.7, 'freq_c': 500, 'waveform': 'sine',
                                                   'mod_index': 10}),
        SynthModularCell(index=(0, 2), parameters={'factor': 0}),
        SynthModularCell(index=(0, 3), parameters={'filter_freq': 15000, 'filter_type': 'low_pass'}),
        SynthModularCell(index=(0, 4), parameters={'attack_t': 0.25, 'decay_t': 0.25, 'sustain_t': 0.25,
                                                   'sustain_level': 0.3, 'release_t': 0.25})
    ]
    a = SynthModular()
    a.apply_architecture()
    a.update_cells(update_params)
    a.generate_signal()
    plt.plot(a.signal.detach().cpu().numpy())
    plt.ylim([-1, 1])
    plt.show()
    for i in range(4):
        play_obj = sa.play_buffer(a.signal.detach().cpu().numpy(),
                                  num_channels=1,
                                  bytes_per_sample=4,
                                  sample_rate=44100)
        play_obj.wait_done()
    num_sounds = 10
    b = torch.rand(10, 44100)
    b = helper.move_to(b, helper.get_device())
    criterion = nn.MSELoss()
    loss = criterion(a.signal, b)
    loss.backward()
    # plt.plot(a.signal.cpu())
    # plt.ylim([-1, 1])
    # plt.show()
    # for i in range(num_sounds):
    #     plt.plot(a.signal[i].detach().cpu().numpy())
    #     plt.ylim([-1, 1])
    #     plt.show()
    #     play_obj = sa.play_buffer(a.signal[i].detach().cpu().numpy(),
    #                               num_channels=1,
    #                               bytes_per_sample=4,
    #                               sample_rate=44100)
    #     play_obj.wait_done()
