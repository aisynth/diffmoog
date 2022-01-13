import matplotlib.pyplot as plt
import numpy
from src.config import SIGNAL_DURATION_SEC, SAMPLE_RATE
from synth.synth_config import OSC_FREQ_LIST, WAVE_TYPE_DIC, NUM_LAYERS, NUM_CHANNELS, MAX_MOD_INDEX, MAX_LFO_FREQ, \
    FILTER_TYPE_DIC, MIN_FILTER_FREQ, MAX_FILTER_FREQ, MODULAR_SYNTH_OPERATIONS, MODULAR_SYNTH_PARAMS
import random
import simpleaudio as sa
import numpy as np
import torch
from torch import nn
import helper
from synth.synth_modules import SynthModules


class SynthModular:
    def __init__(self, num_sounds=1):

        self.architecture = [[SynthModularCell(index=(channel, layer), default_connection=True)
                              for layer in range(NUM_LAYERS)]
                             for channel in range(NUM_CHANNELS)]
        self.num_sound = num_sounds
        self.signal = torch.zeros((1, int(SAMPLE_RATE * SIGNAL_DURATION_SEC)), requires_grad=True)

    def apply_architecture(self, cell_list):
        for cell in cell_list:
            self.apply_cell(cell)

    def apply_cell(self, modular_synth_cell):
        index = modular_synth_cell.index
        self.architecture[index[0]][index[1]] = modular_synth_cell

    def update_cells(self, cell_list):
        for cell in cell_list:
            self.update_cell(cell.index, cell.input_list, cell.operation, cell.parameters, cell.signal, cell.output)

    def update_cell(self, index: tuple, input_list=None, operation=None, parameters=None, signal=None, output=None):
        cell = self.architecture[index[0]][index[1]]
        if input_list is not None:
            cell.input_list = input_list
            for input in input_list:
                input_cell = self.architecture[input[0]][input[1]]
                input_cell.output = cell.index
        if operation is not None:
            if operation == 'osc' and len(cell.input_list) != 0:
                AttributeError('Oscillator does not take input audio')
            elif operation in ['filter', 'env_adsr'] and len(cell.input_list) != 1:
                AttributeError(f'{operation} must have single input')
            elif operation == 'mix' and len(cell.input_list) != 2:
                AttributeError(f'{operation} must have double inputs')
            elif operation in ['fm', 'am'] and (len(cell.input_list) > 2 or cell.input_list is not None):
                AttributeError(f'{operation} must have single or no inputs')

            cell.operation = operation
        if parameters is not None:
            for key in parameters:
                if key not in MODULAR_SYNTH_PARAMS[cell.operation]:
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
            if key not in MODULAR_SYNTH_PARAMS[cell.operation]:
                ValueError("Illegal parameter for the provided operation")
        cell.parameters = parameters

    def generate_random_parmas(self, num_sounds=1):
        for layer in range(NUM_LAYERS):
            for channel in range(NUM_CHANNELS):
                cell = self.architecture[channel][layer]
                operation = cell.operation

                if operation == 'osc':
                    params = {'amp': np.random.random_sample(size=num_sounds),
                              'freq': np.random.uniform(low=0, high=MAX_LFO_FREQ, size=num_sounds),
                              'waveform': random.choices(list(WAVE_TYPE_DIC), k=num_sounds)}
                elif operation == 'fm':
                    params = {'amp_c': np.random.random_sample(size=num_sounds),
                              'freq_c': random.choices(OSC_FREQ_LIST, k=num_sounds),
                              'waveform': random.choices(list(WAVE_TYPE_DIC), k=num_sounds),
                              'mod_index': np.random.uniform(low=0, high=MAX_MOD_INDEX, size=num_sounds)}
                elif operation == 'mix':
                    params = None
                elif operation == 'filter':
                    params = {'filter_type': random.choices(list(FILTER_TYPE_DIC), k=num_sounds),
                              'filter_freq': np.random.uniform(low=MIN_FILTER_FREQ,
                                                               high=MAX_FILTER_FREQ,
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
                                        val > SIGNAL_DURATION_SEC]
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
        synth_module = SynthModules(num_sounds)

        for layer in range(NUM_LAYERS):
            for channel in range(NUM_CHANNELS):
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

                    cell.signal = synth_module.oscillator_fm(amp_c=cell.parameters['amp_c'],
                                                             freq_c=cell.parameters['freq_c'],
                                                             waveform=cell.parameters['waveform'],
                                                             mod_index=cell.parameters['mod_index'],
                                                             modulator=modulator,
                                                             num_sounds=num_sounds)
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

        final_signal = torch.zeros((int(SAMPLE_RATE * SIGNAL_DURATION_SEC)),
                                   requires_grad=True,
                                   device=helper.get_device())
        num_active_channels = 0
        for channel in range(NUM_CHANNELS):
            signal = self.architecture[channel][NUM_LAYERS-1].signal
            if signal is not None:
                final_signal = final_signal + signal
                num_active_channels += 1
        if num_active_channels > 0:
            final_signal = final_signal / num_active_channels
        else:
            final_signal = None

        self.signal = final_signal
        return final_signal


class SynthModularCell:

    def __init__(self,
                 index: tuple,
                 input_list=None,
                 operation=None,
                 parameters=None,
                 signal=None,
                 output=None,
                 default_connection=False):

        self.check_cell(index, input_list, output, operation, parameters)

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
        if layer == NUM_LAYERS - 1:
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
    def check_cell(index, input_list, output, operation, parameters):
        channel = index[0]
        layer = index[1]

        if len(index) != 2 \
                or channel < 0 \
                or channel > NUM_CHANNELS \
                or layer < 0 \
                or layer > NUM_LAYERS:
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
            if operation not in MODULAR_SYNTH_OPERATIONS:
                ValueError("Illegal operation")

            if parameters is not None:
                for key in parameters:
                    if key not in MODULAR_SYNTH_PARAMS[operation]:
                        ValueError("Illegal parameter for the provided operation")


class SynthBasicFlow:
    """A basic synthesizer signal flow architecture.
        The synth is based over common commercial software synthesizers.
        It has dual oscillators followed by FM module, summed together
        and passed in a frequency filter and envelope shaper

        [osc1] -> FM
                    \
                     + -> [frequency filter] -> [envelope shaper] -> output sound
                    /
        [osc2] -> FM

        Args:
            self: Self object
            file_name: name for sound
            parameters_dict(optional): parameters for the synth components to generate specific sounds
            num_sounds: number of sounds to generate.
        """

    def __init__(self, file_name='unnamed_sound', parameters_dict=None, num_sounds=1):
        self.file_name = file_name
        self.params_dict = {}
        # init parameters_dict
        if parameters_dict is None:
            self.init_random_synth_params(num_sounds)
        elif type(parameters_dict) is dict:
            self.params_dict = parameters_dict.copy()
        else:
            ValueError("Provided parameters are not provided as dictionary")

        # generate signal with basic signal flow
        self.signal = self.generate_signal(num_sounds)

    def init_random_synth_params(self, num_sounds):
        """init params_dict with lists of parameters"""

        # todo: refactor: initializations by iterating/referencing synth.PARAM_LIST
        self.params_dict['osc1_amp'] = np.random.random_sample(size=num_sounds)
        self.params_dict['osc1_freq'] = random.choices(OSC_FREQ_LIST, k=num_sounds)
        self.params_dict['osc1_wave'] = random.choices(list(WAVE_TYPE_DIC), k=num_sounds)
        self.params_dict['osc1_mod_index'] = np.random.uniform(low=0, high=MAX_MOD_INDEX, size=num_sounds)
        self.params_dict['lfo1_freq'] = np.random.uniform(low=0, high=MAX_LFO_FREQ, size=num_sounds)

        self.params_dict['osc2_amp'] = np.random.random_sample(size=num_sounds)
        self.params_dict['osc2_freq'] = random.choices(OSC_FREQ_LIST, k=num_sounds)
        self.params_dict['osc2_wave'] = random.choices(list(WAVE_TYPE_DIC), k=num_sounds)
        self.params_dict['osc2_mod_index'] = np.random.uniform(low=0, high=MAX_MOD_INDEX, size=num_sounds)
        self.params_dict['lfo2_freq'] = np.random.uniform(low=0, high=MAX_LFO_FREQ, size=num_sounds)

        self.params_dict['filter_type'] = random.choices(list(FILTER_TYPE_DIC), k=num_sounds)
        self.params_dict['filter_freq'] = \
            np.random.uniform(low=MIN_FILTER_FREQ, high=MAX_FILTER_FREQ, size=num_sounds)

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
        overflow_indices = [idx for idx, val in enumerate(adsr_aggregated_time) if val > SIGNAL_DURATION_SEC]
        attack_t[overflow_indices] -= 1e-6
        decay_t[overflow_indices] -= 1e-6
        sustain_t[overflow_indices] -= 1e-6
        release_t[overflow_indices] -= 1e-6

        self.params_dict['attack_t'] = attack_t
        self.params_dict['decay_t'] = decay_t
        self.params_dict['sustain_t'] = sustain_t
        self.params_dict['release_t'] = release_t
        self.params_dict['sustain_level'] = np.random.random_sample(size=num_sounds)

        for key, val in self.params_dict.items():
            if isinstance(val, numpy.ndarray):
                self.params_dict[key] = val.tolist()

        if num_sounds == 1:
            for key, value in self.params_dict.items():
                self.params_dict[key] = value[0]

    def generate_signal(self, num_sounds):
        osc1_amp = self.params_dict['osc1_amp']
        osc1_freq = self.params_dict['osc1_freq']
        osc1_wave = self.params_dict['osc1_wave']
        osc1_mod_index = self.params_dict['osc1_mod_index']
        lfo1_freq = self.params_dict['lfo1_freq']

        osc2_amp = self.params_dict['osc2_amp']
        osc2_freq = self.params_dict['osc2_freq']
        osc2_wave = self.params_dict['osc2_wave']
        osc2_mod_index = self.params_dict['osc2_mod_index']
        lfo2_freq = self.params_dict['lfo2_freq']

        filter_type = self.params_dict['filter_type']
        filter_freq = self.params_dict['filter_freq']

        attack_t = self.params_dict['attack_t']
        decay_t = self.params_dict['decay_t']
        sustain_t = self.params_dict['sustain_t']
        release_t = self.params_dict['release_t']
        sustain_level = self.params_dict['sustain_level']

        synth = SynthModules(num_sounds)

        lfo1 = synth.oscillator(amp=1,
                                freq=lfo1_freq,
                                phase=0,
                                waveform='sine',
                                num_sounds=num_sounds)

        fm_osc1 = synth.oscillator_fm(amp_c=osc1_amp,
                                      freq_c=osc1_freq,
                                      waveform=osc1_wave,
                                      mod_index=osc1_mod_index,
                                      modulator=lfo1,
                                      num_sounds=num_sounds)

        lfo2 = synth.oscillator(amp=1,
                                freq=lfo2_freq,
                                phase=0,
                                waveform='sine',
                                num_sounds=num_sounds)

        fm_osc2 = synth.oscillator_fm(amp_c=osc2_amp,
                                      freq_c=osc2_freq,
                                      waveform=osc2_wave,
                                      mod_index=osc2_mod_index,
                                      modulator=lfo2,
                                      num_sounds=num_sounds)

        mixed_signal = (fm_osc1 + fm_osc2) / 2

        # mixed_signal = mixed_signal.cpu()

        filtered_signal = synth.filter(mixed_signal, filter_freq, filter_type, num_sounds)

        enveloped_signal = synth.adsr_envelope(filtered_signal,
                                               attack_t,
                                               decay_t,
                                               sustain_t,
                                               sustain_level,
                                               release_t,
                                               num_sounds)

        return enveloped_signal


class SynthOscOnly:
    """A synthesizer that produces a single sine oscillator.

        Args:
            self: Self object
            file_name: name for sound
            parameters_dict(optional): parameters for the synth components to generate specific sounds
            num_sounds: number of sounds to generate.
        """

    def __init__(self, file_name='unnamed_sound', parameters_dict=None, num_sounds=1):
        self.file_name = file_name
        self.params_dict = {}
        # init parameters_dict
        if parameters_dict is None:
            self.init_random_synth_params(num_sounds)
        elif type(parameters_dict) is dict:
            self.params_dict = parameters_dict.copy()
        else:
            ValueError("Provided parameters are not provided as dictionary")

        # generate signal with basic signal flow
        self.signal = self.generate_signal(num_sounds)

    def init_random_synth_params(self, num_sounds):
        """init params_dict with lists of parameters"""

        self.params_dict['osc1_freq'] = random.choices(OSC_FREQ_LIST, k=num_sounds)

        for key, val in self.params_dict.items():
            if isinstance(val, numpy.ndarray):
                self.params_dict[key] = val.tolist()

        if num_sounds == 1:
            for key, value in self.params_dict.items():
                self.params_dict[key] = value[0]

    def generate_signal(self, num_sounds):
        osc_freq = self.params_dict['osc1_freq']

        synthesizer = SynthModules(num_sounds)

        osc = synthesizer.oscillator(amp=1,
                                     freq=osc_freq,
                                     phase=0,
                                     waveform='sine',
                                     num_sounds=num_sounds)
        return osc


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
                    SynthModularCell(index=(0, 0), operation='osc', default_connection=True),
                    SynthModularCell(index=(0, 1), operation='fm', default_connection=True),
                    SynthModularCell(index=(1, 0), operation='osc', default_connection=True),
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
    a.apply_architecture(modular_synth_basic_flow)
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
    a = SynthBasicFlow('audio_example', num_sounds=num_sounds)
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
