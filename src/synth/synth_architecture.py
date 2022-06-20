import matplotlib.pyplot as plt
import numpy
from config import SynthConfig
from synth import synth_modular_presets
import random
import simpleaudio as sa
import numpy as np
import torch
from torch import nn
from model import helper
from synth.synth_modules import SynthModules, make_envelope_shape


class SynthModularCell:

    def __init__(self,
                 index: tuple,
                 input_list=None,
                 operation=None,
                 parameters=None,
                 signal=None,
                 output_list=None,
                 default_connection=False,
                 num_channels=4,
                 num_layers=5,
                 synth_config: SynthConfig = None):

        self.check_cell(index, input_list, output_list, operation, parameters, num_channels, num_layers, synth_config)

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
            self.output_list = None
        elif default_connection:
            self.output = [-1, -1]
            self.output[0] = channel
            self.output[1] = layer + 1
        else:
            self.output_list = output_list

        self.operation = operation
        self.parameters = parameters
        self.signal = signal

    @staticmethod
    def check_cell(index, input_list, output_list, operation, parameters, num_channels, num_layers,
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
            for input_ in input_list:
                if len(input_) != 2:
                    ValueError("Illegal input index")
                input_layer = input_[1]

                if input_layer >= layer:
                    ValueError("Illegal input chain")

        if output_list is not None:
            if type(output_list) is not list:
                ValueError("Illegal input_list - not a list")
            for output_ in output_list:
                if len(output_list) != 2:
                    ValueError("Illegal output index")
                output_layer = output_[1]

                if output_layer <= layer:
                    ValueError("Illegal output chain. Output must be chained to a layer > cell.layer")

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
                 num_sounds_=1,
                 device='cuda:0',
                 preset: str = None
                 ):

        self.architecture = [[SynthModularCell(index=(channel, layer),
                                               default_connection=True,
                                               num_channels=synth_cfg.num_channels,
                                               num_layers=synth_cfg.num_layers)
                              for layer in range(synth_cfg.num_layers)]
                             for channel in range(synth_cfg.num_channels)]
        self.num_channels = synth_cfg.num_channels
        self.num_layers = synth_cfg.num_layers
        self.sample_rate = sample_rate
        self.signal_duration_sec = signal_duration_sec
        self.num_sound = num_sounds_
        self.signal = torch.zeros((1, int(sample_rate * signal_duration_sec)), requires_grad=True)
        self.synth_cfg = synth_cfg
        self.device = device
        self.preset = self.get_preset(preset, synth_cfg)

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

    def update_cells_from_dict(self, params_dict):
        for cell_index, cell_params in params_dict.items():
            self.update_cell(index=cell_index, parameters=cell_params['parameters'], synth_config=self.synth_cfg)

    def reset_signal(self):
        self.signal = torch.zeros((1, int(self.sample_rate * self.signal_duration_sec)), requires_grad=True)
        for c in range(self.num_channels):
            for l in range(self.num_layers):
                self.architecture[c][l].signal = torch.zeros((1, int(self.sample_rate * self.signal_duration_sec)),
                                                             requires_grad=True)

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
            for input_ in input_list:
                input_cell = self.architecture[input_[0]][input_[1]]
                input_cell.output = cell.index
        if operation is not None:
            if operation == 'osc' or operation[:3] == 'lfo' and len(cell.input_list) != 0:
                AttributeError(f'Operation {operation} does not take input audio')
            elif operation in ['filter', 'env_adsr'] and len(cell.input_list) != 1:
                AttributeError(f'{operation} must have single input')
            elif operation == 'mix' and len(cell.input_list) != 2:
                AttributeError(f'{operation} must have double inputs')
            elif operation in ['fm', 'am'] and (len(cell.input_list) > 2 or cell.input_list is not None):
                AttributeError(f'{operation} must have single or no inputs')

            cell.operation = operation
        if parameters is not None and isinstance(parameters, dict):
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

    def generate_random_params(self, synth_cfg: SynthConfig = None, num_sounds_=1):
        params = {}
        np.random.seed(synth_cfg.seed)
        for layer in range(synth_cfg.num_layers):
            for channel in range(synth_cfg.num_channels):
                cell = self.architecture[channel][layer]
                operation = cell.operation
                output_list = cell.output_list

                if operation == 'osc':
                    params = {'amp': np.random.random_sample(size=num_sounds_),
                              'freq': random.choices(synth_cfg.osc_freq_list, k=num_sounds_),
                              'waveform': random.choices(list(synth_cfg.wave_type_dict), k=num_sounds_)}

                elif operation == 'lfo':
                    params = {'freq': np.random.uniform(low=0, high=synth_cfg.max_lfo_freq, size=num_sounds_),
                              'waveform': random.choices([k for k in synth_cfg.wave_type_dict.keys()],
                                                         k=num_sounds_)}

                elif operation == 'lfo_non_sine':
                    params = {'freq': np.random.uniform(low=0, high=synth_cfg.max_lfo_freq, size=num_sounds_),
                              'waveform': random.choices([k for k in synth_cfg.wave_type_dict.keys() if k != 'sine'],
                                                         k=num_sounds_)}

                elif operation == 'lfo_sine':
                    params['active'] = np.random.choice([True, False], size=num_sounds_)
                    params['freq'] = np.random.uniform(low=synth_cfg.min_lfo_freq,
                                                       high=synth_cfg.max_lfo_freq,
                                                       size=num_sounds_) * params['active'].astype(int)
                    rng = np.random.default_rng()
                    #todo: put None or [-1,1] in output of non-active operation
                    params['output'] = rng.choice(output_list, size=num_sounds_, axis=0)

                elif operation == 'fm':
                    params = {'freq_c': self._sample_c_freq(synth_cfg, num_sounds_),
                              'waveform': random.choices(list(synth_cfg.wave_type_dict), k=num_sounds_),
                              'mod_index': np.random.uniform(low=0, high=synth_cfg.max_mod_index, size=num_sounds_)}

                elif operation in ['fm_sine', 'fm_square', 'fm_saw']:
                    params = {'freq_c': self._sample_c_freq(synth_cfg, num_sounds_),
                              'mod_index': np.random.uniform(low=0, high=synth_cfg.max_mod_index, size=num_sounds_)}

                elif operation == 'mix':
                    params = None

                elif operation == 'amplitude_shape':
                    attack_t, decay_t, sustain_t, sustain_level, release_t = \
                        self.generate_random_adsr_values(num_sounds_=num_sounds_)

                    params = {'attack_t': attack_t,
                              'decay_t': decay_t,
                              'sustain_t': sustain_t,
                              'sustain_level': sustain_level,
                              'release_t': release_t,
                              'envelope': torch.full([num_sounds_], -1)}

                    # attack_t, decay_t, sustain_t, sustain_level, release_t = \
                    #     self.generate_random_adsr_values(num_sounds_=num_sounds_)

                    # envelope = self.make_envelope_shape(attack_t,
                    #                                     decay_t,
                    #                                     sustain_t,
                    #                                     sustain_level,
                    #                                     release_t,
                    #                                     num_sounds_)
                    # params = {'envelope': envelope}

                elif operation == 'filter':
                    params = {'filter_type': random.choices(list(synth_cfg.filter_type_dict), k=num_sounds_),
                              'filter_freq': np.random.uniform(low=synth_cfg.min_filter_freq,
                                                               high=synth_cfg.max_filter_freq,
                                                               size=num_sounds_)}

                elif operation == 'lowpass_filter':
                    params = {'filter_freq': np.random.uniform(low=synth_cfg.min_filter_freq,
                                                               high=synth_cfg.max_filter_freq,
                                                               size=num_sounds_),
                              'resonance': np.random.uniform(low=synth_cfg.min_resonance_val,
                                                             high=synth_cfg.max_resonance_val,
                                                             size=num_sounds_)}
                elif operation == 'env_adsr':

                    attack_t, decay_t, sustain_t, sustain_level, release_t = \
                        self.generate_random_adsr_values(num_sounds_=num_sounds_)

                    params = {'attack_t': attack_t,
                              'decay_t': decay_t,
                              'sustain_t': sustain_t,
                              'sustain_level': sustain_level,
                              'release_t': release_t}

                elif operation == 'tremolo':
                    pass

                elif operation is None:
                    params = None

                if params is not None:
                    for key, val in params.items():
                        if isinstance(val, numpy.ndarray):
                            params[key] = val.tolist()

                    if num_sounds_ == 1:
                        for key, value in params.items():
                            params[key] = value[0]

                cell.parameters = params

    def generate_random_adsr_values(self, num_sounds_=1):
        attack_t = np.random.random(size=num_sounds_)
        decay_t = np.random.random(size=num_sounds_)
        sustain_t = np.random.random(size=num_sounds_)
        release_t = np.random.random(size=num_sounds_)
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

        sustain_level = np.random.random(size=num_sounds_)

        return attack_t, decay_t, sustain_t, sustain_level, release_t

    def generate_signal(self, num_sounds_=1):
        synth_module = SynthModules(num_sounds=1,
                                    sample_rate=self.sample_rate,
                                    signal_duration_sec=self.signal_duration_sec,
                                    device=self.device)

        output_signals = {}
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
                                                          num_sounds=num_sounds_)
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
                    cell.signal = synth_module.batch_oscillator(amp=1.0,
                                                                freq=cell.parameters['freq'],
                                                                phase=0,
                                                                waveform='sine')

                elif operation in ['fm', 'fm_sine', 'fm_square', 'fm_saw']:
                    if len(cell.input_list) == 1:
                        input_cell_index = cell.input_list[0]
                        input_cell = self.architecture[input_cell_index[0]][input_cell_index[1]]
                        modulator = input_cell.signal
                    elif cell.input_list is None:
                        modulator = 0
                    else:
                        modulator = 0
                        AttributeError("Illegal cell input")

                    if operation == 'fm':
                        cell.signal = synth_module.batch_oscillator_fm(amp_c=1.0,
                                                                       freq_c=cell.parameters['freq_c'],
                                                                       waveform=cell.parameters['waveform'],
                                                                       mod_index=cell.parameters['mod_index'],
                                                                       modulator=modulator)
                    else:
                        if operation == 'fm_sine':
                            waveform = 'sine'
                        elif operation == 'fm_square':
                            waveform = 'square'
                        elif operation == 'fm_saw':
                            waveform = 'sawtooth'
                        else:
                            ValueError("Unsupported waveform")
                        cell.signal = synth_module.batch_specific_waveform_oscillator_fm(amp_c=1.0,
                                                                                         freq_c=cell.parameters[
                                                                                             'freq_c'],
                                                                                         waveform=waveform,
                                                                                         mod_index=cell.parameters[
                                                                                             'mod_index'],
                                                                                         modulator=modulator)


                elif operation == 'mix':
                    signal = 0
                    num_inputs = len(cell.input_list)
                    for input_num in range(num_inputs):
                        input_index = cell.input_list[input_num]
                        input_sound = self.architecture[input_index[0]][input_index[1]].signal
                        signal += input_sound

                    cell.signal = signal / num_inputs

                elif operation == 'amplitude_shape':
                    if len(cell.input_list) == 1:
                        input_cell_index = cell.input_list[0]
                        input_cell = self.architecture[input_cell_index[0]][input_cell_index[1]]
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
                                                             num_sounds_)
                    else:
                        envelope_shape = envelope

                    # envelope_shape = cell.parameters['envelope']
                    # plt.plot(envelope_shape[1].detach().numpy())
                    # plt.show()
                    cell.signal = synth_module.amplitude_envelope(input_signal, envelope_shape)

                elif operation == 'filter_shape':
                    if len(cell.input_list) == 1:
                        input_cell_index = cell.input_list[0]
                        input_cell = self.architecture[input_cell_index[0]][input_cell_index[1]]
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
                        input_cell = self.architecture[input_cell_index[0]][input_cell_index[1]]
                        input_signal = input_cell.signal
                    else:
                        input_signal = 0
                        AttributeError("Illegal cell input")
                    cell.signal = synth_module.batch_filter(input_signal,
                                                            filter_freq=cell.parameters['filter_freq'],
                                                            filter_type=cell.parameters['filter_type'])

                elif operation == 'lowpass_filter':
                    if len(cell.input_list) == 1:
                        input_cell_index = cell.input_list[0]
                        input_cell = self.architecture[input_cell_index[0]][input_cell_index[1]]
                        input_signal = input_cell.signal
                    else:
                        input_signal = 0
                        AttributeError("Illegal cell input")
                    cell.signal = synth_module.lowpass_batch_filter(input_signal,
                                                                    filter_freq=cell.parameters['filter_freq'],
                                                                    resonance=cell.parameters['resonance'])

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
                                                             num_sounds=num_sounds_)
                    # cell.signal = synth_module.batch_adsr_envelope(input_signal,
                    #                                                attack_t=cell.parameters['attack_t'],
                    #                                                decay_t=cell.parameters['decay_t'],
                    #                                                sustain_t=cell.parameters['sustain_t'],
                    #                                                sustain_level=cell.parameters['sustain_level'],
                    #                                                release_t=cell.parameters['release_t'])
                output_signals[f"({channel}, {layer})"] = cell.signal

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
        return final_signal, output_signals

    @staticmethod
    def get_preset(preset: str, synth_cfg: SynthConfig):

        preset_list = synth_modular_presets.synth_presets_dict.get(preset, None)
        if preset_list is None:
            ValueError("Unknown PRESET")

        preset_as_synth_input = [SynthModularCell(index=cell.get('index'),
                                                  input_list=cell.get('input_list'),
                                                  operation=cell.get('operation'),
                                                  parameters=cell.get('parameters'),
                                                  signal=cell.get('signal'),
                                                  output_list=cell.get('output_list'),
                                                  default_connection=cell.get('default_connection'),
                                                  num_channels=synth_cfg.num_channels,
                                                  num_layers=synth_cfg.num_layers,
                                                  synth_config=cell.get('synth_config'))
                                 for cell in preset_list]

        return preset_as_synth_input

    def collect_params(self):
        params_dict = {}
        for layer in range(self.num_layers):
            for channel in range(self.num_channels):
                cell = self.architecture[channel][layer]
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

    @staticmethod
    def _sample_c_freq(synth_cfg: SynthConfig, num_sounds_: int):

        osc_freq_list = np.asarray(synth_cfg.osc_freq_list)

        base_freqs = np.random.uniform(low=synth_cfg.osc_freq_list[0],
                                       high=synth_cfg.osc_freq_list[-1],
                                       size=num_sounds_)

        idx = np.searchsorted(synth_cfg.osc_freq_list, base_freqs, side="left")
        idx = idx - (np.abs(base_freqs - osc_freq_list[idx - 1]) < np.abs(base_freqs - osc_freq_list[idx]))
        return osc_freq_list[idx]


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
    num_sounds_ = 10
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
