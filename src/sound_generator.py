import matplotlib.pyplot as plt
import numpy
from config import PI, PLOT_SPEC
from src.config import SIGNAL_DURATION_SEC
from synth_config import OSC_FREQ_LIST
import synth
import random
import simpleaudio as sa
import numpy as np
import torch
from torch import nn
import helper
from synth import Synth


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
        self.params_dict['osc1_freq'] = random.choices(synth.OSC_FREQ_LIST, k=num_sounds)
        self.params_dict['osc1_wave'] = random.choices(list(synth.WAVE_TYPE_DIC), k=num_sounds)
        self.params_dict['osc1_mod_index'] = np.random.uniform(low=0, high=synth.MAX_MOD_INDEX, size=num_sounds)
        self.params_dict['lfo1_freq'] = np.random.uniform(low=0, high=synth.MAX_LFO_FREQ, size=num_sounds)

        self.params_dict['osc2_amp'] = np.random.random_sample(size=num_sounds)
        self.params_dict['osc2_freq'] = random.choices(synth.OSC_FREQ_LIST, k=num_sounds)
        self.params_dict['osc2_wave'] = random.choices(list(synth.WAVE_TYPE_DIC), k=num_sounds)
        self.params_dict['osc2_mod_index'] = np.random.uniform(low=0, high=synth.MAX_MOD_INDEX, size=num_sounds)
        self.params_dict['lfo2_freq'] = np.random.uniform(low=0, high=synth.MAX_LFO_FREQ, size=num_sounds)

        self.params_dict['filter_type'] = random.choices(list(synth.FILTER_TYPE_DIC), k=num_sounds)
        self.params_dict['filter_freq'] = \
            np.random.uniform(low=synth.MIN_FILTER_FREQ, high=synth.MAX_FILTER_FREQ, size=num_sounds)

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

        synth = Synth(num_sounds)

        lfo1 = synth.oscillator(amp=1,
                                freq=lfo1_freq,
                                phase=0,
                                waveform='sine',
                                num_sounds=num_sounds)

        fm_osc1 = synth.fm_modulation_by_input_signal(input_signal=lfo1,
                                                      amp_c=osc1_amp,
                                                      freq_c=osc1_freq,
                                                      mod_index=osc1_mod_index,
                                                      waveform=osc1_wave,
                                                      num_sounds=num_sounds)

        lfo2 = synth.oscillator(amp=1,
                                freq=lfo2_freq,
                                phase=0,
                                waveform='sine',
                                num_sounds=num_sounds)

        fm_osc2 = synth.fm_modulation_by_input_signal(input_signal=lfo2,
                                                      amp_c=osc2_amp,
                                                      freq_c=osc2_freq,
                                                      mod_index=osc2_mod_index,
                                                      waveform=osc2_wave,
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

        self.params_dict['osc1_freq'] = random.choices(synth.OSC_FREQ_LIST, k=num_sounds)

        for key, val in self.params_dict.items():
            if isinstance(val, numpy.ndarray):
                self.params_dict[key] = val.tolist()

        if num_sounds == 1:
            for key, value in self.params_dict.items():
                self.params_dict[key] = value[0]

    def generate_signal(self, num_sounds):
        osc_freq = self.params_dict['osc1_freq']

        synthesizer = Synth(num_sounds)

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
    for i in range(num_sounds):
        play_obj = sa.play_buffer(a.signal[i].detach().cpu().numpy(),
                                  num_channels=1,
                                  bytes_per_sample=4,
                                  sample_rate=44100)
        play_obj.wait_done()
