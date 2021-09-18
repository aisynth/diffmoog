import numpy
from synth import Signal
from src.config import PI, SIGNAL_DURATION_SEC
import synth
import random
import matplotlib.pyplot as plt
import simpleaudio as sa
import numpy as np
import torch


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
            self.params_dict = parameters_dict
        else:
            ValueError("Provided parameters are not provided as dictionary")

        # generate signal with basic signal flow
        self.signal = self.generate_signal(num_sounds)

    def init_random_synth_params(self, num_sounds):
        """init params_dict with lists of parameters"""

        # osc1_freq_index = random.randrange(0, 2 * synth.SEMITONES_MAX_OFFSET + 1)
        self.params_dict['osc1_amp'] = np.random.random_sample(size=num_sounds)
        # self.params_dict['osc1_freq'] = synth.OSC_FREQ_LIST[osc1_freq_index]
        self.params_dict['osc1_freq'] = random.choices(synth.OSC_FREQ_LIST, k=num_sounds)
        self.params_dict['osc1_wave'] = random.choices(list(synth.WAVE_TYPE_DIC), k=num_sounds)
        self.params_dict['osc1_mod_index'] = np.random.uniform(low=0, high=synth.MAX_MOD_INDEX, size=num_sounds)
        self.params_dict['lfo1_freq'] = np.random.uniform(low=0, high=synth.MAX_LFO_FREQ, size=num_sounds)
        self.params_dict['lfo1_phase'] = np.random.uniform(low=0, high=2 * PI, size=num_sounds)
        self.params_dict['lfo1_wave'] = random.choices(list(synth.WAVE_TYPE_DIC), k=num_sounds)

        self.params_dict['osc2_amp'] = np.random.random_sample(size=num_sounds)
        self.params_dict['osc2_freq'] = random.choices(synth.OSC_FREQ_LIST, k=num_sounds)
        self.params_dict['osc2_wave'] = random.choices(list(synth.WAVE_TYPE_DIC), k=num_sounds)
        self.params_dict['osc2_mod_index'] = np.random.uniform(low=0, high=synth.MAX_MOD_INDEX, size=num_sounds)
        self.params_dict['lfo2_freq'] = np.random.uniform(low=0, high=synth.MAX_LFO_FREQ, size=num_sounds)
        self.params_dict['lfo2_phase'] = np.random.uniform(low=0, high=2 * PI, size=num_sounds)
        self.params_dict['lfo2_wave'] = random.choices(list(synth.WAVE_TYPE_DIC), k=num_sounds)

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

    def generate_signal(self, num_sounds):
        osc1_amp = self.params_dict['osc1_amp']
        osc1_freq = self.params_dict['osc1_freq']
        osc1_wave = self.params_dict['osc1_wave']
        osc1_mod_index = self.params_dict['osc1_mod_index']
        lfo1_freq = self.params_dict['lfo1_freq']
        lfo1_phase = self.params_dict['lfo1_phase']
        lfo1_wave = self.params_dict['lfo1_wave']

        osc2_amp = self.params_dict['osc2_amp']
        osc2_freq = self.params_dict['osc2_freq']
        osc2_wave = self.params_dict['osc2_wave']
        osc2_mod_index = self.params_dict['osc2_mod_index']
        lfo2_freq = self.params_dict['lfo2_freq']
        lfo2_phase = self.params_dict['lfo2_phase']
        lfo2_wave = self.params_dict['lfo2_wave']

        filter_type = self.params_dict['filter_type']
        filter_freq = self.params_dict['filter_freq']

        attack_t = self.params_dict['attack_t']
        decay_t = self.params_dict['decay_t']
        sustain_t = self.params_dict['sustain_t']
        release_t = self.params_dict['release_t']
        sustain_level = self.params_dict['sustain_level']

        lfo1 = Signal(num_sounds)
        lfo1.oscillator(amp=1,
                        freq=lfo1_freq,
                        phase=lfo1_phase,
                        waveform=lfo1_wave,
                        num_sounds=num_sounds)
        oscillator1 = Signal(num_sounds)
        oscillator1.fm_modulation_by_input_signal(input_signal=lfo1.signal,
                                                  amp_c=osc1_amp,
                                                  freq_c=osc1_freq,
                                                  mod_index=osc1_mod_index,
                                                  waveform=osc1_wave,
                                                  num_sounds=num_sounds)

        lfo2 = Signal(num_sounds)
        lfo2.oscillator(amp=1,
                        freq=lfo2_freq,
                        phase=lfo2_phase,
                        waveform=lfo2_wave,
                        num_sounds=num_sounds)
        oscillator2 = Signal(num_sounds)
        oscillator2.fm_modulation_by_input_signal(input_signal=lfo2.signal,
                                                  amp_c=osc2_amp,
                                                  freq_c=osc2_freq,
                                                  mod_index=osc2_mod_index,
                                                  waveform=osc2_wave,
                                                  num_sounds=num_sounds)

        audio = Signal(num_sounds)
        audio.signal = (oscillator1.signal + oscillator2.signal) / 2
        audio.signal = audio.signal.cpu()

        for i in range(num_sounds):
            if num_sounds == 1:
                filter_frequency = filter_freq[0]
            elif num_sounds > 1:
                if torch.is_tensor(filter_freq[i]):
                    filter_frequency = filter_freq[i].item()
                else:
                    filter_frequency = filter_freq[i]
            if filter_type[i] == 'high_pass':
                audio.high_pass(cutoff_freq=filter_frequency, index=i)
            elif filter_type[i] == 'low_pass':
                audio.low_pass(cutoff_freq=filter_frequency, index=i)
            elif filter_type[i] == "band_pass":
                audio.band_pass(central_freq=filter_frequency, index=i)

        audio.adsr_envelope(attack_t, decay_t, sustain_t, sustain_level, release_t, num_sounds)

        return audio.signal


if __name__ == "__main__":
    a = SynthBasicFlow('audio_example', num_sounds=3)
    # plt.plot(a.signal.cpu())
    # plt.ylim([-1, 1])
    # plt.show()
    play_obj = sa.play_buffer(a.signal.cpu().numpy(),
                              num_channels=1,
                              bytes_per_sample=4,
                              sample_rate=44100)
    play_obj.wait_done()
