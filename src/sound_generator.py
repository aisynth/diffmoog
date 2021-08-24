from synth import Signal
from src.config import PI
import synth
import random
import matplotlib.pyplot as plt
import simpleaudio as sa


# todo: refactor code. use general dictionaries. rearrange self variables.
#  use loops and prevent code duplication
class SynthBasicFlow:
    def __init__(self, file_name='unnamed_sound', parameters_dict=None):
        self.file_name = file_name
        self.params_dict = {}
        # init parameters_dict
        if parameters_dict is None:
            self.init_random_synth_params()
        elif type(parameters_dict) is dict:
            self.params_dict = parameters_dict
        else:
            ValueError("Provided parameters are not provided as dictionary")

        # generate signal with basic signal flow
        self.signal = self.generate_signal()

    def init_random_synth_params(self):
        osc1_freq_index = random.randrange(0, 2 * synth.SEMITONES_MAX_OFFSET + 1)
        self.params_dict['osc1_amp'] = random.random()
        self.params_dict['osc1_freq'] = synth.OSC_FREQ_LIST[osc1_freq_index]
        self.params_dict['osc1_wave'] = random.choice(list(synth.WAVE_TYPE_DIC))
        self.params_dict['osc1_mod_index'] = random.uniform(0, synth.MAX_MOD_INDEX)
        self.params_dict['lfo1_freq'] = random.uniform(0, synth.MAX_LFO_FREQ)
        self.params_dict['lfo1_phase'] = random.uniform(0, 2 * PI)
        self.params_dict['lfo1_wave'] = random.choice(list(synth.WAVE_TYPE_DIC))

        osc2_freq_index = random.randrange(0, 2 * synth.SEMITONES_MAX_OFFSET + 1)
        self.params_dict['osc2_amp'] = random.random()
        self.params_dict['osc2_freq'] = synth.OSC_FREQ_LIST[osc2_freq_index]
        self.params_dict['osc2_wave'] = random.choice(list(synth.WAVE_TYPE_DIC))
        self.params_dict['osc2_mod_index'] = random.uniform(0, synth.MAX_MOD_INDEX)
        self.params_dict['lfo2_freq'] = random.uniform(0, synth.MAX_LFO_FREQ)
        self.params_dict['lfo2_phase'] = random.uniform(0, 2 * PI)
        self.params_dict['lfo2_wave'] = random.choice(list(synth.WAVE_TYPE_DIC))

        self.params_dict['filter_type'] = random.choice(list(synth.FILTER_TYPE_DIC))
        self.params_dict['filter_freq'] = random.uniform(synth.MIN_FILTER_FREQ, synth.MAX_FILTER_FREQ)

        attack_t = random.random()
        decay_t = random.random()
        sustain_t = random.random()
        release_t = random.random()
        adsr_sum = attack_t + decay_t + sustain_t + release_t
        attack_t = attack_t / adsr_sum
        decay_t = decay_t / adsr_sum
        sustain_t = sustain_t / adsr_sum
        release_t = release_t / adsr_sum

        # fixing a numerical issue in case the ADSR times exceeds signal length
        adsr_aggregated_time = attack_t + decay_t + sustain_t + release_t
        if adsr_aggregated_time > synth.SIGNAL_DURATION_SEC:
            sustain_t = sustain_t - 0.001

        self.params_dict['attack_t'] = attack_t
        self.params_dict['decay_t'] = decay_t
        self.params_dict['sustain_t'] = sustain_t
        self.params_dict['release_t'] = release_t
        self.params_dict['sustain_level'] = random.random()

    def generate_signal(self):
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

        lfo1 = Signal()
        lfo1.oscillator(amp=1,
                        freq=lfo1_freq,
                        phase=lfo1_phase,
                        waveform=lfo1_wave)
        oscillator1 = Signal()
        oscillator1.fm_modulation_by_input_signal(input_signal=lfo1.signal,
                                                  amp_c=osc1_amp,
                                                  freq_c=osc1_freq,
                                                  mod_index=osc1_mod_index,
                                                  waveform=osc1_wave)

        lfo2 = Signal()
        lfo2.oscillator(amp=1,
                        freq=lfo2_freq,
                        phase=lfo2_phase,
                        waveform=lfo2_wave)
        oscillator2 = Signal()
        oscillator2.fm_modulation_by_input_signal(input_signal=lfo2.signal,
                                                  amp_c=osc2_amp,
                                                  freq_c=osc2_freq,
                                                  mod_index=osc2_mod_index,
                                                  waveform=osc2_wave)

        audio = Signal()
        audio.signal = (oscillator1.signal + oscillator2.signal) / 2

        if filter_type == 'high_pass':
            audio.high_pass(filter_freq)
        elif filter_type == 'low_pass':
            audio.low_pass(filter_freq)
        elif filter_type == "band_pass":
            audio.band_pass(filter_freq)

        audio.adsr_envelope(attack_t, decay_t, sustain_t, sustain_level, release_t)

        return audio.signal


if __name__ == "__main__":
    a = SynthBasicFlow('audio_example')
    plt.plot(a.signal)
    plt.ylim([-1,1])
    plt.show()
    play_obj = sa.play_buffer(a.signal.numpy(),
                              num_channels=1,
                              bytes_per_sample=4,
                              sample_rate=44100)
    play_obj.wait_done()
