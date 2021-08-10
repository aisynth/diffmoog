from synth import Signal
from config import PI, SEMITONES_MAX_OFFSET, OSC_FREQ_LIST
import synth
import random
import matplotlib.pyplot as plt
import simpleaudio as sa


class SynthBasicFlow:
    def __init__(self, file_name):
        wave_list = ['sine', 'square', 'triangle', 'sawtooth']
        filter_type_list = ['high_pass', 'low_pass', 'band_pass']
        parameters_list = ['file_name',
                           'osc1_amp', 'osc1_freq', 'osc1_wave', 'osc1_mod_index',
                           'lfo1_freq', 'lfo1_phase', 'lfo1_wave',
                           'osc2_amp', 'osc2_freq', 'osc2_wave', 'osc2_mod_index',
                           'lfo2_freq', 'lfo2_phase', 'lfo2_wave',
                           'filter_type', 'filter_freq',
                           'attack_t', 'decay_t', 'sustain_t', 'release_t', 'sustain_level']
        parameters_dict = {}


        # init synthesizer parameters
        osc1_index = random.randrange(-SEMITONES_MAX_OFFSET, SEMITONES_MAX_OFFSET, 1) + SEMITONES_MAX_OFFSET
        osc1_amp = random.random()
        osc1_freq = OSC_FREQ_LIST[osc1_index]
        osc1_wave = random.choice(wave_list)
        osc1_mod_index = random.uniform(0, 100)
        lfo1_freq = random.uniform(0, 20)
        lfo1_phase = random.uniform(0, 2 * PI)
        lfo1_wave = random.choice(wave_list)

        osc2_index = random.randrange(-SEMITONES_MAX_OFFSET, SEMITONES_MAX_OFFSET, 1) + SEMITONES_MAX_OFFSET
        osc2_amp = random.random()
        osc2_freq = OSC_FREQ_LIST[osc2_index]
        osc2_wave = random.choice(wave_list)
        osc2_mod_index = random.uniform(0, 100)
        lfo2_freq = random.uniform(0, 20)
        lfo2_phase = random.uniform(0, 2 * PI)
        lfo2_wave= random.choice(wave_list)

        filter_type = random.choice(filter_type_list)
        filter_freq = random.uniform(20, 20000)

        attack_t = random.random()
        decay_t = random.random()
        sustain_t = random.random()
        release_t = random.random()
        adsr_sum = attack_t + decay_t + sustain_t + release_t
        attack_t = attack_t / adsr_sum
        decay_t = decay_t / adsr_sum
        sustain_t = sustain_t / adsr_sum
        release_t = release_t / adsr_sum
        sustain_level = random.random()

        # fixing a numerical issue in case the ADSR times exceeds signal length
        adsr_aggregated_time = attack_t + decay_t + sustain_t + release_t
        if adsr_aggregated_time > synth.SIGNAL_DURATION_SEC:
            sustain_t = sustain_t - 0.001

        # generate signal with basic signal flow
        lfo1 = Signal()
        lfo1.oscillator(amp=1, freq=lfo1_freq, phase=lfo1_phase, waveform=lfo1_wave)
        oscillator1 = Signal()
        oscillator1.fm_modulation_by_input_signal(input_signal=lfo1.signal,
                                                  amp_c=osc1_amp,
                                                  freq_c=osc1_freq,
                                                  mod_index=osc1_mod_index,
                                                  waveform=osc1_wave)

        lfo2 = Signal()
        lfo2.oscillator(amp=1, freq=lfo2_freq, phase=lfo2_phase, waveform=lfo2_wave)
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

        for variable in parameters_list:
            parameters_dict[variable] = eval(variable)

        self.file_name = file_name
        self.synth_params_dict = parameters_dict
        self.audio = audio.signal


if __name__ == "__main__":
    a = SynthBasicFlow('audio_example')
    plt.plot(a.audio)
    plt.show()
    play_obj = sa.play_buffer(a.audio.numpy(), num_channels=1, bytes_per_sample=4, sample_rate=44100)
    play_obj.wait_done()
