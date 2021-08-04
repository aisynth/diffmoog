import synth
from synth import Signal
import random
import matplotlib.pyplot as plt
import simpleaudio as sa

PI = 3.141592653589793


class SynthBasicFlow:
    def __init__(self):
        wave_list = ['sine', 'square', 'triangle', 'sawtooth']
        filter_type_list = ['high_pass', 'low_pass', 'band_pass']
        middle_c_freq = 261.6255653005985

        # init synthesizer parameters
        osc1_semitones_offset = random.randrange(-24, 25, 1)
        osc1_amp = random.random()
        osc1_freq = middle_c_freq * (2 ** (1 / 12)) ** osc1_semitones_offset
        osc1_waveform = random.choice(wave_list)
        osc1_mod_index = random.uniform(0, 100)
        lfo1_freq = random.uniform(0, 20)
        lfo1_phase = random.uniform(0, 2 * PI)
        lfo1_waveform = random.choice(wave_list)

        osc2_semitones_offset = random.randrange(-24, 25, 1)
        osc2_amp = random.random()
        osc2_freq = middle_c_freq * (2 ** (1 / 12)) ** osc2_semitones_offset
        osc2_waveform = random.choice(wave_list)
        osc2_mod_index = random.uniform(0, 100)
        lfo2_freq = random.uniform(0, 20)
        lfo2_phase = random.uniform(0, 2 * PI)
        lfo2_waveform = random.choice(wave_list)

        filter_type = random.choice(filter_type_list)
        filter_frequency = random.uniform(20, 20000)

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
        lfo1.oscillator(amp=1, freq=lfo1_freq, phase=lfo1_phase, waveform=lfo1_waveform)
        oscillator1 = Signal()
        oscillator1.fm_modulation_by_input_signal(input_signal=lfo1.signal,
                                                  amp_c=osc1_amp,
                                                  freq_c=osc1_freq,
                                                  mod_index=osc1_mod_index,
                                                  waveform=osc1_waveform)

        lfo2 = Signal()
        lfo2.oscillator(amp=1, freq=lfo2_freq, phase=lfo2_phase, waveform=lfo2_waveform)
        oscillator2 = Signal()
        oscillator2.fm_modulation_by_input_signal(input_signal=lfo2.signal,
                                                  amp_c=osc2_amp,
                                                  freq_c=osc2_freq,
                                                  mod_index=osc2_mod_index,
                                                  waveform=osc2_waveform)

        audio = Signal()
        audio.signal = (oscillator1.signal + oscillator2.signal) / 2

        if filter_type == 'high_pass':
            audio.high_pass(filter_frequency)
        elif filter_type == 'low_pass':
            audio.low_pass(filter_frequency)
        elif filter_type == "band_pass":
            audio.band_pass(filter_frequency)

        audio.adsr_envelope(attack_t, decay_t, sustain_t, sustain_level, release_t)

        self.audio = audio.signal


a = SynthBasicFlow()
plt.plot(a.audio)
plt.show
play_obj = sa.play_buffer(a.audio.numpy(), num_channels=1, bytes_per_sample=4, sample_rate=44100)
play_obj.wait_done()