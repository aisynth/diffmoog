from synth import Signal
from config import PI, SEMITONES_MAX_OFFSET, OSC_FREQ_LIST, WAVE_TYPE_DIC_INV, FILTER_TYPE_DIC_INV, OSC_FREQ_DIC_INV
import synth
import random
import matplotlib.pyplot as plt
import simpleaudio as sa


# todo: refactor code. use general general dictionaries. rearrange self variables.
#  use loops and prevent code duplication
class SynthBasicFlow:
    def __init__(self, file_name, params_dict=None):
        self.file_name = file_name
        self.wave_list = ['sine', 'square', 'triangle', 'sawtooth']
        self.filter_type_list = ['high_pass', 'low_pass', 'band_pass']
        self.parameters_list = ['file_name',
                                'osc1_amp', 'osc1_freq', 'osc1_wave', 'osc1_mod_index',
                                'lfo1_freq', 'lfo1_phase', 'lfo1_wave',
                                'osc2_amp', 'osc2_freq', 'osc2_wave', 'osc2_mod_index',
                                'lfo2_freq', 'lfo2_phase', 'lfo2_wave',
                                'filter_type', 'filter_freq',
                                'attack_t', 'decay_t', 'sustain_t', 'release_t', 'sustain_level']
        parameters_dict = {}

        self.osc1_amp = 0
        self.osc1_freq = 0
        self.osc1_wave = 0
        self.osc1_mod_index = 0
        self.lfo1_freq = 0
        self.lfo1_phase = 0
        self.lfo1_wave = 0

        self.osc2_amp = 0
        self.osc2_freq = 0
        self.osc2_wave = 0
        self.osc2_mod_index = 0
        self.lfo2_freq = 0
        self.lfo2_phase = 0
        self.lfo2_wave = 0

        self.filter_type = 0
        self.filter_freq = 0

        self.attack_t = 0
        self.decay_t = 0
        self.sustain_t = 0
        self.release_t = 0
        self.sustain_level = 0

        # init parameters dic by random values
        if params_dict is None:
            self.parameters_dict = {}
            self.init_random_synth_params()
        elif params_dict is dict:
            self.parameters_dict = params_dict
            self.init_synth_params_from_dictionary(params_dict)
        else:
            ValueError("Provided parameters are not provided as dictionary")

        # generate signal with basic signal flow
        audio = self.generate_signal()

        for variable in self.parameters_list:
            parameters_dict[variable] = eval(f"self.{variable}")

        self.synth_params_dict = parameters_dict
        self.audio = audio.signal


    def init_random_synth_params(self):
        # init synthesizer parameters
        # todo: define ranges of params in a different file
        osc1_index = random.randrange(-SEMITONES_MAX_OFFSET, SEMITONES_MAX_OFFSET, 1) + SEMITONES_MAX_OFFSET
        self.osc1_amp = random.random()
        self.osc1_freq = OSC_FREQ_LIST[osc1_index]
        self.osc1_wave = random.choice(self.wave_list)
        self.osc1_mod_index = random.uniform(0, 100)
        self.lfo1_freq = random.uniform(0, 20)
        self.lfo1_phase = random.uniform(0, 2 * PI)
        self.lfo1_wave = random.choice(self.wave_list)

        osc2_index = random.randrange(-SEMITONES_MAX_OFFSET, SEMITONES_MAX_OFFSET, 1) + SEMITONES_MAX_OFFSET
        self.osc2_amp = random.random()
        self.osc2_freq = OSC_FREQ_LIST[osc2_index]
        self.osc2_wave = random.choice(self.wave_list)
        self.osc2_mod_index = random.uniform(0, 100)
        self.lfo2_freq = random.uniform(0, 20)
        self.lfo2_phase = random.uniform(0, 2 * PI)
        self.lfo2_wave = random.choice(self.wave_list)

        self.filter_type = random.choice(self.filter_type_list)
        self.filter_freq = random.uniform(20, 20000)

        self.attack_t = random.random()
        self.decay_t = random.random()
        self.sustain_t = random.random()
        self.release_t = random.random()
        adsr_sum = self.attack_t + self.decay_t + self.sustain_t + self.release_t
        self.attack_t = self.attack_t / adsr_sum
        self.decay_t = self.decay_t / adsr_sum
        self.sustain_t = self.sustain_t / adsr_sum
        self.release_t = self.release_t / adsr_sum
        self.sustain_level = random.random()

        # fixing a numerical issue in case the ADSR times exceeds signal length
        adsr_aggregated_time = self.attack_t + self.decay_t + self.sustain_t + self.release_t
        if adsr_aggregated_time > synth.SIGNAL_DURATION_SEC:
            self.sustain_t = self.sustain_t - 0.001

        self.parameters_dict['osc1_amp'] = self.osc1_amp
        self.parameters_dict['osc1_freq'] = self.osc1_freq
        self.parameters_dict['osc1_wave'] = self.osc1_wave
        self.parameters_dict['osc1_mod_index'] = self.osc1_mod_index
        self.parameters_dict['lfo1_freq'] = self.lfo1_freq
        self.parameters_dict['lfo1_phase'] = self.lfo1_phase
        self.parameters_dict['lfo1_wave'] = self.lfo1_wave

        self.parameters_dict['osc2_amp'] = self.osc2_amp
        self.parameters_dict['osc2_freq'] = self.osc2_freq
        self.parameters_dict['osc2_wave'] = self.osc2_wave
        self.parameters_dict['osc2_mod_index'] = self.osc2_mod_index
        self.parameters_dict['lfo2_freq'] = self.lfo2_freq
        self.parameters_dict['lfo2_phase'] = self.lfo2_phase
        self.parameters_dict['lfo2_wave'] = self.lfo2_wave

        self.parameters_dict['filter_type'] = self.filter_type
        self.parameters_dict['filter_freq'] = self.filter_freq

        self.parameters_dict['attack_t'] = self.attack_t
        self.parameters_dict['decay_t'] = self.decay_t
        self.parameters_dict['sustain_t'] = self.sustain_t
        self.parameters_dict['release_t'] = self.release_t
        self.parameters_dict['sustain_level'] = self.sustain_level

    def init_synth_params_from_dictionary(self, params_dict):
        # init synthesizer parameters
        self.osc1_amp = params_dict['osc1_amp']
        self.osc1_freq = params_dict['osc1_freq']
        if params_dict['osc1_freq'] is float:
            self.osc1_freq = params_dict['osc1_freq']
        elif params_dict['osc1_freq'] is int:
            self.osc1_freq = OSC_FREQ_DIC_INV[params_dict['osc1_freq']]
        if params_dict['osc1_wave'] is str:
            self.osc1_wave = params_dict['osc1_wave']
        elif params_dict['osc1_wave'] is int:
            self.osc1_wave = WAVE_TYPE_DIC_INV[params_dict['osc1_wave']]
        self.osc1_mod_index = params_dict['osc1_mod_index']
        self.lfo1_freq = params_dict['lfo1_freq']
        self.lfo1_phase = params_dict['lfo1_phase']
        if params_dict['lfo1_wave'] is str:
            self.lfo1_wave = params_dict['lfo1_wave']
        elif params_dict['lfo1_wave'] is int:
            self.lfo1_wave = WAVE_TYPE_DIC_INV[params_dict['lfo1_wave']]

        self.osc2_amp = params_dict['osc2_amp']
        if params_dict['osc2_freq'] is float:
            self.osc2_freq = params_dict['osc2_freq']
        elif params_dict['osc2_freq'] is int:
            self.osc2_freq = OSC_FREQ_DIC_INV[params_dict['osc2_freq']]
        if params_dict['osc2_wave'] is str:
            self.osc2_wave = params_dict['osc2_wave']
        elif params_dict['osc2_wave'] is int:
            self.osc2_wave = WAVE_TYPE_DIC_INV[params_dict['osc2_wave']]
        self.osc2_mod_index = params_dict['osc2_mod_index']
        self.lfo2_freq = params_dict['lfo2_freq']
        self.lfo2_phase = params_dict['lfo2_phase']
        if params_dict['lfo2_wave'] is str:
            self.lfo2_wave = params_dict['lfo2_wave']
        elif params_dict['lfo2_wave'] is int:
            self.lfo2_wave = WAVE_TYPE_DIC_INV[params_dict['lfo2_wave']]

        if params_dict['filter_type'] is str:
            self.filter_type = params_dict['filter_type']
        elif params_dict['filter_type'] is int:
            self.filter_type = FILTER_TYPE_DIC_INV[params_dict['filter_type']]
        self.filter_freq = params_dict['filter_freq']

        self.attack_t = params_dict['attack_t']
        self.decay_t = params_dict['decay_t']
        self.sustain_t = params_dict['sustain_t']
        self.release_t = params_dict['release_t']
        self.sustain_level = params_dict['sustain_level']

    def generate_signal(self):
        lfo1 = Signal()
        lfo1.oscillator(amp=1, freq=self.lfo1_freq, phase=self.lfo1_phase, waveform=self.lfo1_wave)
        oscillator1 = Signal()
        oscillator1.fm_modulation_by_input_signal(input_signal=lfo1.signal,
                                                  amp_c=self.osc1_amp,
                                                  freq_c=self.osc1_freq,
                                                  mod_index=self.osc1_mod_index,
                                                  waveform=self.osc1_wave)

        lfo2 = Signal()
        lfo2.oscillator(amp=1, freq=self.lfo2_freq, phase=self.lfo2_phase, waveform=self.lfo2_wave)
        oscillator2 = Signal()
        oscillator2.fm_modulation_by_input_signal(input_signal=lfo2.signal,
                                                  amp_c=self.osc2_amp,
                                                  freq_c=self.osc2_freq,
                                                  mod_index=self.osc2_mod_index,
                                                  waveform=self.osc2_wave)

        audio = Signal()
        audio.signal = (oscillator1.signal + oscillator2.signal) / 2

        if self.filter_type == 'high_pass':
            audio.high_pass(self.filter_freq)
        elif self.filter_type == 'low_pass':
            audio.low_pass(self.filter_freq)
        elif self.filter_type == "band_pass":
            audio.band_pass(self.filter_freq)

        audio.adsr_envelope(self.attack_t, self.decay_t, self.sustain_t, self.sustain_level, self.release_t)

        return audio


if __name__ == "__main__":
    a = SynthBasicFlow('audio_example')
    plt.plot(a.audio)
    plt.show()
    play_obj = sa.play_buffer(a.audio.numpy(), num_channels=1, bytes_per_sample=4, sample_rate=44100)
    play_obj.wait_done()
