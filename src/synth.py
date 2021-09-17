#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 31 15:41:38 2021

@author: moshelaufer, Noy Uzrad
"""
import torch
import torchaudio.functional as taF
from src.config import PI, TWO_PI, DEBUG_MODE, SAMPLE_RATE, SIGNAL_DURATION_SEC
import matplotlib.pyplot as plt
import simpleaudio as sa
import helper

CLASSIFICATION_PARAM_LIST = \
    ['osc1_freq', 'osc1_wave', 'lfo1_wave', 'osc2_freq', 'osc2_wave', 'lfo2_wave', 'filter_type']
REGRESSION_PARAM_LIST = \
    ['osc1_amp', 'osc1_mod_index', 'lfo1_freq', 'lfo1_phase',
     'osc2_amp', 'osc2_mod_index', 'lfo2_freq', 'lfo2_phase',
     'filter_freq', 'attack_t', 'decay_t', 'sustain_t', 'release_t', 'sustain_level']
PARAM_LIST = [CLASSIFICATION_PARAM_LIST, REGRESSION_PARAM_LIST]

WAVE_TYPE_DIC = {"sine": 0,
                 "square": 1,
                 "triangle": 2,
                 "sawtooth": 3}
WAVE_TYPE_DIC_INV = {v: k for k, v in WAVE_TYPE_DIC.items()}

FILTER_TYPE_DIC = {"low_pass": 0,
                   "high_pass": 1,
                   "band_pass": 2}
FILTER_TYPE_DIC_INV = {v: k for k, v in FILTER_TYPE_DIC.items()}

# build a list of possible frequencies
SEMITONES_MAX_OFFSET = 24
MIDDLE_C_FREQ = 261.6255653005985
semitones_list = [*range(-SEMITONES_MAX_OFFSET, SEMITONES_MAX_OFFSET + 1)]
OSC_FREQ_LIST = [MIDDLE_C_FREQ * (2 ** (1 / 12)) ** x for x in semitones_list]
OSC_FREQ_DIC = {round(key, 4): value for value, key in enumerate(OSC_FREQ_LIST)}
OSC_FREQ_DIC_INV = {v: k for k, v in OSC_FREQ_DIC.items()}

MAX_AMP = 1
MAX_MOD_INDEX = 100
MAX_LFO_FREQ = 20
MIN_FILTER_FREQ = 20
MAX_FILTER_FREQ = 20000


class Signal:
    def __init__(self):
        self.sample_rate = SAMPLE_RATE
        self.sig_duration = SIGNAL_DURATION_SEC
        self.time_samples = torch.linspace(0, self.sig_duration, steps=int(self.sample_rate * self.sig_duration))
        self.modulation_time = torch.linspace(0, self.sig_duration, steps=self.sample_rate)
        self.modulation = 0
        self.signal = torch.zeros(self.time_samples.shape, dtype=torch.float32)

        self.device = helper.get_device()
        self.time_samples = helper.move_to(self.time_samples, self.device)
        self.modulation_time = helper.move_to(self.modulation_time, self.device)
        self.signal = helper.move_to(self.signal, self.device)
        # self.room_impulse_responses = torch.load('rir_for_reverb_no_amp')

    def oscillator(self, amp, freq, phase, waveform):
        """Creates a basic oscillator.

            Retrieves a waveform shape and attributes, and construct the respected signal

            Args:
                self: Self object
                amp: Amplitude in range [0, 1]
                freq: Frequency in range [0, 22000]
                phase: Phase in range [0, 2pi]
                waveform: one of [sine, square, triangle, sawtooth]

            Returns:
                A torch with the constructed signal

            Raises:
                ValueError: Provided variables are out of range
            """
        self.signal_values_sanity_check(amp, freq, waveform)
        t = self.time_samples
        phase = phase % TWO_PI
        oscillator = torch.zeros_like(t)
        if waveform == 'sine':
            oscillator = amp * torch.sin(TWO_PI * freq * t + phase)
        elif waveform == 'square':
            oscillator = amp * torch.sign(torch.sin(TWO_PI * freq * t + phase))
        elif waveform == 'triangle':
            oscillator = (2 * amp / PI) * torch.arcsin(torch.sin((TWO_PI * freq * t + phase)))
        elif waveform == 'sawtooth':
            # Sawtooth closed form
            oscillator = 2 * (t * freq - torch.floor(0.5 + t * freq))
            # Phase shift by normalization to range [0,1] and modulo operation
            oscillator = (((oscillator + 1) / 2) + phase / TWO_PI) % 1
            # re-normalization to range [-amp, amp]
            oscillator = amp * (oscillator * 2 - 1)

        self.signal = oscillator

    def mix_signal(self, new_signal, factor):
        """Signal superposition. factor balances the mix
        0 - original signal only, 1 - new signal only, 0.5 evenly balanced. """
        if factor < 0 or factor > 1:
            raise ValueError("Provided factor value is out of range [0, 1]")
        self.signal = factor * self.signal + (1 - factor) * new_signal

    def fm_modulation_by_input_signal(self, input_signal, amp_c, freq_c, mod_index, waveform):
        """FM modulation

            Modulates the frequency of a signal with the given properties, with an input signal as modulator

            Args:
                self: Self object
                input_signal: Modulator signal, to affect carrier frequency
                amp_c: Amplitude in range [0, 1]
                freq_c: Frequency in range [0, 22000]
                mod_index: Modulation index, which affects the amount of modulation
                waveform: One of [sine, square, triangle, sawtooth]

            Returns:
                A torch with the constructed FM signal

            Raises:
                ValueError: Provided variables are out of range
            """
        self.signal_values_sanity_check(amp_c, freq_c, waveform)
        t = self.time_samples
        if waveform == 'sine':
            self.signal = amp_c * torch.sin(TWO_PI * freq_c * t + mod_index * input_signal)
        if waveform == 'square':
            self.signal = amp_c * torch.sign(torch.sin(TWO_PI * freq_c * t + mod_index * input_signal))
        if waveform == 'triangle':
            self.signal = (2 * amp_c / PI) * torch.arcsin(torch.sin((TWO_PI * freq_c * t + mod_index * input_signal)))
        if waveform == 'sawtooth':
            oscillator = 2 * (t * freq_c - torch.floor(0.5 + t * freq_c))
            oscillator = (((oscillator + 1) / 2) + mod_index * input_signal / TWO_PI) % 1
            self.signal = amp_c * (oscillator * 2 - 1)

    def am_modulation(self, amp_c, freq_c, amp_m, freq_m, final_max_amp, waveform):
        """AM modulation

            Modulates the amplitude of a carrier signal with a sine modulator
            see https://en.wikipedia.org/wiki/Amplitude_modulation

            Args:
                self: Self object
                amp_c: Amplitude of carrier in range [0, 1]
                freq_c: Frequency of carrier in range [0, 22000]
                amp_m: Amplitude of modulator in range [0, 1]
                freq_m: Frequency of modulator in range [0, 22000]
                final_max_amp: The final maximum amplitude of the modulated signal
                waveform: One of [sine, square, triangle, sawtooth]

            Returns:
                A torch with the constructed AM signal

            Raises:
                ValueError: Provided variables are out of range
                ValueError: modulation index > 1. Amplitude values must obey amp_m < amp_c
                # todo add documentation for sensible frequency values
            """
        self.signal_values_sanity_check(amp_m, freq_m, waveform)
        self.signal_values_sanity_check(amp_c, freq_c, waveform)
        modulation_index = amp_m / amp_c
        if modulation_index > 1:
            raise ValueError("Provided amplitudes results modulation index > 1, and yields over-modulation ")
        if final_max_amp < 0 or final_max_amp > 1:
            raise ValueError("Provided final max amplitude is not in range [0, 1]")
        # todo: add restriction freq_c >> freq_m

        t = self.time_samples
        dc = 1
        carrier = Signal()
        carrier.oscillator(amp=amp_c, freq=freq_c, phase=0, waveform=waveform)
        modulator = amp_m * torch.sin(TWO_PI * freq_m * t)
        am_signal = (dc + modulator / amp_c) * carrier.signal
        normalized_am_signal = (final_max_amp / (amp_c + amp_m)) * am_signal
        self.signal = normalized_am_signal

    def am_modulation_by_input_signal(self, input_signal, modulation_factor, amp_c, freq_c, waveform):
        """AM modulation by an input signal

            Modulates the amplitude of a carrier signal with a provided input signal
            see https://en.wikipedia.org/wiki/Amplitude_modulation

            Args:
                self: Self object
                input_signal: Input signal to be used as modulator
                modulation_factor: factor to be multiplied by modulator, in range [0, 1]
                amp_c: Amplitude of carrier in range [0, 1]
                freq_c: Frequency of carrier in range [0, 22000]
                waveform: Waveform of carrier. One of [sine, square, triangle, sawtooth]

            Returns:
                A torch with the constructed AM signal

            Raises:
                ValueError: Provided variables are inappropriate
                ValueError: Resulted Amplitude is out of range [-1, 1]
            """
        self.signal_values_sanity_check(amp_c, freq_c, waveform)
        t = self.time_samples
        carrier = Signal()
        carrier.oscillator(amp=1, freq=freq_c, phase=0, waveform=waveform)
        modulated_amplitude = (amp_c + modulation_factor * input_signal)
        if torch.max(modulated_amplitude).item() > 1 or torch.min(modulated_amplitude).item() < -1:
            raise ValueError("AM modulation resulted amplitude out of range [-1, 1].")
        self.signal = modulated_amplitude * carrier.signal

    def adsr_envelope(self, attack_t, decay_t, sustain_t, sustain_level, release_t):
        """Apply an ADSR envelope to the signal

            builds the ADSR shape and multiply by the signal

            Args:
                self: Self object
                attack_t: Length of attack in seconds. Time to go from 0 to 1 amplitude.
                decay_t: Length of decay in seconds. Time to go from 1 amplitude to sustain level.
                sustain_t: Length of sustain in seconds, with sustain level amplitude
                sustain_level: Sustain volume level
                release_t: Length of release in seconds. Time to go ftom sustain level to 0 amplitude

            Raises:
                ValueError: Provided ADSR timings are not the same as the signal length
            """
        if attack_t + decay_t + sustain_t + release_t > self.sig_duration:
            raise ValueError("Provided ADSR durations exceeds signal duration")

        attack_num_samples = int(self.sample_rate * attack_t)
        decay_num_samples = int(self.sample_rate * decay_t)
        sustain_num_samples = int(self.sample_rate * sustain_t)
        release_num_samples = int(self.sample_rate * release_t)

        attack = torch.linspace(0, 1, attack_num_samples)
        decay = torch.linspace(1, sustain_level, decay_num_samples)
        sustain = torch.full((sustain_num_samples,), sustain_level)
        release = torch.linspace(sustain_level, 0, release_num_samples)

        envelope = torch.cat((attack, decay, sustain, release))
        envelope_len = envelope.shape[0]
        signal_len = self.time_samples.shape[0]
        if envelope_len <= signal_len:
            padding = torch.zeros(signal_len - envelope_len)
            envelope = torch.cat((envelope, padding))
        else:
            raise ValueError("Envelope length exceeds signal duration")

        envelope = helper.move_to(envelope, self.device)

        self.signal = self.signal * envelope

        if DEBUG_MODE:
            plt.plot(envelope.cpu())
            plt.plot(self.signal.cpu())
            plt.show()

    def low_pass(self, cutoff_freq, q=0.707):
        self.signal = taF.lowpass_biquad(self.signal, self.sample_rate, cutoff_freq, q)

    def high_pass(self, cutoff_freq, q=0.707):
        self.signal = taF.highpass_biquad(self.signal, self.sample_rate, cutoff_freq, q)

    def band_pass(self, central_freq, q=0.707, const_skirt_gain=False):
        self.signal = \
            taF.bandpass_biquad(self.signal, self.sample_rate, central_freq, q, const_skirt_gain)

    @staticmethod
    def signal_values_sanity_check(amp, freq, waveform):
        """Check signal properties are reasonable."""
        if freq < 0 or freq > 20000:
            raise ValueError("Provided frequency is not in range [0, 20000]")
        if amp < 0 or amp > 1:
            raise ValueError("Provided amplitude is not in range [0, 1]")
        if not any(x == waveform for x in ['sine', 'square', 'triangle', 'sawtooth']):
            raise ValueError("Unknown waveform provided")


"""
Reverb implementation - Currently not working as expected
So it is not used

    def reverb(self, size, dry_wet):
        if size not in [1, 2, 3, 4, 5, 6]:
            raise ValueError("reverb size must be an int in range [1, 6]")
        if dry_wet < 0 or dry_wet > 1:
            raise ValueError("Provided dry/wet value is out of range [0, 1]")

        if size == 1:
            response_name = 'rir0'
        elif size == 2:
            response_name = 'rir20'
        elif size == 3:
            response_name = 'rir40'
        elif size == 4:
            response_name = 'rir60'
        elif size == 5:
            response_name = 'rir80'
        elif size == 6:
            response_name = 'rir100'
        else:
            response_name = 'rir0'

        signal_clean = self.signal
        room_impulse_response = self.room_impulse_responses[response_name]

        # room_response_start_index = 0
        # room_response_end_index = room_impulse_response.size()[0] - 1
        # signal_start_index = 0
        # signal_end_index = signal_clean.size()[0] - 1
        # while room_impulse_response[room_response_start_index] == 0:
        #     room_response_start_index = room_response_start_index+1
        # while room_impulse_response[room_response_end_index] == 0:
        #     room_response_end_index = room_response_end_index-1
        # while signal_clean[signal_start_index] == 0:
        #     signal_start_index = signal_start_index+1
        # while signal_clean[signal_end_index] == 0:
        #     signal_end_index = signal_end_index-1
        # print(room_impulse_response.shape)
        # print(signal_clean.shape)
        # signal_clean = signal_clean[signal_start_index:signal_end_index]
        # room_impulse_response = room_impulse_response[room_response_start_index:room_response_end_index]

        kernel_size = room_impulse_response.shape[0]
        signal_size = signal_clean.shape[0]

        print("kersize", kernel_size)
        print("sigsize", signal_size)
        print(room_impulse_response.shape)
        print(signal_clean.shape)

        room_impulse_response = room_impulse_response.unsqueeze(0).unsqueeze(0)
        signal_clean = signal_clean.unsqueeze(0).unsqueeze(0)

        signal_reverb = F.conv1d(signal_clean, room_impulse_response, bias=None, stride=1)

        signal_reverb = torch.squeeze(signal_reverb)
        signal_clean = torch.squeeze(signal_clean)

        print("sig rev shape", signal_reverb.shape)
        print(signal_reverb.shape[0])

        if signal_reverb.shape[0] > signal_clean.shape[0]:
            padding = torch.zeros(signal_reverb.shape[0] - signal_clean.shape[0])
            signal_clean = torch.cat((signal_clean, padding))
        else:
            padding = torch.zeros(signal_clean.shape[0] - signal_reverb.shape[0])
            signal_reverb = torch.cat((signal_reverb, padding))

        self.signal = dry_wet * signal_reverb + (1 - dry_wet) * signal_clean
        self.signal = torch.squeeze(self.signal)
"""

if __name__ == "__main__":
    a = Signal()
    b = Signal()
    b.oscillator(1, 5, 0, 'sine')
    a.fm_modulation_by_input_signal(b.signal, 1, 440, 10, 'sine')
    # a.oscillator(amp=1, freq=100, phase=0, waveform='sine')
    # a.adsr_envelope(attack_t=0.5, decay_t=0, sustain_t=0.5, sustain_level=0.5, release_t=0)
    # plt.plot(a.signal)
    # plt.show
    play_obj = sa.play_buffer(a.signal.numpy(), num_channels=1, bytes_per_sample=4, sample_rate=a.sample_rate)
    # play_obj.wait_done()
    # b = Signal()
    # a.am_modulation(amp_c=1, freq_c=4, amp_m=0.3, freq_m=0, final_max_amp=0.5, waveform='sine')
    # b.am_modulation_by_input_signal(a.data, modulation_factor=1, amp_c=0.5, freq_c=40, waveform='triangle')
    plt.plot(a.signal)
    # plt.plot(b.data)
    # torch.tensor(0)
    # print(torch.sign(torch.tensor(0)))
    # # b.fm_modulation_by_input_signal(a.data, 440, 1, 10, 'sawtooth')
    # # plt.plot(b.data)
    # plt.show()
    #
    # # plt.plot(a.data)
    # a.low_pass(1000)
    # play_obj = sa.play_buffer(b.data.numpy(), 1, 4, b.sample_rate)
    # play_obj.wait_done()
    # plt.plot(a.data)
    #
    # def fm_modulation(self, amp_mod, fm, fc, Ac, waveform):
    # a.fm_modulation(1, 3, 5, 1, 'tri')
    # print(a.data)
    # plt.plot(a.data)
    #
    # plt.show()
