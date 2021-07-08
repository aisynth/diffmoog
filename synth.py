#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 31 15:41:38 2021

@author: moshelaufer, Noy Uzrad
"""
import numpy as np
from numpy import asarray, zeros, place, nan, mod, pi, extract
import torch.nn as nn
import torch
import torchaudio.functional as taF
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pytorch_forecasting
import simpleaudio as sa

PI = 3.141592653589793
TWO_PI = 2 * PI
SAMPLE_RATE = 44100
SIGNAL_DURATION_SEC = 1.0


# data = torchaudio.functional.compute_kaldi_pitch('sine',sample_rate =2200,frame_length=5000)

class Signal:
    def __init__(self):
        self.sample_rate = SAMPLE_RATE
        self.sig_duration = SIGNAL_DURATION_SEC
        self.time_samples = torch.linspace(0, self.sig_duration, steps=self.sample_rate)
        self.pi = torch.acos(torch.zeros(1).float()) * 2.0
        self.modulation_time = torch.linspace(0, self.sig_duration, steps=self.sample_rate)
        self.modulation = 0
        self.signal = 0
        self.shift = self.sig_duration / (self.sample_rate * 2)
        self.conv = nn.Conv1d(1, 1, 1, 1)
        torch.wave = 0

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
        self.signal = factor * self.signal + (1 - factor) * new_signal

    # todo: maybe delete this function. Has some inaccuracies. and the fm_modulation_for_input generalizes it
    '''Ac*sin(2pi*fc*t + amp_mod*sin(2pi*fm*t))   
    Ac, fc, amp_mod, fm must to be float
    '''

    def fm_modulation(self, amp_m, freq_m, freq_c, amp_c, waveform):
        t = self.time_samples
        self.modulation = self.oscillator(t, amp_m, freq_m, 0, 'sine')
        if waveform == 'sin':
            self.signal = amp_c * torch.sin(TWO_PI * freq_c * t + self.modulation)
        if waveform == 'square':
            self.signal = torch.sign(torch.sin(TWO_PI * freq_c * t + self.modulation) * amp_c) + 1.0
        if waveform == 'tri':
            y = (torch.sign(torch.sin(TWO_PI * freq_c * t + self.modulation) * amp_c))
            self.signal = pytorch_forecasting.utils.autocorrelation(y, dim=0)
            pass
        if waveform == 'saw':
            y = (torch.sign(torch.sin(TWO_PI * freq_c * t + self.modulation) * amp_c)) + 1.0
            y1 = pytorch_forecasting.utils.autocorrelation(y, dim=0) + 1.0
            y2 = torch.roll(y1, shifts=1, dims=0)
            self.signal = torch.where(y2 <= y1, y2, torch.zeros(1))
            self.signal = self.signal[self.signal != 0.0]
            self.signal = torch.cat((self.signal, self.signal))
            pass

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

    # '''
    # calc A envelope
    # A+D+S+R = self.sig_duration ()
    # Ys is sustain value, amp is the max point in A
    # '''
    #
    # def adsr_envelope_moshe(self, amp, A, D, S, Ys, R):
    #     time_sample = torch.linspace(0, self.sig_duration, steps=self.sample_rate, dtype=torch.float64)
    #     time_sample = torch.where(time_sample <= A, time_sample, 0.)
    #     A_env = time_sample * A / amp
    #
    #     '''calc D envleope'''
    #     time_sample = torch.linspace(0, self.sig_duration, steps=self.sample_rate, dtype=torch.float64)
    #     condition = torch.logical_and(A < time_sample, time_sample <= A + D)
    #     time_sample = torch.where(condition, time_sample, 0.)
    #     D_env = (A + D - time_sample) * (A - Ys) / D
    #
    #     '''calc S envelope'''
    #     time_sample = torch.linspace(0, self.sig_duration, steps=self.sample_rate, dtype=torch.float64)
    #     condition = torch.logical_and(A + D < time_sample, time_sample <= self.sig_duration - R)
    #     time_sample = torch.where(condition, time_sample, 0.)
    #     S_env = time_sample * Ys
    #
    #     '''calc R envelope'''
    #     time_sample = torch.linspace(0, self.sig_duration, steps=self.sample_rate, dtype=torch.float64)
    #     condition = torch.logical_and(self.sig_duration - R < time_sample , time_sample <= self.sig_duration)
    #     time_sample = torch.where(condition, 1 - time_sample, 0.)
    #     R_env = time_sample * Ys / (1 - R)
    #
    #     '''build envelope'''
    #     envelope = torch.cat([A_env, D_env, S_env, R_env], axis=0)
    #     plt.plot(envelope)
    #     plt.show()
    #     self.signal = self.signal * envelope

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
        if attack_t + decay_t + sustain_t + release_t != self.sig_duration:
            raise ValueError("Provided ADSR durations does not match signal duration")

        attack_num_samp = int(self.sample_rate * attack_t)
        decay_num_samp = int(self.sample_rate * decay_t)
        sustain_num_samp = int(self.sample_rate * sustain_t)
        release_num_samp = int(self.sample_rate * release_t)

        attack = torch.linspace(0, 1, attack_num_samp)
        decay = torch.linspace(1, sustain_level, decay_num_samp)
        sustain = torch.full((sustain_num_samp,), sustain_level)
        release = torch.linspace(sustain_level, 0, release_num_samp)

        envelope = torch.cat((attack, decay, sustain, release))
        envelope_len = envelope.shape[0]
        signal_len = self.time_samples.shape[0]
        if envelope_len != signal_len:
            padding = torch.zeros(signal_len - envelope_len)
            envelope = torch.cat((envelope, padding))

        self.signal = self.signal * envelope

        # plt.plot(envelope)
        # plt.plot(self.signal)
        # plt.show()

    def low_pass(self, cutoff_freq, q=0.707):
        self.signal = taF.lowpass_biquad(self.signal, self.sample_rate, cutoff_freq, q)

    def high_pass(self, cutoff_freq, q):
        self.signal = taF.highpass_biquad(self.signal, self.sample_rate, cutoff_freq, q)

    def band_pass(self, central_freq, q=0.707, const_skirt_gain=False):
        self.signal = \
            taF.bandpass_biquad(self.signal, self.sample_rate, central_freq, q, const_skirt_gain)

    # todo: Apply reverb, echo and filtering using th DDSP library
    def reverb(input_signal):
        # reverb_fx = ddsp.effects.Reverb()
        # input_signal = reverb_fx(input_signal)
        return input_signal

    @staticmethod
    def signal_values_sanity_check(amp, freq, waveform):
        """Check signal properties are reasonable."""
        if freq < 0 or freq > 22000:
            raise ValueError("Provided frequency is not in range [0, 22000]")
        if amp < 0 or amp > 1:
            raise ValueError("Provided amplitude is not in range [0, 1]")
        if not any(x == waveform for x in ['sine', 'square', 'triangle', 'sawtooth']):
            raise ValueError("Unknown waveform provided")


a = Signal()
a.oscillator(amp=1, freq=100, phase=0, waveform='sine')
a.adsr_envelope(0.241, 0.259, 0.25, 0.5, 0.25)
# b = Signal()
# a.am_modulation(amp_c=1, freq_c=4, amp_m=0.3, freq_m=0, final_max_amp=0.5, waveform='sine')
# b.am_modulation_by_input_signal(a.data, modulation_factor=1, amp_c=0.5, freq_c=40, waveform='triangle')
# plt.plot(a.data)
# plt.plot(b.data)
# torch.tensor(0)
# print(torch.sign(torch.tensor(0)))
# # b.fm_modulation_by_input_signal(a.data, 440, 1, 10, 'sawtooth')
# # plt.plot(b.data)
# plt.show()
#
# play_obj = sa.play_buffer(a.data.numpy(), 1, 4, a.sample_rate)
# play_obj.wait_done()
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

plt.show()
