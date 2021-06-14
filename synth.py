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
import torchaudio.functional
import matplotlib.pyplot as plt
import pytorch_forecasting
import simpleaudio as sa

PI = 3.141592653589793
TWO_PI = 2 * PI


# data = torchaudio.functional.compute_kaldi_pitch('sine',sample_rate =2200,frame_length=5000)

class Signal:
    def __init__(self):
        self.sample_rate = 44100
        self.sig_duration = 1.0  # in seconds
        self.time_sample = torch.linspace(0, 1, steps=44100)
        self.pi = torch.acos(torch.zeros(1).float()) * 2.0
        self.modulation_time = torch.linspace(0, 1, steps=44100)
        self.modulation = 0
        self.data = 0
        self.shift = self.sig_duration / (self.sample_rate * 2)
        self.conv = nn.Conv1d(1, 1, 1, 1)
        torch.wave = 0

    def oscillator(self, t, amp, freq, phase, waveform):
        """Creates a basic oscillator.

            Retrieves a waveform shape and attributes, and construct the respected signal

            Args:
                self: Self object
                t: time vector
                amp: Amplitude in range [0, 1]
                freq: Frequency in range [0, 22000]
                phase: Phase in range [0, 2pi]
                waveform: one of [sine, square, triangle, sawtooth]

            Returns:
                A torch with the constructed signal

            Raises:
                ValueError: Provided variables are out of range
            """
        if freq < 0 or freq > 22000:
            raise ValueError("Provided frequency is not in range [0, 22000]")
        if amp < 0 or amp > 1:
            raise ValueError("Provided amplitude is not in range [0, 1]")
        if not any(x == waveform for x in ['sine', 'square', 'triangle', 'sawtooth']):
            raise ValueError("Unknown waveform provided")

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

        self.data = oscillator

    def mix_signal(self, new_signal, factor):
        """Signal superposition. factor balances the mix
        0 - original signal only, 1 - new signal only, 0.5 evenly balanced. """
        self.data = factor * self.data + (1 - factor) * new_signal

    '''Ac*sin(2pi*fc*t + amp_mod*sin(2pi*fm*t))   
    Ac, fc, amp_mod, fm must to be float
    '''
    def fm_modulation(self, amp_m, freq_m, freq_c, amp_c, waveform):
        t = self.time_sample
        self.modulation = self.oscillator(t, amp_m, freq_m, 0, 'sine')
        if waveform == 'sin':
            self.data = amp_c * torch.sin(TWO_PI * freq_c * t + self.modulation)
        if waveform == 'square':
            self.data = torch.sign(torch.sin(TWO_PI * freq_c * t + self.modulation) * amp_c) + 1.0
        if waveform == 'tri':
            y = (torch.sign(torch.sin(TWO_PI * freq_c * t + self.modulation) * amp_c))
            self.data = pytorch_forecasting.utils.autocorrelation(y, dim=0)
            pass
        if waveform == 'saw':
            y = (torch.sign(torch.sin(TWO_PI * freq_c * t + self.modulation) * amp_c)) + 1.0
            y1 = pytorch_forecasting.utils.autocorrelation(y, dim=0) + 1.0
            y2 = torch.roll(y1, shifts=1, dims=0)
            self.data = torch.where(y2 <= y1, y2, torch.zeros(1))
            self.data = self.data[self.data != 0.0]
            self.data = torch.cat((self.data, self.data))
            pass

    '''
    Ac*sin(2pi*fc*t + amp_mod*signal)   
    Ac, fc, amp_mod, fm must to be float
    '''

    def fm_modulation_for_input(self, input_signal, freq_c, amp_c, mod_index, waveform):
        t = self.time_sample
        if waveform == 'sine':
            self.data = amp_c * torch.sin(TWO_PI * freq_c * t + mod_index * input_signal)
        if waveform == 'square':
            self.data = amp_c * torch.sign(torch.sin(TWO_PI * freq_c * t + mod_index * input_signal))
        if waveform == 'triangle':
            self.data = (2 * amp_c / PI) * torch.arcsin(torch.sin((TWO_PI * freq_c * t + mod_index * input_signal)))
        if waveform == 'sawtooth':
            oscillator = 2 * (t * freq_c - torch.floor(0.5 + t * freq_c))
            oscillator = (((oscillator + 1) / 2) + mod_index * input_signal / TWO_PI) % 1
            self.data = amp_c * (oscillator * 2 - 1)

    '''
    (Ac + Am*cos(2*pi*fm*t)) * cos(2*pi*fc*t)
    Ac, amp_mod, fm, fc, must to be float
    '''

    def am_modulation(self, fm, fc, Ac, Am):
        modulator = Am * torch.cos(2 * self.pi * fm * self.time_sample)
        carrier = torch.cos(2 * self.pi * fc * self.time_sample)
        modulated_amplitude = (Ac + modulator)
        self.data = modulated_amplitude * carrier

    '''
    (Ac + input_signal) * cos(2*pi*fc*t)
    Ac, amp_mod, fm, fc, must to be float
    '''

    def am_modulation_for_input(self, input_signal, fc, Ac, Am):
        modulator = Am * input_signal
        carrier = torch.cos(2 * self.pi * fc * self.time_sample)
        modulated_amplitude = (Ac + modulator)
        self.data = modulated_amplitude * carrier

    '''
    calc A envelope
    A+D+S+R = self.sig_duration ()
    Ys is sustain value, amp is the max point in A
    '''

    def adsr_envelope(self, amp, A, D, S, Ys, R):
        time_sample = torch.linspace(0, 1, steps=44000)
        time_sample = torch.where(time_sample <= A, time_sample, 0)
        A_env = time_sample * A / amp

        '''calc D envleope'''
        time_sample = torch.linspace(0, 1, steps=44000)
        time_sample = torch.where(time_sample > A and time_sample <= A + D, time_sample, 0)
        D_env = (A + D - time_sample) * (A - Ys) / D

        '''calc S envelope'''
        time_sample = torch.linspace(0, 1, steps=44000)
        time_sample = torch.where(time_sample > A + D and time_sample <= self.sig_duration - R, time_sample, 0)
        S_env = time_sample * Ys

        '''calc R envelope'''
        time_sample = torch.linspace(0, 1, steps=44000)
        time_sample = torch.where(time_sample > self.sig_duration - R and time_sample <= self.sig_duration,
                                  1 - time_sample, 0)
        R_env = time_sample * Ys / (1 - R)

        '''build envelope'''
        envelope = torch.cat([A_env, D_env, S_env, R_env], axis=0)
        self.data = self.data * envelope

    def low_pass(self, cutoff_freq, q=0.707):
        self.data = torchaudio.functional.lowpass_biquad(self.data, self.sample_rate, cutoff_freq, q)

    def high_pass(self, cutoff_freq, q):
        self.data = torchaudio.functional.highpass_biquad(self.data, self.sample_rate, cutoff_freq, q)

    def band_pass(self, central_freq, q=0.707, const_skirt_gain=False):
        self.data = \
            torchaudio.functional.bandpass_biquad(self.data, self.sample_rate, central_freq, q, const_skirt_gain)

    # todo: ask Moshe to remove this. It changes only phase of frequencies
    def all_pass(self, fc, pole):
        pass

    # todo: ask Moshe to remove this
    def resonance_filter(self):
        pass

    # todo: Apply reverb, echo and filtering using th DDSP library
    def reverb(input_signal):
        # reverb_fx = ddsp.effects.Reverb()
        # input_signal = reverb_fx(input_signal)
        return input_signal


a = Signal()
b = Signal()
a.oscillator(a.time_sample, amp=0.75, freq=3, phase=0, waveform='sine')
plt.plot(a.data)
torch.tensor(0)
print(torch.sign(torch.tensor(0)))
b.fm_modulation_for_input(a.data, 440, 1, 10, 'sawtooth')
plt.plot(b.data)
plt.show()

print(a.data.shape)
print(a.data.dtype)
play_obj = sa.play_buffer(b.data.numpy(), 1, 4, a.sample_rate)
play_obj.wait_done()
# plt.plot(a.data)
a.low_pass(1000)
# play_obj = sa.play_buffer(a.data.numpy(), 1, 4, a.sample_rate)
# play_obj.wait_done()
# plt.plot(a.data)
#
# def fm_modulation(self, amp_mod, fm, fc, Ac, waveform):
# a.fm_modulation(1, 3, 5, 1, 'tri')
# print(a.data)
# plt.plot(a.data)

plt.show()
