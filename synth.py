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
# import torchaudio.functional
import matplotlib.pyplot as plt
import pytorch_forecasting


# data = torchaudio.functional.compute_kaldi_pitch('sine',sample_rate =2200,frame_length=5000)

class Signal():
    def __init__(self):
        self.freq_sample = 44100
        self.sig_duration = 1.0  # in seconds
        self.time_sample = torch.linspace(0, 1, steps=44100)
        self.pi = torch.acos(torch.zeros(1).float()) * 2.0
        self.modulation_time = torch.linspace(0, 1, steps=44100)
        self.modulation = 0
        self.data = 0
        self.shift = self.sig_duration / (self.freq_sample * 2)
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
        if freq < 0 or freq > 22000:
            raise ValueError("Provided frequency is not in range [0, 22000]")
        if amp < 0 or amp > 1:
            raise ValueError("Provided amplitude is not in range [0, 1]")
        if not any(x == waveform for x in ['sine', 'square', 'triangle', 'sawtooth']):
            raise ValueError("Unknown waveform provided")

        phase = phase % (2 * self.pi.item())
        oscillator = torch.zeros_like(self.time_sample)
        if waveform == 'sine':
            oscillator = amp * torch.sin(2 * self.pi * freq * self.time_sample + phase)
        elif waveform == 'square':
            oscillator = amp * torch.sign(torch.sin(2 * self.pi * freq * self.time_sample + phase))
        elif waveform == 'triangle':
            oscillator = (2 * amp / self.pi) * torch.arcsin(torch.sin((2 * self.pi * freq * self.time_sample)))
        elif waveform == 'sawtooth':
            oscillator = amp * 2 * (self.time_sample * freq - torch.floor(0.5 + self.time_sample * freq))
            oscillator = torch.roll(oscillator, int(self.freq_sample * phase / (2 * self.pi * freq)))

        return oscillator

    '''
    superposition of 2 signals. factor is a number in range [0,1]
    which balances the 2 signals in the mix. 
    with 0 value - the mixed signal will be made from signal1 only 
    with 1 value - the mixed signal will be made from signal2 only
    with 0.5 - the mixed signal composition is evenly balanced. 
    '''

    def mix_signal(self, signal1, signal2, factor):
        if len(signal1) == len(signal2):
            mixed_signal = factor * signal1 + (1 - factor) * signal2
        else:
            print("Exception in mix_signal - signals length are not equal")

    '''Ac*sin(2pi*fc*t + amp_mod*sin(2pi*fm*t))   
    Ac, fc, amp_mod, fm must to be float
    '''

    def fm_modulation(self, amp_mod, fm, fc, Ac, waveform):
        self.modulation = torch.sin(self.modulation_time * self.pi * 2.0 * fm)
        self.modulation = self.modulation * amp_mod
        if waveform == 'sin':
            self.data = Ac * torch.sin(self.time_sample * self.pi * 2.0 * fc + self.modulation)
        if waveform == 'square':
            self.data = torch.sign(torch.sin(self.time_sample * self.pi * 2.0 * fc + self.modulation) * Ac) + 1.0
        if waveform == 'tri':
            y = (torch.sign(torch.sin(self.time_sample * self.pi * 2.0 * fc + self.modulation) * Ac))
            self.data = pytorch_forecasting.utils.autocorrelation(y, dim=0)
            pass
        if waveform == 'saw':
            y = (torch.sign(torch.sin(self.time_sample * self.pi * 2.0 * fc + self.modulation) * Ac)) + 1.0
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

    def fm_modulation_for_input(self, input_signal, fc, Ac, amp_mod):
        if len(self.time_sample) == len(input_signal):
            self.data = modulated_signal = Ac * torch.sin(2 * self.pi * fc * self.time_sample + amp_mod * input_signal)
        else:
            print("Signal length inappropriate. Modulation can't be done")

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

        '''calc D envlope'''
        time_sample = torch.linspace(0, 1, steps=44000)
        time_sample = torch.where(time_sample > A and time_sample <= A + D, time_sample, 0)
        D_env = (A + D - time_sample) * (A - Ys) / D

        '''calc S envlope'''
        time_sample = torch.linspace(0, 1, steps=44000)
        time_sample = torch.where(time_sample > A + D and time_sample <= self.sig_duration - R, time_sample, 0)
        S_env = time_sample * Ys

        '''calc R envlope'''
        time_sample = torch.linspace(0, 1, steps=44000)
        time_sample = torch.where(time_sample > self.sig_duration - R and time_sample <= self.sig_duration,
                                  1 - time_sample, 0)
        R_env = time_sample * Ys / (1 - R)

        '''build envlope'''
        envelope = torch.cat([A_env, D_env, S_env, R_env], axis=0)
        self.data = self.data * envelope

    # todo: write functions behavior
    def low_pass(self, fc, pole):
        pass

    def band_pass(self, fc, pole):
        pass

    def high_pass(self, fc, pole):
        pass

    def all_pass(self, fc, pole):
        pass

    def Resonance_filter(self):
        pass

    # todo: Apply reverb, echo and filtering using th DDSP library
    def reverb(input_signal):
        # reverb_fx = ddsp.effects.Reverb()
        # input_signal = reverb_fx(input_signal)
        return input_signal


a = Signal()
b = a.oscillator(0.5, 5, 0, 'sawtooth')
plt.plot(b)

# def fm_modulation(self, amp_mod, fm, fc, Ac, waveform):
a.fm_modulation(0.0, 3.0, 5, 1, 'tri')
# print(a.data)
# plt.plot(a.data)

plt.show()
