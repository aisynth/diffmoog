#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 31 15:41:38 2021

@author: Moshe Laufer, Noy Uzrad
"""
import numpy as np

import torch
import math

from torchaudio.functional.filtering import lowpass_biquad, highpass_biquad
from torchaudio.transforms import Spectrogram, GriffinLim
import matplotlib.pyplot as plt
from model import helper
import julius
from julius.lowpass import lowpass_filter_new
from config import SynthConfig

try:
    from functorch import vmap
    has_vmap = True
except ModuleNotFoundError:
    has_vmap = False


PI = math.pi
TWO_PI = 2 * PI


class SynthModules:
    def __init__(self, num_sounds=1, sample_rate=44100, signal_duration_sec=1.0, device='cuda:0'):
        self.sample_rate = sample_rate
        self.sig_duration = signal_duration_sec
        self.time_samples = torch.linspace(0, self.sig_duration, steps=int(self.sample_rate * self.sig_duration),
                                           requires_grad=True)
        self.modulation_time = torch.linspace(0, self.sig_duration, steps=self.sample_rate)
        self.modulation = 0
        self.signal = torch.zeros(size=(num_sounds, self.time_samples.shape[0]), dtype=torch.float32,
                                  requires_grad=True)

        self.device = device

        self.time_samples = helper.move_to(self.time_samples, self.device)
        self.modulation_time = helper.move_to(self.modulation_time, self.device)
        self.signal = helper.move_to(self.signal, self.device)

        self.wave_type_indices = {k: torch.tensor(v, dtype=torch.long, device=self.device).squeeze()
                                  for k, v in SynthConfig.wave_type_dict.items()}
        self.filter_type_indices = {k: torch.tensor(v, dtype=torch.long, device=self.device).squeeze()
                                    for k, v in SynthConfig.filter_type_dict.items()}
        # self.room_impulse_responses = torch.load('rir_for_reverb_no_amp')

    def oscillator(self, amp, freq, waveform, phase=0, num_sounds=1):
        """Creates a basic oscillator.

            Retrieves a waveform shape and attributes, and construct the respected signal

            Args:
                self: Self object
                amp: Amplitude in range [0, 1]
                freq: Frequency in range [0, 22000]
                phase: Phase in range [0, 2pi], default is 0
                waveform: a string, one of ['sine', 'square', 'triangle', 'sawtooth'] or a probability vector
                num_sounds: number of sounds to process

            Returns:
                A torch with the constructed signal

            Raises:
                ValueError: Provided variables are out of range
                :rtype: object
            """

        self.signal_values_sanity_check(amp, freq, waveform)
        t = self.time_samples
        # oscillator = torch.zeros_like(t, requires_grad=True)

        oscillator_tensor = torch.tensor((), requires_grad=True).to(self.device)
        first_time = True
        for i in range(num_sounds):

            if num_sounds == 1 \
                    and not isinstance(amp, list)\
                    and not torch.is_tensor(amp):
                freq_float = freq
                amp_float = amp
                waveform_current = waveform
            else:
                if isinstance(amp, int):
                    amp_float = amp
                else:
                    amp_float = amp[i]
                freq_float = freq[i]
                waveform_current = waveform[i]
            sine_wave = amp_float * torch.sin(TWO_PI * freq_float * t + phase)
            square_wave = amp_float * torch.sign(torch.sin(TWO_PI * freq_float * t + phase))
            triangle_wave = (2 * amp_float / PI) * torch.arcsin(torch.sin((TWO_PI * freq_float * t + phase)))
            # triangle_wave = amp * torch.sin(TWO_PI * freq_float * t + phase)
            sawtooth_wave = 2 * (t * freq_float - torch.floor(0.5 + t * freq_float))  # Sawtooth closed form
            # Phase shift (by normalization to range [0,1] and modulo operation)
            sawtooth_wave = (((sawtooth_wave + 1) / 2) + phase / TWO_PI) % 1
            sawtooth_wave = amp_float * (sawtooth_wave * 2 - 1)  # re-normalization to range [-amp, amp]

            if isinstance(waveform_current, str):
                if waveform_current == 'sine':
                    oscillator = sine_wave
                elif waveform_current == 'square':
                    oscillator = square_wave
                elif waveform_current == 'triangle':
                    oscillator = triangle_wave
                elif waveform_current == 'sawtooth':
                    oscillator = sawtooth_wave
                else:
                    AssertionError("Unknown waveform")

            else:
                waveform_probabilities = waveform[i]
                oscillator = waveform_probabilities[0] * sine_wave \
                             + waveform_probabilities[1] * square_wave \
                             + waveform_probabilities[2] * sawtooth_wave
                # + waveform_probabilities[2] * triangle_wave \

            if first_time:
                oscillator_tensor = torch.cat((oscillator_tensor, oscillator), dim=0).unsqueeze(dim=0)
                first_time = False
            else:
                oscillator = oscillator.unsqueeze(dim=0)
                oscillator_tensor = torch.cat((oscillator_tensor, oscillator), dim=0)

        return oscillator_tensor

    def batch_oscillator(self, amp, freq, waveform, phase=0):
        """Creates a basic oscillator.

            Retrieves a waveform shape and attributes, and construct the respected signal

            Args:
                self: Self object
                amp: batch of n amplitudes in range [0, 1]
                freq: batch of n frequencies in range [0, 22000]
                phase: batch of n phases in range [0, 2pi], default is 0
                waveform: batch of n strings, one of ['sine', 'square', 'triangle', 'sawtooth'] or a probability vector

            Returns:
                A torch tensor with the constructed signal

            Raises:
                ValueError: Provided variables are out of range
                :rtype: object
            """

        self.signal_values_sanity_check(amp, freq, waveform)

        amp = self._standardize_batch_input(amp, requested_dtype=torch.float32, requested_dims=2)
        freq = self._standardize_batch_input(freq, requested_dtype=torch.float32, requested_dims=2)

        sine_wave, square_wave, sawtooth_wave = self._generate_wave_tensors(amp, freq, phase)

        waves_tensor = torch.stack([sine_wave, square_wave, sawtooth_wave])
        oscillator_tensor = self._mix_waveforms(waves_tensor, waveform, self.wave_type_indices)

        return oscillator_tensor

    def _generate_wave_tensors(self, amp, freq, phase_mod):

        t = self.time_samples

        sine_wave = amp * torch.sin(TWO_PI * freq * t + phase_mod)

        square_wave = amp * torch.sign(torch.sin(TWO_PI * freq * t + phase_mod))

        sawtooth_wave = 2 * (t * freq - torch.floor(0.5 + t * freq))  # Sawtooth closed form

        # Phase shift (by normalization to range [0,1] and modulo operation)
        sawtooth_wave = (((sawtooth_wave + 1) / 2) + phase_mod / TWO_PI) % 1
        sawtooth_wave = amp * (sawtooth_wave * 2 - 1)  # re-normalization to range [-amp, amp]

        return sine_wave, square_wave, sawtooth_wave

    def mix_signal(self, new_signal, factor):
        """Signal superposition. factor balances the mix
        1 - original signal only, 0 - new signal only, 0.5 evenly balanced. """
        if factor < 0 or factor > 1:
            raise ValueError("Provided factor value is out of range [0, 1]")
        self.signal = factor * self.signal + (1 - factor) * new_signal

    def batch_specific_waveform_oscillator_fm(self, amp_c, freq_c, waveform, mod_index, modulator):
        """Sine/Square/Saw oscillator with FM modulation

            Creates an oscillator and modulates its frequency by a given modulator

            Args:
                self: Self object
                amp_c: Vector of amplitude in range [0, 1]
                freq_c: Vector of Frequencies in range [0, 22000]
                waveform: type from ['sine', 'square', 'triangle', 'sawtooth']
                mod_index: Vector of modulation indexes, which affects the amount of modulation
                modulator: Vector of modulator signals, to affect carrier frequency

            Returns:
                A torch with the constructed FM signal

            Raises:
                ValueError: Provided variables are out of range
            """

        self.signal_values_sanity_check(amp_c, freq_c, waveform)

        amp_c = self._standardize_batch_input(amp_c, requested_dtype=torch.float32, requested_dims=2)
        freq_c = self._standardize_batch_input(freq_c, requested_dtype=torch.float32, requested_dims=2)
        mod_index = self._standardize_batch_input(mod_index, requested_dtype=torch.float32, requested_dims=2)
        modulator = self._standardize_batch_input(modulator, requested_dtype=torch.float32, requested_dims=2)

        t = self.time_samples

        if waveform == 'sine':
            oscillator_tensor = amp_c * torch.sin(TWO_PI * freq_c * t + mod_index * torch.cumsum(modulator, dim=1))

        elif waveform == 'square':
            oscillator_tensor = amp_c * torch.sign(
                torch.sin(TWO_PI * freq_c * t + mod_index * torch.cumsum(modulator, dim=1)))

        # fm_triangle_wave = (2 * amp_c / PI) * torch.arcsin(torch.sin((TWO_PI * freq_c * t + mod_index * torch.cumsum(modulator, dim=1))))

        elif waveform == 'sawtooth':
            fm_sawtooth_wave = 2 * (t * freq_c - torch.floor(0.5 + t * freq_c))
            fm_sawtooth_wave = (((fm_sawtooth_wave + 1) / 2) + mod_index * torch.cumsum(modulator, dim=1) / TWO_PI) % 1
            oscillator_tensor = amp_c * (fm_sawtooth_wave * 2 - 1)
        else:
            oscillator_tensor = -1
            ValueError("Provided waveform is not supported")

        return oscillator_tensor

    def batch_oscillator_fm(self, amp_c, freq_c, waveform, mod_index, modulator):
        """Batch oscillator with FM modulation

            Creates an oscillator and modulates its frequency by a given modulator
            Args come as vector of values, with length of the numer of sounds to create
            Args:
                self: Self object
                amp_c: Vector of amplitude in range [0, 1]
                freq_c: Vector of Frequencies in range [0, 22000]
                waveform: Vector of waveform with type from [sine, square, triangle, sawtooth]
                mod_index: Vector of modulation indexes, which affects the amount of modulation
                modulator: Vector of modulator signals, to affect carrier frequency

            Returns:
                A torch with the constructed FM signal

            Raises:
                ValueError: Provided variables are out of range
            """

        self.signal_values_sanity_check(amp_c, freq_c, waveform)

        amp_c = self._standardize_batch_input(amp_c, requested_dtype=torch.float32, requested_dims=2)
        freq_c = self._standardize_batch_input(freq_c, requested_dtype=torch.float32, requested_dims=2)
        mod_index = self._standardize_batch_input(mod_index, requested_dtype=torch.float32, requested_dims=2)
        modulator = self._standardize_batch_input(modulator, requested_dtype=torch.float32, requested_dims=2)

        t = self.time_samples

        fm_sine_wave = amp_c * torch.sin(TWO_PI * freq_c * t + mod_index * torch.cumsum(modulator, dim=1))

        fm_square_wave = amp_c * torch.sign(
            torch.sin(TWO_PI * freq_c * t + mod_index * torch.cumsum(modulator, dim=1)))

        # fm_triangle_wave = (2 * amp_c / PI) * torch.arcsin(torch.sin((TWO_PI * freq_c * t + mod_index * torch.cumsum(modulator, dim=1))))

        fm_sawtooth_wave = 2 * (t * freq_c - torch.floor(0.5 + t * freq_c))
        fm_sawtooth_wave = (((fm_sawtooth_wave + 1) / 2) + mod_index * torch.cumsum(modulator, dim=1) / TWO_PI) % 1
        fm_sawtooth_wave = amp_c * (fm_sawtooth_wave * 2 - 1)

        waves_tensor = torch.stack([fm_sine_wave, fm_square_wave, fm_sawtooth_wave])
        oscillator_tensor = self._mix_waveforms(waves_tensor, waveform, self.wave_type_indices)

        return oscillator_tensor

    @staticmethod
    def _mix_waveforms(waves_tensor, raw_waveform_selector, type_indices):

        if isinstance(raw_waveform_selector, str):
            idx = type_indices[raw_waveform_selector]
            return waves_tensor[idx]

        if isinstance(raw_waveform_selector[0], str):
            oscillator_tensor = torch.stack([waves_tensor[type_indices[wt]][i] for i, wt in
                                             enumerate(raw_waveform_selector)])
            return oscillator_tensor

        oscillator_tensor = 0
        for i in range(len(waves_tensor)):
            oscillator_tensor += raw_waveform_selector.t()[i].unsqueeze(1) * waves_tensor[i]

        return oscillator_tensor

    def oscillator_fm(self, amp_c, freq_c, waveform, mod_index, modulator, num_sounds=1):
        """Basic oscillator with FM modulation

            Creates an oscillator and modulates its frequency by a given modulator

            Args:
                self: Self object
                amp_c: Amplitude in range [0, 1]
                freq_c: Frequency in range [0, 22000]
                waveform: One of [sine, square, triangle, sawtooth]
                mod_index: Modulation index, which affects the amount of modulation
                modulator: Modulator signal, to affect carrier frequency
                num_sounds: number of sounds to process

            Returns:
                A torch with the constructed FM signal

            Raises:
                ValueError: Provided variables are out of range
            """

        self.signal_values_sanity_check(amp_c, freq_c, waveform)
        t = self.time_samples
        oscillator_tensor = torch.tensor((), requires_grad=True).to(self.device)
        first_time = True
        for i in range(num_sounds):
            if num_sounds == 1:
                amp_float = amp_c
                mod_index_float = mod_index
                freq_float = freq_c
                input_signal_cur = modulator
            else:
                amp_float = amp_c[i]
                mod_index_float = mod_index[i]
                freq_float = freq_c[i]
                input_signal_cur = modulator[i]

            fm_sine_wave = amp_float * torch.sin(TWO_PI * freq_float * t + mod_index_float * input_signal_cur)
            fm_square_wave = amp_float * torch.sign(
                torch.sin(TWO_PI * freq_float * t + mod_index_float * input_signal_cur))
            # fm_triangle_wave = (2 * amp_float / PI) * torch.arcsin(torch.sin((TWO_PI * freq_float * t + mod_index_float * input_signal_cur)))
            fm_triangle_wave = amp_float * torch.sin(TWO_PI * freq_float * t + mod_index_float * input_signal_cur)

            fm_sawtooth_wave = 2 * (t * freq_float - torch.floor(0.5 + t * freq_float))
            fm_sawtooth_wave = (((fm_sawtooth_wave + 1) / 2) + mod_index_float * input_signal_cur / TWO_PI) % 1
            fm_sawtooth_wave = amp_float * (fm_sawtooth_wave * 2 - 1)

            if isinstance(waveform, list):
                waveform = waveform[i]

            if isinstance(waveform, str):
                if waveform == 'sine':
                    oscillator = fm_sine_wave
                elif waveform == 'square':
                    oscillator = fm_square_wave
                elif waveform == 'triangle':
                    oscillator = fm_triangle_wave
                elif waveform == 'sawtooth':
                    oscillator = fm_sawtooth_wave

            else:
                if num_sounds == 1:
                    waveform_probabilities = waveform

                elif torch.is_tensor(waveform):
                    waveform_probabilities = waveform[i]

                oscillator = waveform_probabilities[0] * fm_sine_wave \
                             + waveform_probabilities[1] * fm_square_wave \
                             + waveform_probabilities[2] * fm_sawtooth_wave

            if first_time:
                if num_sounds == 1:
                    oscillator_tensor = oscillator
                else:
                    oscillator_tensor = torch.cat((oscillator_tensor, oscillator), dim=0).unsqueeze(dim=0)
                    first_time = False
            else:
                oscillator = oscillator.unsqueeze(dim=0)
                oscillator_tensor = torch.cat((oscillator_tensor, oscillator), dim=0)

        return oscillator_tensor

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
        carrier = SynthModules(device=self.device)
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
        carrier = SynthModules()
        carrier.oscillator(amp=1, freq=freq_c, phase=0, waveform=waveform)
        modulated_amplitude = (amp_c + modulation_factor * input_signal)
        if torch.max(modulated_amplitude).item() > 1 or torch.min(modulated_amplitude).item() < -1:
            raise ValueError("AM modulation resulted amplitude out of range [-1, 1].")
        self.signal = modulated_amplitude * carrier.signal

    def non_diff_adsr_envelope(self, input_signal, attack_t, decay_t, sustain_t, sustain_level, release_t, num_sounds=1):
        """Apply an ADSR envelope to the signal. Non-differential operation

            builds the ADSR shape and multiply by the signal

            Args:
                self: Self object
                input_signal: target signal to apply adsr
                attack_t: Length of attack in seconds. Time to go from 0 to 1 amplitude.
                decay_t: Length of decay in seconds. Time to go from 1 amplitude to sustain level.
                sustain_t: Length of sustain in seconds, with sustain level amplitude
                sustain_level: Sustain volume level
                release_t: Length of release in seconds. Time to go ftom sustain level to 0 amplitude
                num_sounds: number of sounds to process

            Raises:
                ValueError: Provided ADSR timings are not the same as the signal length
            """
        check_adsr_timings(attack_t, decay_t, sustain_t, sustain_level, release_t, self.sig_duration, num_sounds)

        if num_sounds == 1:
            attack_num_samples = int(self.sig_duration * self.sample_rate * attack_t)
            decay_num_samples = int(self.sig_duration * self.sample_rate * decay_t)
            sustain_num_samples = int(self.sig_duration * self.sample_rate * sustain_t)
            release_num_samples = int(self.sig_duration * self.sample_rate * release_t)
        else:
            attack_num_samples = [torch.floor(self.sig_duration * self.sample_rate * attack_t[k]) for k in range(num_sounds)]
            decay_num_samples = [torch.floor(self.sig_duration * self.sample_rate * decay_t[k]) for k in range(num_sounds)]
            sustain_num_samples = [torch.floor(self.sig_duration * self.sample_rate * sustain_t[k]) for k in range(num_sounds)]
            release_num_samples = [torch.floor(self.sig_duration * self.sample_rate * release_t[k]) for k in range(num_sounds)]
            attack_num_samples = torch.stack(attack_num_samples)
            decay_num_samples = torch.stack(decay_num_samples)
            sustain_num_samples = torch.stack(sustain_num_samples)
            release_num_samples = torch.stack(release_num_samples)

        if num_sounds > 1:
            # todo: change the check with sustain_level[0]
            if torch.is_tensor(sustain_level[0]):
                sustain_level = [sustain_level[i] for i in range(num_sounds)]
                sustain_level = torch.stack(sustain_level)
            else:
                sustain_level = [sustain_level[i] for i in range(num_sounds)]

        enveloped_signal_tensor = torch.tensor((), requires_grad=True).to(helper.get_device())
        first_time = True
        for i in range(num_sounds):

            if num_sounds == 1:
                attack = torch.linspace(0, 1, attack_num_samples)
                decay = torch.linspace(1, sustain_level, decay_num_samples)
                sustain = torch.full((sustain_num_samples,), sustain_level)
                release = torch.linspace(sustain_level, 0, release_num_samples)
            else:
                # convert 1d vector to scalar tensor to be used in linspace
                # todo: lost gradients here! Using int() loses gradients since only float tensors can hold gradients
                attack_num_samples_sc = attack_num_samples[i].int().squeeze()
                decay_num_samples_sc = decay_num_samples[i].int().squeeze()
                sustain_num_samples_sc = sustain_num_samples[i].int().squeeze()
                sustain_level_sc = sustain_level[i].squeeze()
                release_num_samples_sc = release_num_samples[i].int().squeeze()

                attack = torch.linspace(0, 1, attack_num_samples_sc)
                decay = torch.linspace(1, sustain_level_sc, decay_num_samples_sc)
                sustain = torch.full((sustain_num_samples_sc,), sustain_level_sc)
                release = torch.linspace(sustain_level_sc, 0, release_num_samples_sc)

                # todo: make sure ADSR behavior is differentiable. linspace has to know to get tensors
                # attack_mod = helper.linspace(torch.tensor(0), torch.tensor(1), attack_num_samples[i])
                # decay_mod = helper.linspace(torch.tensor(1), sustain_level[i], decay_num_samples[i])
                # sustain_mod = torch.full((sustain_num_samples[i],), sustain_level[i])
                # release_mod = helper.linspace(sustain_num_samples[i], torch.tensor(0), release_num_samples[i])

            envelope = torch.cat((attack, decay, sustain, release))
            envelope = helper.move_to(envelope, self.device)

            # envelope = torch.cat((attack_mod, decay_mod, sustain_mod, release_mod))

            envelope_len = envelope.shape[0]
            signal_len = self.time_samples.shape[0]
            if envelope_len <= signal_len:
                padding = torch.zeros((signal_len - envelope_len), device=helper.get_device())
                envelope = torch.cat((envelope, padding))
            else:
                raise ValueError("Envelope length exceeds signal duration")

            if torch.is_tensor(input_signal) and num_sounds > 1:
                signal_to_shape = input_signal[i]
            else:
                signal_to_shape = input_signal

            enveloped_signal = signal_to_shape * envelope

            if first_time:
                if num_sounds == 1:
                    enveloped_signal_tensor = enveloped_signal
                else:
                    enveloped_signal_tensor = torch.cat((enveloped_signal_tensor, enveloped_signal), dim=0).unsqueeze(
                        dim=0)
                    first_time = False
            else:
                enveloped = enveloped_signal.unsqueeze(dim=0)
                enveloped_signal_tensor = torch.cat((enveloped_signal_tensor, enveloped), dim=0)

        return enveloped_signal_tensor

    def adsr_envelope(self, input_signal, attack_t, decay_t, sustain_t, sustain_level, release_t, num_sounds=1):
        """Apply an ADSR envelope to the signal. Differential operation

            builds the ADSR shape and multiply by the signal

            Args:
                self: Self object
                input_signal: target signal to apply adsr
                attack_t: Length of attack in seconds. Time to go from 0 to 1 amplitude.
                decay_t: Length of decay in seconds. Time to go from 1 amplitude to sustain level.
                sustain_t: Length of sustain in seconds, with sustain level amplitude
                sustain_level: Sustain volume level
                release_t: Length of release in seconds. Time to go ftom sustain level to 0 amplitude
                num_sounds: number of sounds to process

            Raises:
                ValueError: Provided ADSR timings are not the same as the signal length
            """
        check_adsr_timings(attack_t, decay_t, sustain_t, sustain_level, release_t, self.sig_duration, num_sounds)

        if num_sounds > 1:
            if torch.is_tensor(sustain_level[0]):
                sustain_level = [sustain_level[i] for i in range(num_sounds)]
                sustain_level = torch.stack(sustain_level)
            else:
                sustain_level = [sustain_level[i] for i in range(num_sounds)]

        enveloped_signal_tensor = torch.tensor((), requires_grad=True).to(self.device)
        first_time = True
        x = torch.linspace(0, 1.0, int(self.sample_rate * self.sig_duration), device=self.device)
        for i in range(num_sounds):
            if num_sounds == 1:
                #todo fix according to note off time
                note_off = attack_t + decay_t + sustain_t
                attack = x / attack_t
                attack = torch.clamp(attack, max=1.0)
                decay = (x - attack_t) * (sustain_level - 1) / (decay_t + 1e-5)
                decay = torch.clamp(decay, max=0.0)
                sustain = (x - note_off) * (-sustain_level / (release_t + 1e-5))

                envelope = (attack + decay + sustain)
                envelope = torch.clamp(envelope, min=0.0, max=1.0)
            else:
                note_off = attack_t[i] + decay_t[i] + sustain_t[i]
                attack = x / (attack_t[i] / note_off)
                attack = torch.clamp(attack, max=1.0)
                decay = (x - (attack_t[i] / note_off)) * (sustain_level[i] - 1) / ((decay_t[i] / note_off) + 1e-5)
                decay = torch.clamp(decay, max=0.0, min=sustain_level[i].item() - 1)
                sustain = (x - (note_off / self.sig_duration)) * (-sustain_level[i] / ((release_t[i] / note_off) + 1e-5))
                sustain = torch.clamp(sustain, max=0.0)

                envelope = (attack + decay + sustain)
                envelope = torch.clamp(envelope, min=0.0, max=1.0)

            envelope = helper.move_to(envelope, self.device)

            envelope_len = envelope.shape[0]
            signal_len = self.time_samples.shape[0]
            if envelope_len <= signal_len:
                padding = torch.zeros((signal_len - envelope_len), device=self.device)
                envelope = torch.cat((envelope, padding))
            else:
                raise ValueError("Envelope length exceeds signal duration")

            if torch.is_tensor(input_signal) and num_sounds > 1:
                signal_to_shape = input_signal[i]
            else:
                signal_to_shape = input_signal

            enveloped_signal = signal_to_shape * envelope

            if first_time:
                if num_sounds == 1:
                    enveloped_signal_tensor = enveloped_signal
                else:
                    enveloped_signal_tensor = torch.cat((enveloped_signal_tensor, enveloped_signal), dim=0).unsqueeze(
                        dim=0)
                    first_time = False
            else:
                enveloped = enveloped_signal.unsqueeze(dim=0)
                enveloped_signal_tensor = torch.cat((enveloped_signal_tensor, enveloped), dim=0)

        return enveloped_signal_tensor

    def batch_adsr_envelope(self, input_signal, attack_t, decay_t, sustain_t, sustain_level, release_t):
        """Apply an ADSR envelope to the signal

            builds the ADSR shape and multiply by the signal

            Args:
                self: Self object
                input_signal: target signal to apply adsr
                attack_t: Length of attack in seconds. Time to go from 0 to 1 amplitude.
                decay_t: Length of decay in seconds. Time to go from 1 amplitude to sustain level.
                sustain_t: Length of sustain in seconds, with sustain level amplitude
                sustain_level: Sustain volume level
                release_t: Length of release in seconds. Time to go ftom sustain level to 0 amplitude
                num_sounds: number of sounds to process

            Raises:
                ValueError: Provided ADSR timings are not the same as the signal length
            """
        attack_t = self._standardize_batch_input(attack_t, requested_dtype=torch.float64, requested_dims=2)
        decay_t = self._standardize_batch_input(decay_t, requested_dtype=torch.float64, requested_dims=2)
        sustain_t = self._standardize_batch_input(sustain_t, requested_dtype=torch.float64, requested_dims=2)
        release_t = self._standardize_batch_input(release_t, requested_dtype=torch.float64, requested_dims=2)
        sustain_level = self._standardize_batch_input(sustain_level, requested_dtype=torch.float64, requested_dims=2)

        if torch.any(attack_t + decay_t + sustain_t + release_t > self.sig_duration):
            raise ValueError("Provided ADSR durations exceeds signal duration")

        attack_num_samples = torch.floor(self.sig_duration * self.sample_rate * attack_t)
        decay_num_samples = torch.floor(self.sig_duration * self.sample_rate * decay_t)
        sustain_num_samples = torch.floor(self.sig_duration * self.sample_rate * sustain_t)
        release_num_samples = torch.floor(self.sig_duration * self.sample_rate * release_t)

        attack = torch.cat([torch.linspace(0, 1, int(attack_steps.item()), device=helper.get_device()) for attack_steps
                            in attack_num_samples])
        decay = torch.cat([torch.linspace(1, sustain_val.int(), int(decay_steps), device=helper.get_device())
                           for sustain_val, decay_steps in zip(sustain_level, decay_num_samples)])
        sustain = torch.cat([torch.full(sustain_steps, sustain_val, device=helper.get_device())
                             for sustain_steps, sustain_val in zip(sustain_num_samples, sustain_level)])
        release = torch.cat([torch.linspace(sustain_val, 0, int(release_steps), device=helper.get_device())
                             for sustain_val, release_steps in zip(sustain_level, release_num_samples)])

        envelope = torch.cat((attack, decay, sustain, release))

        envelope_len = envelope.shape[0]
        signal_len = self.time_samples.shape[0]
        if envelope_len <= signal_len:
            padding = torch.zeros((signal_len - envelope_len), device=helper.get_device())
            envelope = torch.cat((envelope, padding))
        else:
            raise ValueError("Envelope length exceeds signal duration")

        enveloped_signal = input_signal * envelope

        return enveloped_signal

    def amplitude_envelope(self, input_signal, envelope_shape):
        enveloped_signal = input_signal * envelope_shape
        return enveloped_signal

    def filter_envelope(self, input_signal, envelope_shape):
        spectrogram_transform = Spectrogram(n_fft=1024)
        inverse_spectrogram_transform = GriffinLim(n_fft=1024, length=16000)
        input_signal = input_signal.to('cpu')
        input_signal_spec = spectrogram_transform(input_signal)

        #todo: this is just an example linear time changing filter:
        envelope_shape = torch.ones_like(input_signal_spec)
        linspace = torch.floor(torch.linspace(0, envelope_shape.shape[1], steps=envelope_shape.shape[2]))
        for j in range(envelope_shape.shape[0]):
            k = 0
            for i in linspace.tolist():
                a = int(i)
                print(f"{j, a, k}")
                envelope_shape[j][0:a, k] = 0
                k += 1
        plt.imshow(envelope_shape[1])
        plt.show()
        plt.imshow(input_signal_spec[1].detach().cpu().numpy())
        plt.show()
        filtered_signal_spec = input_signal_spec * envelope_shape
        plt.imshow(filtered_signal_spec[1].detach().cpu().numpy())
        plt.show()

        filtered_signal = inverse_spectrogram_transform(filtered_signal_spec)

        return filtered_signal.to(self.device)

        # enveloped_signal = input_signal * envelope_shape
        # return enveloped_signal

    def filter(self, input_signal, filter_freq, filter_type, num_sounds=1):
        """Apply an ADSR envelope to the signal

            builds the ADSR shape and multiply by the signal

            Args:
                self: Self object
                :param input_signal: 1D or 2D array or tensor to apply filter along rows
                :param filter_type: one of ['low_pass', 'high_pass', 'band_pass']
                :param filter_freq: corner or central frequency
                :param num_sounds: number of sounds in the input

            Raises:
                none

            """
        filtered_signal_tensor = torch.tensor((), requires_grad=True).to(helper.get_device())
        first_time = True
        for i in range(num_sounds):
            if num_sounds == 1:
                filter_frequency = filter_freq
            elif num_sounds > 1:
                filter_frequency = filter_freq[i]

            if torch.is_tensor(filter_frequency):
                filter_frequency = helper.move_to(filter_frequency, "cpu")
            high_pass_signal = self.high_pass(input_signal[i], cutoff_freq=filter_frequency, index=i)
            low_pass_signal = self.low_pass(input_signal[i], cutoff_freq=filter_frequency, index=i)

            if isinstance(filter_type, list):
                filter_type = filter_type[i]

            if isinstance(filter_type, str):
                if filter_type == 'high_pass':
                    filtered_signal = high_pass_signal
                elif filter_type == 'low_pass':
                    filtered_signal = low_pass_signal

            else:
                if num_sounds == 1:
                    filter_type_probabilities = filter_type
                else:
                    filter_type_probabilities = filter_type[i]

                filter_type_probabilities = filter_type_probabilities.cpu()
                filtered_signal = filter_type_probabilities[0] * high_pass_signal \
                                  + filter_type_probabilities[1] * low_pass_signal
                filtered_signal = filtered_signal.to(helper.get_device())

            if first_time:
                if num_sounds == 1:
                    filtered_signal_tensor = filtered_signal
                else:
                    filtered_signal_tensor = torch.cat((filtered_signal_tensor, filtered_signal), dim=0).unsqueeze(
                        dim=0)
                    first_time = False
            else:
                filtered_signal = filtered_signal.unsqueeze(dim=0)
                filtered_signal_tensor = torch.cat((filtered_signal_tensor, filtered_signal), dim=0)

        return filtered_signal_tensor

    def batch_filter(self, input_signal, filter_freq, filter_type):
        """Apply an ADSR envelope to the signal

            builds the ADSR shape and multiply by the signal

            Args:
                self: Self object
                :param input_signal: 1D or 2D array or tensor to apply filter along rows
                :param filter_type: one of ['low_pass', 'high_pass', 'band_pass']
                :param filter_freq: corner or central frequency
                :param num_sounds: number of sounds in the input

            Raises:
                none

            """

        filter_freq = self._standardize_batch_input(filter_freq, requested_dtype=torch.float64, requested_dims=2)
        num_sounds = len(filter_freq)
        assert torch.all(filter_freq <= (self.sample_rate / 2)), "Filter cutoff frequency higher then Nyquist." \
                                                                 " Please check config"

        if has_vmap:
            high_pass_signal = vmap(highpass_biquad)(input_signal.double(), cutoff_freq=filter_freq,
                                                     sample_rate=self.sample_rate)
            low_pass_signal = vmap(lowpass_biquad)(input_signal.double(), cutoff_freq=filter_freq,
                                                   sample_rate=self.sample_rate)
        else:
            low_pass_signal = [self.low_pass(input_signal[i], filter_freq[i].cpu()) for i in range(num_sounds)]
            high_pass_signal = [self.high_pass(input_signal[i], filter_freq[i].cpu()) for i in range(num_sounds)]

            low_pass_signal = torch.stack(low_pass_signal)
            high_pass_signal = torch.stack(high_pass_signal)

        if torch.any(torch.isnan(high_pass_signal)) or torch.any(torch.isnan(low_pass_signal)):
            raise RuntimeError("Synth filter module: Signal has NaN. Exiting...")

        waves_tensor = torch.stack([low_pass_signal, high_pass_signal])

        filtered_signal_tensor = self._mix_waveforms(waves_tensor, filter_type, self.filter_type_indices)

        return filtered_signal_tensor

    def lowpass_batch_filter(self, input_signal, filter_freq, resonance):
        """Apply an ADSR envelope to the signal

            builds the ADSR shape and multiply by the signal

            Args:
                self: Self object
                :param input_signal: 1D or 2D array or tensor to apply filter along rows
                :param filter_freq: corner or central frequency
                :param num_sounds: number of sounds in the input

            Raises:
                none

            """

        filter_freq = self._standardize_batch_input(filter_freq, requested_dtype=torch.float64, requested_dims=2)
        resonance = self._standardize_batch_input(resonance, requested_dtype=torch.float64, requested_dims=2)

        num_sounds = len(filter_freq)
        assert torch.all(filter_freq <= (self.sample_rate / 2)), "Filter cutoff frequency higher then Nyquist." \
                                                                 " Please check config"

        if has_vmap:
            low_pass_signal = vmap(lowpass_biquad)(input_signal.double(), cutoff_freq=filter_freq,
                                                   sample_rate=self.sample_rate,
                                                   Q=resonance)
        else:
            low_pass_signal = [self.low_pass(input_signal[i], filter_freq[i].cpu()) for i in range(num_sounds)]

            low_pass_signal = torch.stack(low_pass_signal)

        if torch.any(torch.isnan(low_pass_signal)):
            raise RuntimeError("Synth filter module: Signal has NaN. Exiting...")

        filtered_signal_tensor = low_pass_signal

        return filtered_signal_tensor


    def low_pass(self, input_signal, cutoff_freq, q=0.707, index=0):
        # filtered_waveform = taF.lowpass_biquad(input_signal, self.sample_rate, cutoff_freq, q)
        # filtered_waveform = julius.lowpass_filter(input_signal, cutoff_freq.item()/44100)
        if cutoff_freq == 0:
            return input_signal
        else:
            filtered_waveform_new = lowpass_filter_new(input_signal, cutoff_freq / self.sample_rate)
            return filtered_waveform_new

    def high_pass(self, input_signal, cutoff_freq, q=0.707, index=0):
        # filtered_waveform = julius.lowpass_filter(input_signal, cutoff_freq.item()/44100)
        # filtered_waveform_new = julius.highpass_filter(input_signal, cutoff_freq/44100)
        if cutoff_freq == 0:
            return input_signal
        else:
            filtered_waveform_new = julius.highpass_filter_new(input_signal, cutoff_freq / self.sample_rate)
            return filtered_waveform_new

    def tremolo_by_modulator_params(self, input_signal, amount, freq_m, waveform_m):
        """tremolo effect for an input signal

            This is a kind of AM modulation, where the signal is multiplied as a whole by a given modulator.
            The modulator is shifted such that it resides in range [start, 1], where start is <1 - amount>.
            so start is > 0, such that the original amplitude of the input audio is preserved and there is no phase
            shift due to multiplication by negative number.

            Args:
                self: Self object
                input_signal: Input signal to be used as carrier
                amount: amount of effect, in range [0, 1]
                freq_m: Frequency of modulator in range [0, 20]
                waveform_m: Waveform of modulator. One of [sine, square, triangle, sawtooth]

            Returns:
                A torch with the constructed AM signal

            Raises:
                ValueError: Provided variables are inappropriate
                ValueError: Amount is out of range [-1, 1]
            """
        self.signal_values_sanity_check(amp=1, freq=freq_m, waveform=waveform_m)
        if amount > 1 or amount < 0:
            ValueError("amount is out of range [0, 1]")
        modulator = SynthModules()
        modulator.signal = modulator.oscillator(amp=1, freq=freq_m, phase=0, waveform=waveform_m)
        modulator.signal = amount * (modulator.signal + 1) / 2 + (1 - amount)

        am_signal = input_signal * modulator.signal

        return am_signal

    def tremolo_by_modulator_signal(self, input_signal, modulator_signal, amount):
        """tremolo effect for an input signal

            This is a kind of AM modulation, where the signal is multiplied as a whole by a given modulator.
            The modulator is shifted such that it resides in range [start, 1], where start is <1 - amount>.
            so start is > 0, such that the original amplitude of the input audio is preserved and there is no phase
            shift due to multiplication by negative number.

            Args:
                self: Self object
                input_signal: Input signal to be used as carrier
                modulator_signal: modulator signal to modulate the input
                amount: amount of effect, in range [0, 1]


            Returns:
                A torch with the constructed AM signal

            Raises:
                ValueError: Provided variables are inappropriate
                ValueError: Amount is out of range [-1, 1]
            """
        if isinstance(amount, int):
            if amount > 1 or amount < 0:
                ValueError("amount is out of range [0, 1]")
        if isinstance(amount, list):
            if max(amount) > 1 or min(amount) < 0:
                ValueError("amount is out of range [0, 1]")
        amount = torch.tensor(amount, device=self.device).unsqueeze(dim=1)
        tremolo = torch.add(torch.mul(amount, (modulator_signal + 1) / 2), (1 - amount))

        tremolo_signal = input_signal * tremolo

        return tremolo_signal

    def _standardize_batch_input(self, input_val, requested_dtype, requested_dims):

        # Single scalar input value
        if isinstance(input_val, (float, np.floating, int, np.int)):
            return input_val

        # List, ndarray or tensor input
        if isinstance(input_val, (np.ndarray, list)):
            output_tensor = torch.tensor(input_val, dtype=requested_dtype, device=self.device)
        elif torch.is_tensor(input_val):
            output_tensor = input_val.to(self.device)
        else:
            raise TypeError(f"Unsupported input of type {type(input_val)} to synth module")

        # Add batch dim if doesn't exist
        if output_tensor.ndim == 0:
            output_tensor = torch.unsqueeze(output_tensor, dim=0)
        if output_tensor.ndim == 1 and len(output_tensor) != self.sample_rate:
            output_tensor = torch.unsqueeze(output_tensor, dim=1)
        if output_tensor.ndim == 1 and len(output_tensor) == self.sample_rate:
            output_tensor = torch.unsqueeze(output_tensor, dim=0)

        assert output_tensor.ndim == requested_dims, f"Input has unexpected number of dimensions: {output_tensor.ndim}"

        return output_tensor



    @staticmethod
    # todo: remove all except list instances
    def signal_values_sanity_check(amp, freq, waveform):
        """Check signal properties are reasonable."""
        if isinstance(freq, float):
            if freq < 0 or freq > 20000:
                raise ValueError("Provided frequency is not in range [0, 20000]")
        elif isinstance(freq, list):
            if any(element is not None and (element < 0 or element > 2000) for element in freq):
                raise ValueError("Provided frequency is not in range [0, 20000]")
        if isinstance(amp, int):
            if amp < 0 or amp > 1:
                raise ValueError("Provided amplitude is not in range [0, 1]")
        elif isinstance(amp, list):
            if any(element < 0 or element > 1 for element in amp):
                raise ValueError("Provided amplitude is not in range [0, 1]")
        if isinstance(waveform, str):
            if not any(x in waveform for x in ['sine', 'square', 'triangle', 'sawtooth']):
                raise ValueError("Unknown waveform provided")


def check_adsr_timings(attack_t, decay_t, sustain_t, sustain_level, release_t, signal_duration, num_sounds=1):
    """
    The function checks that:
    1.ADSR timings does not exceed signal duration
    2.Sustain level within [0,1] range

    :exception: throws value error when faulty.
    """
    if num_sounds == 1:
        if attack_t + decay_t + sustain_t + release_t > signal_duration:
            raise ValueError("Provided ADSR durations exceeds signal duration")
        if sustain_level < 0 or sustain_level > 1:
            raise ValueError("Provided sustain level is out of range [0, 1]")
    else:
        for i in range(num_sounds):
            if attack_t[i] + decay_t[i] + sustain_t[i] + release_t[i] > signal_duration:
                raise ValueError("Provided ADSR durations exceeds signal duration")
            if sustain_level[i] < 0 or sustain_level[i] > 1:
                raise ValueError("Provided sustain level is out of range [0, 1]")


def make_envelope_shape(attack_t,
                        decay_t,
                        sustain_t,
                        sustain_level,
                        release_t,
                        signal_duration,
                        sample_rate,
                        device,
                        num_sounds=1):

    if isinstance(attack_t, list) or isinstance(attack_t, np.ndarray):
        attack_t = torch.tensor(attack_t, dtype=torch.float64)
        decay_t = torch.tensor(decay_t, dtype=torch.float64)
        sustain_t = torch.tensor(sustain_t, dtype=torch.float64)
        sustain_level = torch.tensor(sustain_level, dtype=torch.float64)
        release_t = torch.tensor(release_t, dtype=torch.float64)

    # todo: prevent substruction so we get negative numbers here. consider substract only when check_adsr_timngs fail
    # with try catch exception
    epsilon = 1e-5
    attack_t = attack_t - epsilon
    decay_t = decay_t - epsilon
    sustain_t = sustain_t - epsilon
    release_t = release_t - epsilon

    check_adsr_timings(attack_t,
                       decay_t,
                       sustain_t,
                       sustain_level,
                       release_t,
                       signal_duration,
                       num_sounds)

    if num_sounds == 1:
        attack_num_samples = int(signal_duration * sample_rate * attack_t)
        decay_num_samples = int(signal_duration * sample_rate * decay_t)
        sustain_num_samples = int(signal_duration * sample_rate * sustain_t)
        release_num_samples = int(signal_duration * sample_rate * release_t)
    else:
        attack_num_samples = [torch.floor(signal_duration * sample_rate * attack_t[k]) for k in range(num_sounds)]
        decay_num_samples = [torch.floor(signal_duration * sample_rate * decay_t[k]) for k in range(num_sounds)]
        sustain_num_samples = [torch.floor(signal_duration * sample_rate * sustain_t[k]) for k in range(num_sounds)]
        release_num_samples = [torch.floor(signal_duration * sample_rate * release_t[k]) for k in range(num_sounds)]
        attack_num_samples = torch.stack(attack_num_samples)
        decay_num_samples = torch.stack(decay_num_samples)
        sustain_num_samples = torch.stack(sustain_num_samples)
        release_num_samples = torch.stack(release_num_samples)

    if num_sounds == 1:
        sustain_level = sustain_level.item()
    else:
        if torch.is_tensor(sustain_level[0]):
            sustain_level = [sustain_level[i] for i in range(num_sounds)]
            sustain_level = torch.stack(sustain_level)
        else:
            sustain_level = [sustain_level[i] for i in range(num_sounds)]

    envelopes_tensor = torch.tensor((), requires_grad=True).to(device)
    first_time = True
    for k in range(num_sounds):
        if num_sounds == 1:
            attack = torch.linspace(0, 1, attack_num_samples)
            decay = torch.linspace(1, sustain_level, decay_num_samples)
            sustain = torch.full((sustain_num_samples,), sustain_level)
            release = torch.linspace(sustain_level, 0, release_num_samples)
        else:
            # convert 1d vector to scalar tensor to be used in linspace
            # IMPORTANT: lost gradients here! Using int() loses gradients since only float tensors have gradients
            attack_num_samples_sc = attack_num_samples[k].int().squeeze()
            decay_num_samples_sc = decay_num_samples[k].int().squeeze()
            sustain_num_samples_sc = sustain_num_samples[k].int().squeeze()
            sustain_level_sc = sustain_level[k].squeeze()
            release_num_samples_sc = release_num_samples[k].int().squeeze()

            attack = torch.linspace(0, 1, attack_num_samples_sc)
            decay = torch.linspace(1, sustain_level_sc, decay_num_samples_sc)
            sustain = torch.full((sustain_num_samples_sc,), sustain_level_sc)
            release = torch.linspace(sustain_level_sc, 0, release_num_samples_sc)

        envelope = torch.cat((attack, decay, sustain, release)).to(device)

        envelope_num_samples = envelope.shape[0]
        signal_num_samples = int(signal_duration * sample_rate)
        if envelope_num_samples <= signal_num_samples:
            padding = torch.zeros((signal_num_samples - envelope_num_samples), device=device)
            envelope = torch.cat((envelope, padding))
        else:
            raise ValueError("Envelope length exceeds signal duration")

        if first_time:
            if num_sounds == 1:
                envelopes_tensor = envelope
            else:
                envelopes_tensor = torch.cat((envelopes_tensor, envelope), dim=0).unsqueeze(dim=0)
                first_time = False
        else:
            envelope = envelope.unsqueeze(dim=0)
            envelopes_tensor = torch.cat((envelopes_tensor, envelope), dim=0)

    return envelopes_tensor

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
    a = SynthModules()
    b = SynthModules()
    # b.oscillator(1, 5, 0, 'sine')
    a.oscillator_fm(amp_c=1, freq_c=440, waveform='sine', mod_index=10, modulator=b.signal)
    # a.oscillator(amp=1, freq=100, phase=0, waveform='sine')
    # a.adsr_envelope(attack_t=0.5, decay_t=0, sustain_t=0.5, sustain_level=0.5, release_t=0)
    # plt.plot(a.signal)
    # plt.show
    # play_obj = sa.play_buffer(a.signal.numpy(), num_channels=1, bytes_per_sample=4, sample_rate=a.sample_rate)
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
