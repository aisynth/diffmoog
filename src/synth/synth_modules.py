#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 31 15:41:38 2021

@author: Moshe Laufer, Noy Uzrad
"""
from abc import ABC, abstractmethod
from typing import Dict, Tuple

import numpy as np

import torch
import math

from torchaudio.functional.filtering import lowpass_biquad, highpass_biquad
from torchaudio.transforms import Spectrogram, GriffinLim
import matplotlib.pyplot as plt
from model import helper
import julius
from julius.lowpass import lowpass_filter_new
from synth.synth_constants import SynthConstants
from utils.types import TensorLike

try:
    from functorch import vmap
    has_vmap = True
except ModuleNotFoundError:
    has_vmap = False


PI = math.pi
TWO_PI = 2 * PI


class SynthModule(ABC):

    def __init__(self, name: str, device: str, synth_structure: SynthConstants):
        self.synth_structure = synth_structure
        self.device = device
        self.name = name

    @abstractmethod
    def generate_sound(self, input_signal: torch.Tensor, modulator_signal: torch.Tensor, params: dict,
                       sample_rate: int, signal_duration: float, batch_size: int = 1):
        pass

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

    def standardize_input(self, input_val, requested_dtype, requested_dims: int, batch_size: int,
                          value_range: Tuple = None):

        # Single scalar input value
        if isinstance(input_val, (float, np.floating, int, np.int)):
            assert batch_size == 1, f"Input expected to be of batch size {batch_size} but is scalar"
            input_val = torch.tensor(input_val, dtype=requested_dtype, device=self.device)

        # List, ndarray or tensor input
        if isinstance(input_val, (np.ndarray, list)):
            output_tensor = torch.tensor(input_val, dtype=requested_dtype, device=self.device)
        elif torch.is_tensor(input_val):
            output_tensor = input_val.to(dtype=requested_dtype, device=self.device)
        else:
            raise TypeError(f"Unsupported input of type {type(input_val)} to synth module")

        # Add batch dim if doesn't exist
        if output_tensor.ndim == 0:
            output_tensor = torch.unsqueeze(output_tensor, dim=0)
        if output_tensor.ndim == 1 and len(output_tensor) == batch_size:
            output_tensor = torch.unsqueeze(output_tensor, dim=1)
        elif output_tensor.ndim == 1 and len(output_tensor) != batch_size:
            assert batch_size == 1, f"Input expected to be of batch size {batch_size} but is of batch size 1, " \
                                    f"shape {output_tensor.shape}"
            output_tensor = torch.unsqueeze(output_tensor, dim=0)

        assert output_tensor.ndim == requested_dims, f"Input has unexpected number of dimensions: {output_tensor.ndim}"

        if value_range is not None:
            assert torch.all(output_tensor >= value_range[0]) and torch.all(output_tensor <= value_range[1]), \
                f"Parameter value outside of expected range {value_range}"

        return output_tensor

    def verify_input_params(self, params_to_test: dict):

        expected_params = self.synth_structure.modular_synth_params[self.name]
        assert set(params_to_test.keys()) == set(expected_params),\
            f'Expected input parameters {expected_params} but got {list(params_to_test.keys())}'


class Oscillator(SynthModule):

    def __init__(self, name:str, device: str, synth_structure: SynthConstants):
        super().__init__(name=name, device=device, synth_structure=synth_structure)
        self.wave_type_indices = {k: torch.tensor(v, dtype=torch.long, device=self.device).squeeze()
                                  for k, v in self.synth_structure.wave_type_dict.items()}

    def generate_sound(self, input_signal: torch.Tensor, modulator_signal: torch.Tensor, params: dict,
                       sample_rate: int, signal_duration: float, batch_size: int = 1):
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

        self.verify_input_params(params)

        amp = self.standardize_input(params['amp'], requested_dtype=torch.float32, requested_dims=2,
                                     batch_size=batch_size)
        freq = self.standardize_input(params['freq'], requested_dtype=torch.float32, requested_dims=2,
                                      batch_size=batch_size)

        t = torch.linspace(0, signal_duration, steps=int(sample_rate * signal_duration), requires_grad=True)
        sine_wave, square_wave, sawtooth_wave = self._generate_wave_tensors(t, amp, freq, phase_mod=0)

        waves_tensor = torch.stack([sine_wave, square_wave, sawtooth_wave])
        oscillator_tensor = self._mix_waveforms(waves_tensor, params['waveform'], self.wave_type_indices)

        return oscillator_tensor

    @staticmethod
    def _generate_wave_tensors(t, amp, freq, phase_mod):

        sine_wave = amp * torch.sin(TWO_PI * freq * t + phase_mod)

        square_wave = amp * torch.sign(torch.sin(TWO_PI * freq * t + phase_mod))

        sawtooth_wave = 2 * (t * freq - torch.floor(0.5 + t * freq))  # Sawtooth closed form

        # Phase shift (by normalization to range [0,1] and modulo operation)
        sawtooth_wave = (((sawtooth_wave + 1) / 2) + phase_mod / TWO_PI) % 1
        sawtooth_wave = amp * (sawtooth_wave * 2 - 1)  # re-normalization to range [-amp, amp]

        return sine_wave, square_wave, sawtooth_wave


class FMOscillator(Oscillator):

    def __init__(self, device: str, synth_structure: SynthConstants, waveform: str = None):

        if waveform is not None:
            assert waveform in ['sine', 'square', 'saw'], f'Unexpected waveform {waveform} given to FMOscillator'
            name = f"fm_{waveform}"
        else:
            name = 'fm'

        super().__init__(name=name, device=device, synth_structure=synth_structure)

        self.waveform = waveform

    def generate_sound(self, input_signal: torch.Tensor, modulator_signal: torch.Tensor, params: dict,
                       sample_rate: int, signal_duration: float, batch_size: int = 1):

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

        self.verify_input_params(params)

        parsed_params = {}
        for k in ['amp_c', 'freq_c', 'mod_index', 'modulator']:
            parsed_params[k] = self.standardize_input(params[k], requested_dtype=torch.float32, requested_dims=2,
                                                      batch_size=batch_size)

        t = torch.linspace(0, signal_duration, steps=int(sample_rate * signal_duration), requires_grad=True)
        wave_tensors = self._generate_modulated_wave_tensors(t, **parsed_params)

        if self.waveform is not None:
            return wave_tensors[self.waveform]

        waves_tensor = torch.stack([wave_tensors['sine'], wave_tensors['square'], wave_tensors['saw']])
        oscillator_tensor = self._mix_waveforms(waves_tensor, params['waveform'], self.wave_type_indices)

        return oscillator_tensor

    def _generate_modulated_wave_tensors(self, t, amp_c, freq_c, mod_index, modulator):

        wave_tensors = {}

        if self.waveform == 'sine' or self.waveform is None:
            fm_sine_wave = amp_c * torch.sin(TWO_PI * freq_c * t + mod_index * torch.cumsum(modulator, dim=1))
            wave_tensors['sine'] = fm_sine_wave

        if self.waveform == 'square' or self.waveform is None:
            fm_square_wave = amp_c * torch.sign(torch.sin(TWO_PI * freq_c * t + mod_index *
                                                          torch.cumsum(modulator, dim=1)))
            wave_tensors['square'] = fm_square_wave

        if self.waveform == 'saw' or self.waveform is None:
            fm_sawtooth_wave = 2 * (t * freq_c - torch.floor(0.5 + t * freq_c))
            fm_sawtooth_wave = (((fm_sawtooth_wave + 1) / 2) + mod_index * torch.cumsum(modulator, dim=1) / TWO_PI) % 1
            fm_sawtooth_wave = amp_c * (fm_sawtooth_wave * 2 - 1)
            wave_tensors['saw'] = fm_sawtooth_wave

        return wave_tensors


class Filter(SynthModule):

    def __init__(self, device: str, synth_structure: SynthConstants, filter_type: str = None):
        super().__init__(name='filter', device=device, synth_structure=synth_structure)

        if filter_type is not None:
            assert filter_type in ['lowpass', 'highpass'], f'Got unexpected filter type {filter_type} in Filter'
            self.name = f'{filter_type}_filter'

        self.filter_type = filter_type

        self.filter_type_indices = {k: torch.tensor(v, dtype=torch.long, device=self.device).squeeze()
                                    for k, v in synth_structure.filter_type_dict.items()}

    def generate_sound(self, input_signal: torch.Tensor, modulator_signal: torch.Tensor, params: dict,
                       sample_rate: int, signal_duration: float, batch_size: int = 1):

        self.verify_input_params(params)

        filter_freq = self.standardize_input(params['filter_freq'], requested_dtype=torch.float64, requested_dims=2,
                                             batch_size=batch_size)

        assert torch.all(filter_freq <= (sample_rate / 2)), "Filter cutoff frequency higher then Nyquist." \
                                                            " Please check config"

        filtered_signal = self._generate_filtered_signal(input_signal, filter_freq, sample_rate, batch_size)
        for k, v in filtered_signal.items():
            if torch.any(torch.isnan(v)):
                raise RuntimeError("Synth filter module: Signal has NaN. Exiting...")

        if self.filter_type is not None:
            return filtered_signal[self.filter_type]

        waves_tensor = torch.stack([filtered_signal['lowpass'], filtered_signal['highpass']])
        filtered_signal_tensor = self._mix_waveforms(waves_tensor, params['filter_type'], self.filter_type_indices)

        return filtered_signal_tensor

    def _generate_filtered_signal(self, input_signal: torch.Tensor, filter_freq: torch.tensor, sample_rate: int,
                                  batch_size: int) -> dict:

        filtered_signals = {}
        if self.filter_type == 'lowpass' or self.filter_type is None:
            if has_vmap:
                low_pass_signals = vmap(lowpass_biquad)(input_signal.double(), cutoff_freq=filter_freq,
                                                        sample_rate=sample_rate)
            else:
                low_pass_signals = [self.low_pass(input_signal[i], filter_freq[i].cpu(), sample_rate) for i in
                                    range(batch_size)]
            filtered_signals['lowpass'] = torch.stack(low_pass_signals)

        if self.filter_type == 'highpass' or self.filter_type is None:
            if has_vmap:
                high_pass_signals = vmap(highpass_biquad)(input_signal.double(), cutoff_freq=filter_freq,
                                                          sample_rate=sample_rate)
            else:
                high_pass_signals = [self.high_pass(input_signal[i], filter_freq[i].cpu(), sample_rate) for i in
                                     range(batch_size)]
            filtered_signals['highpass'] = torch.stack(high_pass_signals)

        return filtered_signals

    @staticmethod
    def low_pass(input_signal, cutoff_freq, sample_rate, q=0.707, index=0):
        if cutoff_freq == 0:
            return input_signal
        else:
            filtered_waveform_new = lowpass_filter_new(input_signal, cutoff_freq / sample_rate)
            return filtered_waveform_new

    @staticmethod
    def high_pass(input_signal, cutoff_freq, sample_rate, q=0.707, index=0):
        if cutoff_freq == 0:
            return input_signal
        else:
            filtered_waveform_new = julius.highpass_filter_new(input_signal, cutoff_freq / sample_rate)
            return filtered_waveform_new


class Tremolo(SynthModule):

    def __init__(self, device: str, synth_structure: SynthConstants):
        super().__init__(name='tremolo', device=device, synth_structure=synth_structure)

    def generate_sound(self, input_signal: torch.Tensor, modulator_signal: torch.Tensor, params: dict,
                       sample_rate: int, signal_duration: float, batch_size: int = 1):
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

        self.verify_input_params(params)

        amount = self.standardize_input(params['amount'], requested_dtype=torch.float64, requested_dims=2,
                                        batch_size=batch_size, value_range=(0, 1))

        tremolo = torch.add(torch.mul(amount, (modulator_signal + 1) / 2), (1 - amount))

        tremolo_signal = input_signal * tremolo

        return tremolo_signal


def get_synth_module(op_name: str, device: str, synth_structure: SynthConstants):

    if op_name in ['osc']:
        return Oscillator(op_name, device, synth_structure)
    elif op_name[:2] == 'fm':
        if len(op_name) > 2:
            waveform = op_name.split('_')[1]
            assert waveform in ['sine', 'saw', 'square']
        else:
            waveform = None
        return FMOscillator(device, synth_structure, waveform)
    elif 'filter' in op_name:
        if op_name != 'filter':
            filter_type = op_name.split('_')[0]
        else:
            filter_type = None
        return Filter(device, synth_structure, filter_type)
    elif op_name == 'tremolo':
        return Tremolo(device, synth_structure)
    else:
        raise ValueError(f"Unsupported synth module {op_name}")



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


    def mix_signal(self, new_signal, factor):
        """Signal superposition. factor balances the mix
        1 - original signal only, 0 - new signal only, 0.5 evenly balanced. """
        if factor < 0 or factor > 1:
            raise ValueError("Provided factor value is out of range [0, 1]")
        self.signal = factor * self.signal + (1 - factor) * new_signal

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
