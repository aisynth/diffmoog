from typing import Callable, Union

import torch
import torchaudio
from torch import nn

from model.loss.spectral_loss_presets import loss_presets
from synth.synth_constants import SynthConstants

loss_type_to_function = {'mag': lambda x: x,
                         'delta_time': lambda x: torch.diff(x, n=1, dim=1),
                         'delta_freq': lambda x: torch.diff(x, n=1, dim=2),
                         'cumsum_time': lambda x: torch.cumsum(x, dim=2),
                         'cumsum_freq': lambda x: torch.cumsum(x, dim=1),
                         'logmag': lambda x: torch.log(x + 1)}

class BaseSpectralLoss(nn.Module):
    """ Base class for spectral loss"""
    def __init__(self, loss_preset: Union[str, dict], synth_constants: SynthConstants, device='cuda:0'):
        super().__init__()

        self.loss_preset = loss_presets[loss_preset] if isinstance(loss_preset, str) else loss_preset
        self.device = device
        self.sample_rate = synth_constants.sample_rate

        if self.loss_preset['multi_spectral_loss_norm'] == 'L1':
            self.criterion = nn.L1Loss()
        elif self.loss_preset['multi_spectral_loss_norm'] == 'L2':
            self.criterion = nn.MSELoss()
        else:
            raise ValueError("unknown loss type")

    def calc_loss(self, loss_type: str, pre_loss_fn: Callable, target_mag: torch.Tensor, value_mag: torch.Tensor,
                  n_fft: int, step: int) -> (float, float):

        if self.loss_preset.get(f'multi_spectral_{loss_type}_weight', 0) == 0 or \
                step < self.loss_preset.get(f'multi_spectral_{loss_type}_warmup', -1):
            return torch.tensor(0.0), torch.tensor(0.0)

        target = pre_loss_fn(target_mag)
        value = pre_loss_fn(value_mag)

        loss_val = self.criterion(target, value)

        weighted_loss_val = loss_val * self.loss_preset[f'multi_spectral_{loss_type}_weight']

        if self.loss_preset['normalize_loss_by_nfft']:
            n_fft_normalization_factor = (300.0 / n_fft)
            if loss_type in ['cumsum_time']:
                n_fft_normalization_factor = 1
            elif loss_type not in ['delta_time', 'logmag']:
                n_fft_normalization_factor = n_fft_normalization_factor ** 2
            weighted_loss_val *= n_fft_normalization_factor

        if self.loss_preset.get(f'multi_spectral_{loss_type}_gradual', False):
            warmup = self.loss_preset.get(f'multi_spectral_{loss_type}_warmup_steps')
            gradual_loss_factor = min((step / warmup), 1)
            weighted_loss_val *= gradual_loss_factor

        return loss_val, weighted_loss_val

class SpectralLoss(BaseSpectralLoss):
    """From DDSP code:
    https://github.com/magenta/ddsp/blob/8536a366c7834908f418a6721547268e8f2083cc/ddsp/losses.py#L144"""
    """Multiscale spectrogram loss.
    This loss is the bread-and-butter of comparing two audio signals. It offers
    a range of options to compare spectrograms, many of which are redundant, but
    emphasize different aspects of the signal. By far, the most common comparisons
    are magnitudes (mag_weight) and log magnitudes (logmag_weight).
    """

    def __init__(self, loss_preset: Union[str, dict], synth_constants: SynthConstants, device='cuda:0'):
        """Initializes the loss.
    Args:
        loss_preset: A string or dict of preset parameters for the loss.
        synth_constants: SynthConstants object containing constants for the synthesizer.
        device: Device to run the loss on.

    """

        super().__init__(loss_preset, synth_constants, device)
        self.loss_transform = self.loss_preset['transform']

        self.spectrogram_ops = {}
        for fft_size in self.loss_preset['fft_sizes']:
            frame_overlap = self.loss_preset.get('frame_overlap', 0.75)  # Default frame overlap of 75%
            hop_size = int(fft_size * (1 - frame_overlap))  # Compute the hop size based on the frame overlap
            self.add_spectrogram_ops(fft_size, hop_size)

    def add_spectrogram_ops(self, fft_size, hop_size):
        if self.loss_transform == 'BOTH' or self.loss_transform == 'SPECTROGRAM':
            spec_transform = torchaudio.transforms.Spectrogram(n_fft=fft_size,
                                                               hop_length=hop_size,
                                                               power=2.0).to(self.device)
            self.spectrogram_ops[f'{fft_size}_spectrogram'] = spec_transform

        if self.loss_transform == 'BOTH' or self.loss_transform == 'MEL_SPECTROGRAM':
            n_mels = self.loss_preset.get('n_mels', 256)
            f_min = self.loss_preset.get('f_min', 0.0)
            f_max = self.loss_preset.get('f_max', None)
            mel_spec_transform = torchaudio.transforms.MelSpectrogram(sample_rate=self.sample_rate,
                                                                      n_fft=fft_size,
                                                                      hop_length=hop_size,
                                                                      n_mels=n_mels,
                                                                      f_min=f_min,
                                                                      f_max=f_max,
                                                                      power=2.0).to(self.device)
            self.spectrogram_ops[f'{fft_size}_mel'] = mel_spec_transform

    def call(self, target_audio, predicted_audio, step: int, return_spectrogram: bool = False):
        """ execute multi-spectral loss computation between two audio signals

        Args:
            :param target_audio: target audio signal
            :param predicted_audio: predicted audio signal
            :param step: current training step
            :param return_spectrogram: if True, return spectrograms of target and predicted audio signals
        """
        loss = torch.tensor(0.0, requires_grad=True).to(self.device)

        # Compute loss for each fft size.
        loss_dict, weighted_loss_dict = {}, {}
        spectrograms_dict = {}
        for loss_name, loss_op in self.spectrogram_ops.items():

            n_fft = loss_op.n_fft

            target_mag = loss_op(target_audio.float())
            value_mag = loss_op(predicted_audio.float())

            c_loss = torch.tensor(0.0, requires_grad=True).to(self.device)
            for loss_type, pre_loss_fn in loss_type_to_function.items():
                raw_loss, weighted_loss = self.calc_loss(loss_type, pre_loss_fn, target_mag, value_mag, n_fft, step)

                if weighted_loss == 0:
                    continue

                loss_dict[f"{loss_name}_{loss_type}"] = raw_loss
                weighted_loss_dict[f"{loss_name}_{loss_type}"] = weighted_loss

                c_loss += weighted_loss

            loss += c_loss

            spectrograms_dict[loss_name] = {'pred': value_mag.detach(), 'target': target_mag.detach()}

        if return_spectrogram:
            return loss, loss_dict, weighted_loss_dict, spectrograms_dict

        return loss, loss_dict, weighted_loss_dict

class ControlSpectralLoss(BaseSpectralLoss):
    """
    This loss aims at comparing control signals such as LFOs. Control signals contain only low frequency components, and
     therefore need to be down sampled and compared using a Spectrogram with lower nyquist frequency.
    """

    def __init__(self, preset_name: str, synth_constants: SynthConstants, device='cuda:0'):
        super().__init__(preset_name, synth_constants, device)

        target_sample_rate = synth_constants.lfo_signal_sampling_rate
        self.resample = torchaudio.transforms.Resample(self.sample_rate, target_sample_rate)
        self.spectrogram = torchaudio.transforms.Spectrogram(n_fft=128,
                                                             win_length=128,
                                                             hop_length=512,
                                                             center=True,
                                                             pad_mode="reflect",
                                                             power=2.0).to(self.device)
        self.db = torchaudio.transforms.AmplitudeToDB(stype='magnitude')

    def call(self, target_control_signal, predicted_control_signal, step: int, return_spectrogram: bool = False):
        """ execute spectral loss computation between two control signals

        Args:
            :param target_control_signal: target control signal
            :param predicted_control_signal: predicted control signal
            :param step: current training step
            :param return_spectrogram: if True, return spectrograms of target and predicted audio signals
        """
        loss = 0.0

        resampled_target_control_signal = self.resample(target_control_signal)
        resampled_predicted_control_signal = self.resample(predicted_control_signal)
        target_mag = self.spectrogram(resampled_target_control_signal.float())
        value_mag = self.spectrogram(resampled_predicted_control_signal.float())

        # fig, axs = plt.subplots(1, 1, figsize=(13, 2.5))
        # im = axs.imshow(librosa.power_to_db(target_mag[0].detach().cpu().numpy()), origin="lower", aspect="auto",
        #                 cmap='inferno')

        c_loss = 0.0
        n_fft = 128
        loss_dict, weighted_loss_dict = {}, {}
        spectrograms_dict = {}
        for loss_type, pre_loss_fn in loss_type_to_function.items():
            raw_loss, weighted_loss = self.calc_loss(loss_type, pre_loss_fn, target_mag, value_mag, n_fft, step)

            if weighted_loss == 0:
                continue

            loss_dict[f"{loss_type}"] = raw_loss
            weighted_loss_dict[f"{loss_type}"] = weighted_loss

            c_loss += weighted_loss

        loss += c_loss

        spectrograms_dict['control_signal_spectrograms'] = {'pred': value_mag.detach(),
                                                            'target': target_mag.detach()}

        if return_spectrogram:
            return loss, loss_dict, weighted_loss_dict, spectrograms_dict

        return loss, loss_dict, weighted_loss_dict





