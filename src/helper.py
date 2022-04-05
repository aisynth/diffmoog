import torch
import os

from matplotlib.backends.backend_agg import FigureCanvasAgg
from torch import Tensor
import torchaudio
import matplotlib
import matplotlib.pyplot as plt
import librosa
from torch.utils.tensorboard import SummaryWriter

from synth import synth_config
from torch import nn
from config import SynthConfig, Config
from pathlib import Path
import math
import numpy as np


def get_device(gpu_index: int = 0):
    if int(gpu_index) >= 0 and torch.cuda.is_available():
        device = torch.device("cuda:" + str(gpu_index))
        print('using device: ', torch.cuda.get_device_name(device))
    else:
        device = torch.device("cpu")
        # print('using cpu')
    return device


def move_to(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            res[k] = move_to(v, device)
        return res
    elif isinstance(obj, list):
        res = []
        for v in obj:
            res.append(move_to(v, device))
        return res
    else:
        raise TypeError("Invalid type for move_to")


def spectrogram_transform():
    return torchaudio.transforms.Spectrogram(  # win_length default = n_fft. hop_length default = win_length / 2
                                             n_fft=512,
                                             power=2.0)


def mel_spectrogram_transform(sample_rate):
    return torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,
                                                n_fft=1024,
                                                hop_length=256,
                                                n_mels=128,
                                                power=2.0,
                                                f_min=0,
                                                f_max=8000)


def alt_mel_spectrogram_transform(sample_rate):
    return torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,
                                                n_fft=1024,
                                                hop_length=256,
                                                n_mels=128,
                                                power=1.0,
                                                f_min=0,
                                                f_max=8000)


amplitude_to_db_transform = torchaudio.transforms.AmplitudeToDB()


# log_mel_spec_transform = torch.nn.Sequential(mel_spectrogram_transform, amplitude_to_db_transform).to(get_device())

def map_classification_params_from_ints(params_dic: dict):
    """ map classification params from ints (inverse operation of map_classification_params_to_ints) """

    mapped_params_dict = {}
    for key, val in params_dic.items():
        if key in synth_config.CLASSIFICATION_PARAM_LIST:
            if key == "osc1_freq" or key == "osc2_freq":
                # classification_params_dic[key] = synth.OSC_FREQ_DIC[round(val, 4)]
                mapped_params_dict[key] = torch.tensor([synth_config.OSC_FREQ_DIC_INV[x.item()] for x in val])
            if "wave" in key:
                mapped_params_dict[key] = [synth_config.WAVE_TYPE_DIC_INV[x.item()] for x in val]
            elif "filter_type" == key:
                mapped_params_dict[key] = [synth_config.FILTER_TYPE_DIC_INV[x.item()] for x in val]

    return mapped_params_dict


def clamp_regression_params(parameters_dict: dict, synth_cfg: SynthConfig, cfg: Config):
    """ clamp regression parameters to values that can be interpreted by the synth module,
        add classification parameters as is"""

    clamped_params_dict = {}
    for key, val in parameters_dict.items():
        operation = val['operation']
        operation_params = val['params']

        if operation == 'osc':
            clamped_params_dict[key] = \
                {'operation': operation,
                 'params':
                     {'amp': operation_params['amp'],
                      'freq': torch.clamp(operation_params['freq'], min=0, max=synth_cfg.oscillator_freq),
                      'waveform': operation_params['waveform']
                      }
                 }

        if operation == 'lfo':
            clamped_params_dict[key] = \
                {'operation': operation,
                 'params':
                     {'amp': operation_params['amp'],
                      'freq': torch.clamp(operation_params['freq'], min=0, max=synth_cfg.max_lfo_freq)
                      }
                 }

        elif operation == 'fm':
            clamped_params_dict[key] = \
                {'operation': operation,
                 'params':
                     {'amp_c': operation_params['amp_c'],
                      'freq_c':
                          torch.clamp(operation_params['freq_c'],
                                      min=0, max=synth_cfg.oscillator_freq),
                      'waveform': operation_params['waveform'],
                      'mod_index': torch.clamp(operation_params['freq_c'], min=0, max=synth_cfg.max_mod_index)
                      }
                 }

        elif operation == 'filter':
            clamped_params_dict[key] = \
                {'operation': operation,
                 'params':
                     {'filter_type': operation_params['filter_type'],
                      'filter_freq': torch.clamp(operation_params['filter_freq'],
                                                 min=synth_cfg.min_filter_freq,
                                                 max=synth_cfg.max_filter_freq)
                      }
                 }

        elif operation == 'env_adsr':
            attack_t = torch.clamp(operation_params['attack_t'], min=0, max=cfg.signal_duration_sec)
            decay_t = torch.clamp(operation_params['decay_t'], min=0, max=cfg.signal_duration_sec)
            sustain_t = torch.clamp(operation_params['release_t'], min=0, max=cfg.signal_duration_sec)
            release_t = torch.clamp(operation_params['release_t'], min=0, max=cfg.signal_duration_sec)

            clamped_attack, clamped_decay, clamped_sustain, clamped_release = \
                clamp_adsr_superposition(attack_t, decay_t, sustain_t, release_t)

            clamped_params_dict[key] = \
                {'operation': operation,
                 'params':
                     {'attack_t': clamped_attack,
                      'decay_t': clamped_decay,
                      'sustain_t': clamped_sustain,
                      'sustain_level': torch.clamp(operation_params['sustain_level'], min=0,
                                                   max=synth_config.MAX_AMP),
                      'release_t': clamped_release
                      }
                 }

    return clamped_params_dict


def clamp_adsr_superposition(attack_t, decay_t, sustain_t, release_t, cfg: Config):
    """This function clamps the superposition of adsr times, so it does not exceed signal length"""

    adsr_length_in_sec = attack_t + decay_t + sustain_t + release_t

    adsr_clamp_indices = torch.nonzero(adsr_length_in_sec >= cfg.signal_duration_sec, as_tuple=True)[0]

    normalized_attack_list = []
    normalized_decay_list = []
    normalized_sustain_list = []
    normalized_release_list = []

    for i in range(adsr_length_in_sec.shape[0]):
        if i in adsr_clamp_indices.tolist():
            # add small number to normalization to prevent numerical issue where the sum exceeds 1
            normalization_value = adsr_length_in_sec[i] + 1e-3
            normalized_attack = attack_t[i] / normalization_value
            normalized_decay = decay_t[i] / normalization_value
            normalized_sustain = sustain_t[i] / normalization_value
            normalized_release = release_t[i] / normalization_value

        else:
            normalized_attack = attack_t[i]
            normalized_decay = decay_t[i]
            normalized_sustain = sustain_t[i]
            normalized_release = release_t[i]

        normalized_attack_list.append(normalized_attack)
        normalized_decay_list.append(normalized_decay)
        normalized_sustain_list.append(normalized_sustain)
        normalized_release_list.append(normalized_release)

    normalized_attack_tensor = torch.stack(normalized_attack_list)
    normalized_decay_tensor = torch.stack(normalized_decay_list)
    normalized_sustain_tensor = torch.stack(normalized_sustain_list)
    normalized_release_tensor = torch.stack(normalized_release_list)

    return normalized_attack_tensor, normalized_decay_tensor, normalized_sustain_tensor, normalized_release_tensor


class Normalizer:
    """ normalize/de-normalise regression parameters"""

    def __init__(self, signal_duration_sec, synth_cfg: SynthConfig):
        self.mod_index_normalizer = MinMaxNormaliser(target_min_val=0,
                                                     target_max_val=1,
                                                     original_min_val=0,
                                                     original_max_val=synth_cfg.max_mod_index)

        self.lfo_freq_normalizer = MinMaxNormaliser(target_min_val=0,
                                                    target_max_val=1,
                                                    original_min_val=0,
                                                    original_max_val=synth_cfg.max_lfo_freq)

        self.lfo_phase_normalizer = MinMaxNormaliser(target_min_val=0,
                                                     target_max_val=1,
                                                     original_min_val=0,
                                                     original_max_val=math.pi)

        self.adsr_normalizer = MinMaxNormaliser(target_min_val=0,
                                                target_max_val=1,
                                                original_min_val=0,
                                                original_max_val=signal_duration_sec)

        self.filter_freq_normalizer = MinMaxNormaliser(target_min_val=0,
                                                       target_max_val=1,
                                                       original_min_val=0,
                                                       original_max_val=synth_cfg.max_filter_freq)

        self.oscillator_freq_normalizer = MinMaxNormaliser(target_min_val=0,
                                                           target_max_val=1,
                                                           original_min_val=0,
                                                           original_max_val=synth_cfg.oscillator_freq)

    def normalize(self, parameters_dict: dict):
        normalized_params_dict = {
            'osc1_mod_index': self.mod_index_normalizer.normalise(parameters_dict['osc1_mod_index']),
            'lfo1_freq': self.lfo_freq_normalizer.normalise(parameters_dict['lfo1_freq']),
            'lfo1_phase': self.lfo_phase_normalizer.normalise(parameters_dict['lfo1_phase']),
            'osc2_mod_index': self.mod_index_normalizer.normalise(parameters_dict['osc2_mod_index']),
            'lfo2_freq': self.lfo_freq_normalizer.normalise(parameters_dict['lfo2_freq']),
            'filter_freq': self.filter_freq_normalizer.normalise(parameters_dict['filter_freq']),
            'attack_t': self.adsr_normalizer.normalise(parameters_dict['attack_t']),
            'decay_t': self.adsr_normalizer.normalise(parameters_dict['decay_t']),
            'sustain_t': self.adsr_normalizer.normalise(parameters_dict['sustain_t']),
            'release_t': self.adsr_normalizer.normalise(parameters_dict['release_t'])}

        return normalized_params_dict

    def denormalize(self, parameters_dict: dict):

        denormalized_params_dict = {}
        for key, val in parameters_dict.items():
            operation = val['operation']
            params = val['params']

            if operation == 'osc':
                denormalized_params_dict[key] = \
                    {'operation': operation,
                     'params':
                         {'amp': params['amp'],
                          'freq': self.oscillator_freq_normalizer.denormalise(params['freq']),
                          'waveform': params['waveform']
                          }
                     }

            elif operation == 'lfo':
                denormalized_params_dict[key] = \
                    {'operation': operation,
                     'params':
                         {'amp': params['amp'],
                          'freq': self.lfo_freq_normalizer.denormalise(params['freq'])
                          }
                     }

            elif operation == 'fm':
                denormalized_params_dict[key] = \
                    {'operation': operation,
                     'params':
                         {'amp_c': params['amp_c'],
                          'freq_c': self.oscillator_freq_normalizer.denormalise(params['freq_c']),
                          'waveform': params['waveform'],
                          'mod_index': self.mod_index_normalizer.denormalise(params['mod_index'])
                          }
                     }

            elif operation == 'filter':
                denormalized_params_dict[key] = \
                    {'operation': operation,
                     'params':
                         {'filter_type': params['filter_type'],
                          'filter_freq': self.filter_freq_normalizer.denormalise(params['filter_freq'])
                          }
                     }

            elif operation == 'env_adsr':
                denormalized_params_dict[key] = \
                    {'operation': operation,
                     'params':
                         {'attack_t': self.adsr_normalizer.denormalise(params['attack_t']),
                          'decay_t': self.adsr_normalizer.denormalise(params['decay_t']),
                          'sustain_t': self.adsr_normalizer.denormalise(params['sustain_t']),
                          'sustain_level': params['sustain_level'],
                          'release_t': self.filter_freq_normalizer.denormalise(params['release_t'])
                          }
                     }

        return denormalized_params_dict


def plot_spectrogram(spec, scale='linear', title=None, x_label='frame', ylabel='freq_bin', aspect='auto', xmax=None):
    matplotlib.use('Qt5Agg')
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or 'Spectrogram (db)')
    axs.set_ylabel(ylabel)
    axs.set_xlabel(x_label)
    if scale == 'linear':
        im = axs.imshow(spec, origin='lower', aspect=aspect)
    elif scale == 'dB:':
        im = axs.imshow(librosa.power_to_db(spec), origin='lower', aspect=aspect)
    if xmax:
        axs.set_xlim((0, xmax))
    fig.colorbar(im, ax=axs)
    return fig
    # plt.show(block=False)


class MinMaxNormaliser:
    """MinMaxNormaliser applies min max normalisation to a tensor"""

    def __init__(self, target_min_val, target_max_val, original_min_val, original_max_val):
        self.target_min_val = target_min_val
        self.target_max_val = target_max_val
        self.original_max_val = original_max_val
        self.original_min_val = original_min_val

    def normalise(self, array):
        norm_array = (array - self.original_min_val) / (self.original_max_val - self.original_min_val)
        return norm_array

    def denormalise(self, norm_array):
        array = (norm_array - self.target_min_val) / (self.target_max_val - self.target_min_val)
        array = array * (self.original_max_val - self.original_min_val) + self.original_min_val
        return array


class LogNormaliser:
    """LogNormaliser applies log normalisation to a tensor"""

    def __init__(self):
        pass

    def normalise(self, array):
        # add small value to prevent -inf for log(0)
        norm_array = torch.log(array + 1e-10)
        return norm_array

    def denormalise(self, norm_array):
        array = torch.exp(norm_array) - 1e-10
        return array


# from https://github.com/pytorch/pytorch/issues/61292
def linspace(start: Tensor, stop: Tensor, num: Tensor, device):
    """
    Creates a tensor of shape [num, *start.shape] whose values are evenly spaced from start to end, inclusive.
    Replicates but the multi-dimensional bahaviour of numpy.linspace in PyTorch.
    """
    start = start.to(device)
    stop = stop.to(device)
    num = num.to(device)
    # create a tensor of 'num' steps from 0 to 1

    # todo: make sure ADSR behavior is differentiable. arange has to know to get tensors
    # OPTION1
    # arange_list_of_tensors = []
    # current_tensor = torch.tensor([0], dtype=torch.float32, device=get_device(), requires_grad=True)
    # while current_tensor.item() < num.item():
    #     arange_list_of_tensors.append(current_tensor)
    #     current_tensor = current_tensor + 1
    #
    # arange_tensor1 = torch.stack(arange_list_of_tensors).to(get_device()).squeeze()

    # OPTION2
    arange_tensor2 = torch.arange(num, dtype=torch.float32, device=device, requires_grad=True)

    steps = arange_tensor2 / (num - 1)

    # reshape the 'steps' tensor to [-1, *([1]*start.ndim)] to allow for broadcasting
    # - using 'steps.reshape([-1, *([1]*start.ndim)])' would be nice here but torchscript
    #   "cannot statically infer the expected size of a list in this contex", hence the code below
    for i in range(start.ndim):
        steps = steps.unsqueeze(-1)

    # the output starts at 'start' and increments until 'stop' in each dimension
    out = start[None] + steps * (stop - start)[None]

    return out


def lsd_loss(input_spectrogram: Tensor, ouput_spectrogram: Tensor) -> Tensor:
    """ Log Spectral Density loss
    https://en.wikipedia.org/wiki/Log-spectral_distance
    """
    log_spectral_distance = torch.sum(torch.square(10 * torch.log10((input_spectrogram + 1) / (ouput_spectrogram + 1))))
    return log_spectral_distance


def regression_freq_accuracy(output_dic, target_params_dic, device_arg, synth_cfg: SynthConfig, cfg: Config):
    osc_freq_tensor = torch.tensor(synth_cfg.osc_freq_list, device=device_arg)
    param_dict_to_synth = output_dic
    closest_frequency_index = torch.searchsorted(osc_freq_tensor, param_dict_to_synth['osc1_freq'])
    num_correct_predictions = 0

    frequency_id = list(range(len(synth_cfg.osc_freq_list)))
    frequency_list = synth_cfg.osc_freq_list
    prediction_success = []
    frequency_predictions = []
    frequency_model_output = []

    for i in range(len(param_dict_to_synth['osc1_freq'])):
        predicted_osc = param_dict_to_synth['osc1_freq'][i]
        frequency_model_output.append(predicted_osc.item())
        closest_osc_index_from_below = closest_frequency_index[i] - 1
        closest_osc_index_from_above = closest_frequency_index[i]
        if closest_osc_index_from_below == -1:
            rounded_predicted_freq = synth_cfg.osc_freq_list[0]
        elif closest_osc_index_from_above == len(synth_cfg.osc_freq_list):
            rounded_predicted_freq = synth_cfg.osc_freq_list[len(synth_cfg.osc_freq_list) - 1]
        else:
            below_ratio = predicted_osc / synth_cfg.osc_freq_list[closest_osc_index_from_below.item()]
            above_ratio = synth_cfg.osc_freq_list[closest_osc_index_from_above.item()] / predicted_osc

            if below_ratio < above_ratio:
                rounded_predicted_freq = synth_cfg.osc_freq_list[closest_osc_index_from_below.item()]
            else:
                rounded_predicted_freq = synth_cfg.osc_freq_list[closest_osc_index_from_above.item()]

        frequency_predictions.append(rounded_predicted_freq)
        target_osc = synth_cfg.osc_freq_dic_inv[target_params_dic['classification_params']['osc1_freq'][i].item()]

        if abs(rounded_predicted_freq - target_osc) < 1:
            prediction_success.append(1)
            num_correct_predictions += 1

        else:
            prediction_success.append(0)

    stats = []
    for i in range(len(param_dict_to_synth['osc1_freq'])):
        dict_record = {
            'frequency_id': frequency_id[i],
            'frequency(Hz)': round(frequency_list[i], 3),
            'prediction_success': prediction_success[i],
            'predicted_frequency': round(frequency_predictions[i], 3),
            'frequency_model_output': round(frequency_model_output[i], 3)
        }
        stats.append(dict_record)

    if cfg.print_accuracy_stats:
        fmt = [
            ('Frequency ID', 'frequency_id', 13),
            ('Frequency (Hz)', 'frequency(Hz)', 17),
            ('Prediction Status', 'prediction_success', 20),
            ('Predicted Frequency ', 'predicted_frequency', 20),
            ('Frequency Model Out', 'frequency_model_output', 20),
        ]

        print(TablePrinter(fmt, ul='=')(stats))

    accuracy = num_correct_predictions / len(closest_frequency_index)
    return accuracy, stats


def diff(x, axis=-1):
    """DDSP code:
    https://github.com/magenta/ddsp/blob/8536a366c7834908f418a6721547268e8f2083cc/ddsp/spectral_ops.py#L1"""
    """Take the finite difference of a tensor along an axis.
    Args:
        x: Input tensor of any dimension.
        axis: Axis on which to take the finite difference.
    Returns:
        d: Tensor with size less than x by 1 along the difference dimension.
    Raises:
        ValueError: Axis out of range for tensor.
      """
    shape = list(x.shape)
    ndim = len(shape)
    if axis >= ndim:
        raise ValueError('Invalid axis index: %d for tensor with only %d axes.' % (axis, ndim))

    # begin_back = [0 for _ in range(ndim)]
    # begin_front = [0 for _ in range(ndim)]
    # begin_front[axis] = 1

    shape[axis] -= 1
    # slice_front = x[begin_front[0]:begin_front[0] + shape[0], begin_front[1]:begin_front[1] + shape[1]]
    # slice_back = x[begin_back[0]:begin_back[0] + shape[0], begin_back[1]:begin_back[1] + shape[1]]

    slice_front = torch.narrow(x, axis, 1, shape[axis])
    slice_back = torch.narrow(x, axis, 0, shape[axis])

    d = slice_front - slice_back
    return d


class RMSLELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, actual):
        return torch.sqrt(self.mse(torch.log(pred + 1), torch.log(actual + 1)))


class TablePrinter(object):
    "Print a list of dicts as a table"

    def __init__(self, fmt, sep=' ', ul=None):
        """
        @param fmt: list of tuple(heading, key, width)
                        heading: str, column label
                        key: dictionary key to value to print
                        width: int, column width in chars
        @param sep: string, separation between columns
        @param ul: string, character to underline column label, or None for no underlining
        """
        super(TablePrinter, self).__init__()
        self.fmt = str(sep).join('{lb}{0}:{1}{rb}'.format(key, width, lb='{', rb='}') for heading, key, width in fmt)
        self.head = {key: heading for heading, key, width in fmt}
        self.ul = {key: str(ul) * width for heading, key, width in fmt} if ul else None
        self.width = {key: width for heading, key, width in fmt}

    def row(self, data):
        return self.fmt.format(**{k: str(data.get(k, ''))[:w] for k, w in self.width.items()})

    def __call__(self, datalist):
        _r = self.row
        res = [_r(data) for data in datalist]
        res.insert(0, _r(self.head))
        if self.ul:
            res.insert(1, _r(self.ul))
        return '\n'.join(res)


def kullback_leibler(y_hat, y):
    """Generalized Kullback Leibler divergence.
    :param y_hat: The predicted distribution.
    :type y_hat: torch.Tensor
    :param y: The true distribution.
    :type y: torch.Tensor
    :return: The generalized Kullback Leibler divergence\
             between predicted and true distributions.
    :rtype: torch.Tensor
    """
    return (y * (y.add(1e-6).log() - y_hat.add(1e-6).log()) + (y_hat - y)).sum(dim=-1).mean()


def earth_mover_distance(y_true, y_pred):
    y_pred_cumsum0 = torch.cumsum(y_pred, dim=1)
    y_true_cumsum0 = torch.cumsum(y_true, dim=1)
    square = torch.square(y_true_cumsum0 - y_pred_cumsum0)
    final = torch.mean(square)
    return final


class SpectralLoss:
    """From DDSP code:
    https://github.com/magenta/ddsp/blob/8536a366c7834908f418a6721547268e8f2083cc/ddsp/losses.py#L144"""
    """Multiscale spectrogram loss.
    This loss is the bread-and-butter of comparing two audio signals. It offers
    a range of options to compare spectrograms, many of which are redunant, but
    emphasize different aspects of the signal. By far, the most common comparisons
    are magnitudes (mag_weight) and log magnitudes (logmag_weight).
    """

    def __init__(self,
                 cfg: Config,
                 fft_sizes=(2048, 1024, 512, 256, 128, 64),
                 loss_type='L1',
                 mag_weight=1.0,
                 delta_time_weight=0.0,
                 delta_freq_weight=0.0,
                 cumsum_freq_weight=1.0,
                 logmag_weight=1,
                 loudness_weight=0.0,
                 device='cuda:0',
                 normalize_by_size=False,
                 name='spectral_loss'):
        """Constructor, set loss weights of various components.
    Args:
      fft_sizes: Compare spectrograms at each of this list of fft sizes. Each
        spectrogram has a time-frequency resolution trade-off based on fft size,
        so comparing multiple scales allows multiple resolutions.
      loss_type: One of 'L1', 'L2', or 'COSINE'.
      mag_weight: Weight to compare linear magnitudes of spectrograms. Core
        audio similarity loss. More sensitive to peak magnitudes than log
        magnitudes.
      delta_time_weight: Weight to compare the first finite difference of
        spectrograms in time. Emphasizes changes of magnitude in time, such as
        at transients.
      delta_freq_weight: Weight to compare the first finite difference of
        spectrograms in frequency. Emphasizes changes of magnitude in frequency,
        such as at the boundaries of a stack of harmonics.
      cumsum_freq_weight: Weight to compare the cumulative sum of spectrograms
        across frequency for each slice in time. Similar to a 1-D Wasserstein
        loss, this hopefully provides a non-vanishing gradient to push two
        non-overlapping sinusoids towards eachother.
      logmag_weight: Weight to compare log magnitudes of spectrograms. Core
        audio similarity loss. More sensitive to quiet magnitudes than linear
        magnitudes.
      loudness_weight: Weight to compare the overall perceptual loudness of two
        signals. Very high-level loss signal that is a subset of mag and
        logmag losses.
      name: Name of the module.
    """
        self.normalize_by_size = normalize_by_size
        self.fft_sizes = fft_sizes
        self.loss_type = loss_type
        self.mag_weight = mag_weight
        self.delta_time_weight = delta_time_weight
        self.delta_freq_weight = delta_freq_weight
        self.cumsum_freq_weight = cumsum_freq_weight
        self.logmag_weight = logmag_weight
        self.loudness_weight = loudness_weight
        self.device = device

        self.spectrogram_ops = {}
        for size in self.fft_sizes:
            if cfg.multi_spectral_loss_spec_type == 'BOTH' or cfg.multi_spectral_loss_spec_type == 'SPECTROGRAM':
                spec_transform = torchaudio.transforms.Spectrogram(
                    n_fft=size,
                    power=2.0
                ).to(self.device)
                self.spectrogram_ops[f'{size}_spectrogram'] = spec_transform

            if cfg.multi_spectral_loss_spec_type == 'BOTH' or cfg.multi_spectral_loss_spec_type == 'MEL_SPECTROGRAM':
                mel_spec_transform = torchaudio.transforms.MelSpectrogram(
                    sample_rate=cfg.sample_rate,
                    n_fft=size,
                    hop_length=int(size / 4),
                    n_mels=256,
                    power=2.0,
                    f_min=0,
                    f_max=1300
                ).to(self.device)
                self.spectrogram_ops[f'{size}_mel'] = mel_spec_transform

    def call(self,
             target_audio,
             audio,
             summary_writer: SummaryWriter,
             signal_chain_index: str,
             global_step: int,
             return_spectrogram: bool=False):
        """ execute multi-spectral loss computation between two audio signals

        Args:
          target_audio: target audio signal
          audio:        audio signal
          :param summary_writer:
        """
        loss = 0.0

        if self.loss_type == 'L1':
            criterion = nn.L1Loss(reduction='sum')
        elif self.loss_type == 'L2':
            criterion = nn.MSELoss()
        else:
            criterion = -1
            ValueError("unknown loss type")
        # Compute loss for each fft size.
        loss_dict, weighted_loss_dict = {}, {}
        spectrograms_dict = {}
        for loss_name, loss_op in self.spectrogram_ops.items():
            target_mag = loss_op(target_audio.float())
            value_mag = loss_op(audio.float())

            n_fft = loss_op.n_fft
            c_loss = 0.0

            # Add magnitude loss.
            if self.mag_weight > 0:
                magnitude_loss = criterion(target_mag, value_mag)
                loss_dict[f"{loss_name}_magnitude"] = magnitude_loss
                weighted_loss_dict[f"{loss_name}_magnitude"] = self.mag_weight * magnitude_loss
                c_loss += self.mag_weight * magnitude_loss

            if self.delta_time_weight > 0:
                target = torch.diff(target_mag, n=1, dim=1)
                value = torch.diff(value_mag, n=1, dim=1)
                delta_time_loss = criterion(target, value)
                loss_dict[f"{loss_name}_delta_time"] = delta_time_loss
                weighted_loss_dict[f"{loss_name}_delta_time"] = self.delta_time_weight * delta_time_loss
                c_loss += self.delta_time_weight * delta_time_loss

            if self.delta_freq_weight > 0:
                target = torch.diff(target_mag, n=1, dim=2)
                value = torch.diff(value_mag, n=1, dim=2)
                delta_freq_loss = criterion(target, value)
                loss_dict[f"{loss_name}_delta_freq"] = delta_freq_loss
                weighted_loss_dict[f"{loss_name}_delta_freq"] = self.delta_freq_weight * delta_freq_loss
                c_loss += self.delta_freq_weight * delta_freq_loss

            # TODO(kyriacos) normalize cumulative spectrogram
            if self.cumsum_freq_weight > 0:
                target = torch.cumsum(target_mag, dim=2)
                value = torch.cumsum(value_mag, dim=2)
                emd_loss = criterion(target, value)
                loss_dict[f"{loss_name}_emd"] = emd_loss
                weighted_loss_dict[f"{loss_name}_emd"] = self.cumsum_freq_weight * emd_loss
                c_loss += self.cumsum_freq_weight * emd_loss

            # Add logmagnitude loss, reusing spectrogram.
            if self.logmag_weight > 0:
                target = torch.log(target_mag + 1)
                value = torch.log(value_mag + 1)
                logmag_loss = criterion(target, value)
                loss_dict[f"{loss_name}_logmag"] = logmag_loss
                weighted_loss_dict[f"{loss_name}_logmag"] = self.logmag_weight * logmag_loss
                c_loss += self.logmag_weight * logmag_loss

            if self.normalize_by_size:
                c_loss /= (n_fft / 100.0)

            loss += c_loss

            spectrograms_dict[loss_name] = {'pred': value_mag.detach(), 'target': target_mag.detach()}

        # if self.loudness_weight > 0:
        #     target = spectral_ops.compute_loudness(target_audio, n_fft=2048,
        #                                            use_tf=True)
        #     value = spectral_ops.compute_loudness(audio, n_fft=2048, use_tf=True)
        #     loss += self.loudness_weight * mean_difference(
        #         target, value, self.loss_type, weights=weights)

        for loss_name, loss_val in loss_dict.items():
            summary_writer.add_scalar(f"sub_losses/{signal_chain_index}/{loss_name}", loss_val, global_step=global_step)

        for loss_name, loss_val in weighted_loss_dict.items():
            summary_writer.add_scalar(f"weighted_sub_losses/{signal_chain_index}/{loss_name}", loss_val, global_step=global_step)

        if return_spectrogram:
            return loss, spectrograms_dict

        return loss


def save_model(cur_epoch, model, optimiser_arg, avg_epoch_loss, loss_list, ckpt_path, txt_path, numpy_path):
    # save model checkpoint

    np.save(numpy_path, np.asarray(loss_list))
    torch.save({
        'epoch': cur_epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimiser_arg.state_dict(),
        'loss': avg_epoch_loss
    }, ckpt_path)

    text_file = open(txt_path, 'a')
    text_file.write(f"epoch:{cur_epoch}\tloss: " + str(avg_epoch_loss) + "\n")
    text_file.close()


def print_synth_param_stats(predicted_param_dict, target_param_dict, synth_cfg: SynthConfig, device):
    mapped_target_param_dict = {}
    for idx, value in target_param_dict.items():

        cell_dict = target_param_dict[idx]
        params_dict = cell_dict['parameters']

        mapped_params_dict = {}
        if params_dict != 'None' and not isinstance(params_dict, list):
            for key, val in params_dict.items():
                if key in ['waveform', 'freq_c', 'filter_type']:
                    if key == 'waveform':
                        mapped_params_dict[key] = torch.Tensor([synth_cfg.wave_type_dict[x] for x in val]).to(device)
                    elif key == 'freq_c':
                        val_list = val.tolist()
                        mapped_params_dict[key] = torch.Tensor(
                            [synth_cfg.osc_freq_dic[round(x, 4)] for x in val_list]).to(device)
                    elif key == 'filter_type':
                        mapped_params_dict[key] = synth_cfg.filter_type_dict[val]
                else:
                    mapped_params_dict[key] = params_dict[key].clone().detach().requires_grad_(True).to(device)

        mapped_cell_dict = {'operation': cell_dict['operation'],
                            'parameters': mapped_params_dict}
        mapped_target_param_dict[idx] = mapped_cell_dict

    print("Synth parameters statistics\n---------------------------")
    for index, operation_dict in predicted_param_dict.items():
        operation = operation_dict['operation']
        result = all(elem == operation for elem in target_param_dict[index]['operation'])
        if not result:
            AssertionError("Unpredictable operation prediction behavior")

        if operation == 'osc':
            predicted_carrier_amp = operation_dict['params']['amp'].squeeze()
            predicted_carrier_freq = operation_dict['params']['freq'].squeeze()
            predicted_waveform = operation_dict['params']['waveform'].argmax(dim=1)

            target_amp = mapped_target_param_dict[index]['parameters']['amp']
            target_freq = mapped_target_param_dict[index]['parameters']['freq']
            target_waveform = mapped_target_param_dict[index]['parameters']['waveform']

            amp_dist = torch.sqrt(torch.sum(torch.square(predicted_carrier_amp - target_amp)))
            freq_dist = torch.sqrt(torch.sum(torch.square(predicted_carrier_freq - target_freq)))
            waveform_accuracy = \
                torch.sum(torch.eq(predicted_waveform, target_waveform)) * 100 / target_waveform.shape[0]

            print(f"{operation} at index {index} param stats")
            print(f"\tamp l2 dist: {amp_dist}")
            print(f"\tfreq l2 dist: {freq_dist}")
            print(f"\twaveform accuracy: {waveform_accuracy}%\n")

        elif operation == 'fm':
            predicted_carrier_amp = operation_dict['params']['amp_c'].squeeze()
            predicted_carrier_freq = operation_dict['params']['freq_c'].squeeze()
            predicted_carrier_waveform = operation_dict['params']['waveform'].argmax(dim=1)
            predicted_mod_index = operation_dict['params']['mod_index'].squeeze()

            target_carrier_amp = mapped_target_param_dict[index]['parameters']['amp_c']
            target_carrier_freq = mapped_target_param_dict[index]['parameters']['freq_c']
            target_carrier_waveform = mapped_target_param_dict[index]['parameters']['waveform']
            target_mod_index = mapped_target_param_dict[index]['parameters']['mod_index']

            carrier_amp_dist = torch.sqrt(torch.sum(torch.square(predicted_carrier_amp - target_carrier_amp)))
            carrier_freq_dist = torch.sqrt(torch.sum(torch.square(predicted_carrier_freq - target_carrier_freq)))
            carrier_waveform_accuracy = \
                torch.sum(torch.eq(predicted_carrier_waveform, target_carrier_waveform)) \
                * 100 / target_waveform.shape[0]
            mod_index_dist = torch.sqrt(torch.sum(torch.square(predicted_mod_index - target_mod_index)))

            print(f"{operation} at index {index} param stats")
            print(f"\tcarrier amp l2 dist: {carrier_amp_dist}")
            print(f"\tcarrier freq l2 dist: {carrier_freq_dist}")
            print(f"\tcarrier waveform accuracy: {carrier_waveform_accuracy}%")
            print(f"\tmod_index l2 dist: {mod_index_dist}\n")

        elif operation == 'filter':
            predicted_filter_freq = operation_dict['params']['filter_freq'].squeeze()
            predicted_filter_type = operation_dict['params']['filter_type'].argmax(dim=1)

            target_filter_freq = target_param_dict[index]['parameters']['filter_freq']
            target_filter_type = target_param_dict[index]['parameters']['filter_type']

            filter_freq_dist = torch.sqrt(torch.sum(torch.square(predicted_filter_freq, target_filter_freq)))
            filter_type_accuracy = \
                torch.sum(torch.eq(predicted_filter_type, target_filter_type)) * 100 / target_filter_type.shape[0]

            print(f"{operation} at index {index} param stats")
            print(f"\tfilter_freq l2 dist: {filter_freq_dist}")
            print(f"\tfilter_type accuracy: {filter_type_accuracy}%\n")

        elif operation == 'env_adsr':
            predicted_attack_t = operation_dict['params']['attack_t'].squeeze()
            predicted_decay_t = operation_dict['params']['decay_t'].squeeze()
            predicted_sustain_t = operation_dict['params']['sustain_t'].squeeze()
            predicted_sustain_level = operation_dict['params']['sustain_level'].squeeze()
            predicted_release_t = operation_dict['params']['release_t'].squeeze()

            target_attack_t = target_param_dict[index]['parameters']['attack_t']
            target_decay_t = target_param_dict[index]['parameters']['decay_t']
            target_sustain_t = target_param_dict[index]['parameters']['sustain_t']
            target_sustain_level = target_param_dict[index]['parameters']['sustain_level']
            target_release_t = target_param_dict[index]['parameters']['release_t']

            attack_t_dist = torch.sqrt(torch.sum(torch.square(predicted_attack_t - target_attack_t)))
            decay_t_dist = torch.sqrt(torch.sum(torch.square(predicted_decay_t - target_decay_t)))
            sustain_t_dist = torch.sqrt(torch.sum(torch.square(predicted_sustain_t - target_sustain_t)))
            sustain_level_dist = torch.sqrt(torch.sum(torch.square(predicted_sustain_level - target_sustain_level)))
            release_t_dist = torch.sqrt(torch.sum(torch.square(predicted_release_t - target_release_t)))

            print(f"{operation} at index {index} param stats")
            print(f"\tattack_t l2 dist: {attack_t_dist}")
            print(f"\tdecay_t l2 dist: {decay_t_dist}")
            print(f"\tsustain_t l2 dist: {sustain_t_dist}")
            print(f"\tsustain_level l2 dist: {sustain_level_dist}")
            print(f"\trelease_t l2 dist: {release_t_dist}\n")
