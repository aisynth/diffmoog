import torch
import torchaudio
import synth
import matplotlib.pyplot as plt
import librosa
from config import TWO_PI, DEBUG_MODE, SAMPLE_RATE


def get_device():
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    if DEBUG_MODE:
        print(f"Using device {device}")
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


# class TwoWayDict(dict):
#     def __len__(self):
#         return dict.__len__(self) / 2
#
#     def __setitem__(self, key, value):
#         dict.__setitem__(self, key, value)
#         dict.__setitem__(self, value, key)


mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=1024,
    hop_length=512,
    n_mels=64,
    power=2.0
).to(get_device())

amplitude_to_db_transform = torchaudio.transforms.AmplitudeToDB().to(get_device())


def map_classification_params_to_ints(classification_params_dic: dict):
    """ map classification params to ints, to input them for a neural network """

    for key, val in classification_params_dic.items():
        if key == "osc1_freq" or key == "osc2_freq":
            classification_params_dic[key] = torch.tensor([synth.OSC_FREQ_DIC[round(x.item(), 4)] for x in val])
        if isinstance(val, list):
            if "wave" in key:
                classification_params_dic[key] = torch.tensor([synth.WAVE_TYPE_DIC[x] for x in val])
            elif "filter_type" == key:
                classification_params_dic[key] = torch.tensor([synth.FILTER_TYPE_DIC[x] for x in val])


def map_classification_params_from_ints(classification_params_dic: dict):
    """ map classification params from ints (inverse operation of map_classification_params_to_ints),
     to input them for a neural network """

    for key, val in classification_params_dic.items():
        if key == "osc1_freq" or key == "osc2_freq":
            # classification_params_dic[key] = synth.OSC_FREQ_DIC[round(val, 4)]
            classification_params_dic[key] = torch.tensor([synth.OSC_FREQ_DIC_INV[x.item()] for x in val])
        if "wave" in key:
            classification_params_dic[key] = [synth.WAVE_TYPE_DIC_INV[x.item()] for x in val]
        elif "filter_type" == key:
            classification_params_dic[key] = [synth.FILTER_TYPE_DIC_INV[x.item()] for x in val]


def clamp_regression_params(parameters_dict: dict):
    """ clamp regression parameters to values that can be interpreted by the synth module"""
    '''
    ['osc1_amp', 'osc1_mod_index', 'lfo1_freq', 'lfo1_phase',
     'osc2_amp', 'osc2_mod_index', 'lfo2_freq', 'lfo2_phase',
     'filter_freq', 'attack_t', 'decay_t', 'sustain_t', 'release_t', 'sustain_level']'''

    parameters_dict['osc1_amp'] = torch.clamp(parameters_dict['osc1_amp'], min=0, max=synth.MAX_AMP)
    parameters_dict['osc1_mod_index'] = torch.clamp(parameters_dict['osc1_mod_index'], min=0, max=synth.MAX_MOD_INDEX)
    parameters_dict['lfo1_freq'] = torch.clamp(parameters_dict['lfo1_freq'], min=0, max=synth.MAX_LFO_FREQ)
    parameters_dict['lfo1_phase'] = parameters_dict['lfo1_freq'] % TWO_PI
    parameters_dict['osc2_amp'] = torch.clamp(parameters_dict['osc2_amp'], min=0, max=synth.MAX_AMP)
    parameters_dict['osc2_mod_index'] = torch.clamp(parameters_dict['osc2_mod_index'], min=0, max=synth.MAX_MOD_INDEX)
    parameters_dict['lfo2_freq'] = torch.clamp(parameters_dict['lfo2_freq'], min=0, max=synth.MAX_LFO_FREQ)
    parameters_dict['lfo2_phase'] = parameters_dict['lfo2_freq'] % TWO_PI

    parameters_dict['filter_freq'] = torch.clamp(parameters_dict['filter_freq'],
                                                 min=synth.MIN_FILTER_FREQ,
                                                 max=synth.MAX_FILTER_FREQ)

    parameters_dict['attack_t'] = torch.clamp(parameters_dict['attack_t'], min=0, max=synth.SIGNAL_DURATION_SEC)
    parameters_dict['decay_t'] = torch.clamp(parameters_dict['decay_t'], min=0, max=synth.SIGNAL_DURATION_SEC)
    parameters_dict['sustain_t'] = torch.clamp(parameters_dict['release_t'], min=0, max=synth.SIGNAL_DURATION_SEC)
    parameters_dict['release_t'] = torch.clamp(parameters_dict['release_t'], min=0, max=synth.SIGNAL_DURATION_SEC)

    # clamp aggregated adsr parameters that are longer than signal duration
    adsr_length_in_sec = parameters_dict['attack_t'] \
                         + parameters_dict['decay_t'] \
                         + parameters_dict['sustain_t'] \
                         + parameters_dict['release_t']

    adsr_clamp_indices = torch.nonzero(adsr_length_in_sec >= synth.SIGNAL_DURATION_SEC, as_tuple=True)[0]

    for i in adsr_clamp_indices.tolist():
        # add small number to normalization to prevent numerical issue where the sum exceeds 1
        normalization_value = adsr_length_in_sec[i] + 1e-5
        parameters_dict['attack_t'][i] /= normalization_value
        parameters_dict['decay_t'][i] /= normalization_value
        parameters_dict['sustain_t'][i] /= normalization_value
        parameters_dict['release_t'][i] /= normalization_value

    parameters_dict['sustain_level'] = torch.clamp(parameters_dict['sustain_level'], min=0, max=synth.MAX_AMP)


class Normalizer:
    """ normalize/de-normalise regression parameters"""
    '''
    ['osc1_amp', 'osc1_mod_index', 'lfo1_freq', 'lfo1_phase',
     'osc2_amp', 'osc2_mod_index', 'lfo2_freq', 'lfo2_phase',
     'filter_freq', 'attack_t', 'decay_t', 'sustain_t', 'release_t', 'sustain_level']'''

    def __init__(self):
        self.mod_index_normalizer = MinMaxNormaliser(target_min_val=0,
                                                     target_max_val=1,
                                                     original_min_val=0,
                                                     original_max_val=synth.MAX_MOD_INDEX)

        self.lfo_freq_normalizer = MinMaxNormaliser(target_min_val=0,
                                                    target_max_val=1,
                                                    original_min_val=0,
                                                    original_max_val=synth.MAX_LFO_FREQ)

        self.lfo_phase_normalizer = MinMaxNormaliser(target_min_val=0,
                                                     target_max_val=1,
                                                     original_min_val=0,
                                                     original_max_val=TWO_PI)

        self.adsr_normalizer = MinMaxNormaliser(target_min_val=0,
                                                target_max_val=1,
                                                original_min_val=0,
                                                original_max_val=synth.SIGNAL_DURATION_SEC)

        self.filter_freq_normalizer = LogNormaliser()

    def normalize(self, parameters_dict: dict):

        parameters_dict['osc1_mod_index'] = self.mod_index_normalizer.normalise(parameters_dict['osc1_mod_index'])
        parameters_dict['lfo1_freq'] = self.lfo_freq_normalizer.normalise(parameters_dict['lfo1_freq'])
        parameters_dict['lfo1_phase'] = parameters_dict['lfo1_phase'] % TWO_PI
        parameters_dict['lfo1_phase'] = self.lfo_phase_normalizer.normalise(parameters_dict['lfo1_phase'])
        parameters_dict['osc2_mod_index'] = self.mod_index_normalizer.normalise(parameters_dict['osc2_mod_index'])
        parameters_dict['lfo2_freq'] = self.lfo_freq_normalizer.normalise(parameters_dict['lfo2_freq'])
        parameters_dict['lfo2_phase'] = parameters_dict['lfo2_phase'] % TWO_PI
        parameters_dict['lfo2_phase'] = self.lfo_phase_normalizer.normalise(parameters_dict['lfo2_phase'])
        parameters_dict['lfo2_phase'] = self.lfo_phase_normalizer.normalise(parameters_dict['lfo2_phase'])
        parameters_dict['filter_freq'] = self.filter_freq_normalizer.normalise(parameters_dict['filter_freq'])
        parameters_dict['attack_t'] = self.adsr_normalizer.normalise(parameters_dict['attack_t'])
        parameters_dict['decay_t'] = self.adsr_normalizer.normalise(parameters_dict['decay_t'])
        parameters_dict['sustain_t'] = self.adsr_normalizer.normalise(parameters_dict['sustain_t'])
        parameters_dict['release_t'] = self.adsr_normalizer.normalise(parameters_dict['release_t'])

    def denormalize(self, parameters_dict: dict):

        parameters_dict['osc1_mod_index'] = self.mod_index_normalizer.denormalise(parameters_dict['osc1_mod_index'])
        parameters_dict['lfo1_freq'] = self.lfo_freq_normalizer.denormalise(parameters_dict['lfo1_freq'])
        parameters_dict['lfo1_phase'] = parameters_dict['lfo1_phase'] % TWO_PI
        parameters_dict['lfo1_phase'] = self.lfo_phase_normalizer.denormalise(parameters_dict['lfo1_phase'])
        parameters_dict['osc2_mod_index'] = self.mod_index_normalizer.denormalise(parameters_dict['osc2_mod_index'])
        parameters_dict['lfo2_freq'] = self.lfo_freq_normalizer.denormalise(parameters_dict['lfo2_freq'])
        parameters_dict['lfo2_phase'] = parameters_dict['lfo2_phase'] % TWO_PI
        parameters_dict['lfo2_phase'] = self.lfo_phase_normalizer.denormalise(parameters_dict['lfo2_phase'])
        parameters_dict['lfo2_phase'] = self.lfo_phase_normalizer.denormalise(parameters_dict['lfo2_phase'])
        parameters_dict['filter_freq'] = self.filter_freq_normalizer.denormalise(parameters_dict['filter_freq'])
        parameters_dict['attack_t'] = self.adsr_normalizer.denormalise(parameters_dict['attack_t'])
        parameters_dict['decay_t'] = self.adsr_normalizer.denormalise(parameters_dict['decay_t'])
        parameters_dict['sustain_t'] = self.adsr_normalizer.denormalise(parameters_dict['sustain_t'])
        parameters_dict['release_t'] = self.adsr_normalizer.denormalise(parameters_dict['release_t'])


def plot_spectrogram(spec, title=None, ylabel='freq_bin', aspect='auto', xmax=None):
    """ note that the spectrogram shall be given from STFT computation.
        before display, an amplitude to db conversion is done"""
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or 'Spectrogram')
    axs.set_ylabel(ylabel)
    axs.set_xlabel('frame')
    im = axs.imshow(spec, origin='lower', aspect=aspect)
    if xmax:
        axs.set_xlim((0, xmax))
    fig.colorbar(im, ax=axs)
    plt.show(block=False)


class MinMaxNormaliser:
    """MinMaxNormaliser applies min max normalisation to a tensor"""

    def __init__(self, target_min_val, target_max_val, original_min_val, original_max_val):
        self.target_min_val = target_min_val
        self.target_max_val = target_max_val
        self.original_max_val = original_min_val
        self.original_min_val = original_max_val

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
