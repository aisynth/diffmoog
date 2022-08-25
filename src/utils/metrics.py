import numpy as np
from librosa.feature import mfcc

from scipy.stats import pearsonr
from scipy.fft import fft


def lsd(spec1, spec2):
    """ spec1, spec2 one channel - positive values"""

    assert spec1.ndim == 3 and spec2.ndim == 3, "Input must be a batch of 2d spectrograms"

    batch_diff = np.log10(spec1 + 1e-5) - np.log10(spec2 + 1e-5)
    lsd_val = [np.linalg.norm(10 * x, ord='fro') / x.shape[-1] for x in batch_diff]

    return lsd_val


def pearsonr_dist(x1, x2, input_type='spec'):

    if input_type == 'spec':
        assert x1.ndim == 3 and x2.ndim == 3, "Input must be a batch of 2d spectrograms"
        x1 = [x.flatten() for x in x1]
        x2 = [x.flatten() for x in x2]
    elif input_type == 'audio':
        assert x1.ndim == 2 and x2.ndim == 2, "Input must be a batch of 1d wavelet"
        x1 = np.abs(fft(x1))
        x2 = np.abs(fft(x2))
    else:
        AttributeError("Unknown input_type")

    pearson_r = [pearsonr(c_x1, c_x2)[0] for c_x1, c_x2 in zip(x1, x2)]

    return pearson_r


def mae(spec1, spec2):

    assert spec1.ndim == 3 and spec2.ndim == 3, "Input must be a batch of 2d spectrograms"

    abs_diff = np.abs(np.log10(spec1 + 1e-5) - np.log10(spec2 + 1e-5))
    mae_val = [sample_diff.mean() for sample_diff in abs_diff]

    return mae_val


def mfcc_distance(sound_batch1, sound_batch2, sample_rate):

    assert sound_batch1.ndim == 2 and sound_batch2.ndim == 2, "Input must be a batch of 1d wavelet"

    res = []
    for sound1, sound2 in zip(sound_batch1, sound_batch2):
        mfcc1 = mfcc(y=sound1, sr=sample_rate, n_mfcc=40)
        mfcc2 = mfcc(y=sound2, sr=sample_rate, n_mfcc=40)

        abs_diff = np.abs(mfcc1 - mfcc2)
        mfcc_dist = abs_diff.mean()
        res.append(mfcc_dist)

    return res


def spectral_convergence(target_spec_batch, pred_spec_batch):

    assert target_spec_batch.ndim == 3 and pred_spec_batch.ndim == 3, "Input must be a batch of 2d spectrograms"

    res = []
    for target_spec, pred_spec in zip(target_spec_batch, pred_spec_batch):

        abs_diff = np.abs(target_spec) - np.abs(pred_spec)
        nom = np.linalg.norm(abs_diff, ord='fro')

        denom = np.linalg.norm(np.abs(target_spec), ord='fro')

        sc_val = nom / denom

        res.append(sc_val)

    return res


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
    y_pred_cumsum0 = np.cumsum(y_pred, dim=1)
    y_true_cumsum0 = np.cumsum(y_true, dim=1)
    square = np.square(y_true_cumsum0 - y_pred_cumsum0)
    final = np.mean(square)
    return final
