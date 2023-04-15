import torch
from torchaudio.transforms import MFCC

from torchmetrics.functional import pearson_corrcoef


amp = lambda x: x[..., 0]**2 + x[..., 1]**2


def spectrogram(audio, size=2048, hop_length=1024, power=2, center=False, window=None):
    power_spec = amp(torch.view_as_real(torch.stft(audio, size, window=window, hop_length=hop_length, center=center,
                                                   return_complex=True)))
    if power == 2:
        spec = power_spec
    elif power == 1:
        spec = power_spec.sqrt()
    return spec


def paper_lsd(orig_audio, resyn_audio):
    window = torch.hann_window(1024).to(orig_audio.device)
    orig_power_s = spectrogram(orig_audio, 1024, 256, window=window).detach()
    resyn_power_s = spectrogram(resyn_audio, 1024, 256, window=window).detach()
    square_log_diff = ((10 * (torch.log10(resyn_power_s+1e-5)-torch.log10(orig_power_s+1e-5)))**2)
    lsd_val = torch.sqrt(square_log_diff.sum(dim=(1, 2))) / orig_power_s.shape[-1]
    lsd_val = lsd_val.mean()
    return lsd_val


def lsd(spec1, spec2, reduction=None):
    """ spec1, spec2 one channel - positive values"""

    assert spec1.ndim == 3 and spec2.ndim == 3, "Input must be a batch of 2d spectrograms"

    batch_diff = torch.log10(spec1 + 1e-5) - torch.log10(spec2 + 1e-5)
    lsd_val = torch.linalg.norm(10 * batch_diff, ord='fro', dim=(1, 2)) / batch_diff.shape[-1]
    
    if reduction is not None:
        lsd_val = reduction(lsd_val)

    return lsd_val


def pearsonr_dist(x1, x2, input_type='spec', reduction=None):

    if input_type == 'spec':
        assert x1.ndim == 3 and x2.ndim == 3, "Input must be a batch of 2d spectrograms"
        x1 = [x.flatten() for x in x1]
        x2 = [x.flatten() for x in x2]
    elif input_type == 'audio':
        assert x1.ndim == 2 and x2.ndim == 2, "Input must be a batch of 1d wavelet"
        x1 = torch.abs(torch.fft.fft(x1))
        x2 = torch.abs(torch.fft.fft(x2))
    else:
        AttributeError("Unknown input_type")

    pearson_r = torch.tensor([pearson_corrcoef(c_x1, c_x2) for c_x1, c_x2 in zip(x1, x2)])
    pearson_r[torch.isnan(pearson_r)] = 0
    
    if reduction is not None:
        pearson_r = reduction(pearson_r)
    
    return pearson_r


def mae(spec1, spec2, reduction=None):

    assert spec1.ndim == 3 and spec2.ndim == 3, "Input must be a batch of 2d spectrograms"

    abs_diff = torch.abs(torch.log10(spec1 + 1e-5) - torch.log10(spec2 + 1e-5))
    mae_val = abs_diff.mean(dim=(1, 2))
    
    if reduction is not None:
        mae_val = reduction(mae_val)

    return mae_val


def mfcc_distance(sound_batch1, sound_batch2, sample_rate, device, reduction=None):

    mfcc = MFCC(sample_rate).to(device)

    assert sound_batch1.ndim == 2 and sound_batch2.ndim == 2, "Input must be a batch of 1d wavelet"

    mfcc1 = mfcc(sound_batch1)
    mfcc2 = mfcc(sound_batch2)

    abs_diff = torch.abs(mfcc1 - mfcc2)
    mfcc_dist = abs_diff.mean(dim=(1, 2))

    if reduction is not None:
        mfcc_dist = reduction(mfcc_dist)

    return mfcc_dist


def spectral_convergence(target_spec_batch, pred_spec_batch, reduction=None):

    assert target_spec_batch.ndim == 3 and pred_spec_batch.ndim == 3, "Input must be a batch of 2d spectrograms"

    abs_diff = torch.abs(target_spec_batch) - torch.abs(pred_spec_batch)

    nom = torch.linalg.norm(abs_diff, ord='fro', dim=(1, 2))
    denom = torch.linalg.norm(torch.abs(target_spec_batch), ord='fro', dim=(1, 2))

    sc_val = nom / denom
    sc_val[torch.isinf(sc_val)] = 10

    if reduction is not None:
        sc_val = reduction(sc_val)

    return sc_val


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
