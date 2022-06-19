from collections import defaultdict

import torch
import numpy as np

from torch.utils.data.dataloader import DataLoader

from librosa.feature import mfcc

from scipy.stats import pearsonr
from scipy.fft import fft

from config import SynthConfig, Config
from model import helper
from synth.synth_architecture import SynthModular


def inference_loop(cfg: Config, synth_cfg: SynthConfig, synth: SynthModular, test_dataloader: DataLoader, preprocess_fn: callable, eval_fn: callable,
                   post_process_fn: callable, device: str = 'cuda:0'):

    step, n_samples = 0, 0
    results = defaultdict(float)
    for raw_target_signal, target_param_dict, signal_index in test_dataloader:

        input_batch = (raw_target_signal, target_param_dict, signal_index)

        _, metrics = process_batch_inference(input_batch, preprocess_fn, eval_fn, post_process_fn, synth, device, cfg)
        for k, val in metrics.items():
            results[k] += val

        # # Add Accuracy measures for inference
        # # -----------Discretize output-----------------
        # denormalized_discrete_output_params = {cell_idx: {'operation': cell_params['operation'],
        #                                                   'parameters': discretize_params(cell_params['operation'],
        #                                                                                   cell_params['parameters'],
        #                                                                                   synth_cfg)}
        #                                        for cell_idx, cell_params in processed_output.items()}
        #
        # discrete_target_params = {cell_idx: {'operation': cell_params['operation'],
        #                                      'parameters': discretize_params(cell_params['operation'][0],
        #                                                                      cell_params['parameters'], synth_cfg)}
        #                           for cell_idx, cell_params in target_param_dict.items() if
        #                           cell_params['operation'][0] != 'None'}
        #
        # # -----------Compare results-----------------
        # correct_preds = compare_params(discrete_target_params, denormalized_discrete_output_params)
        # for cell_idx, cell_data in correct_preds.items():
        #     for param_name, correct_preds in cell_data.items():
        #         results[f'{cell_idx}_{param_name}'] += correct_preds

        n_samples += len(raw_target_signal)
        step += 1

        if step % 100 == 0:
            print(f'processed {step} batches')

    for k, v in results.items():
        results[k] = v / n_samples

    return results


def process_batch_inference(input_batch, preprocess_fn, eval_fn, post_process_fn, synth, device, cfg):

    results, metrics = {}, defaultdict(float)

    raw_target_signal, target_param_dict, signal_index = input_batch
    num_sounds = len(signal_index)

    # -----------Run Model-----------------
    raw_target_signal = raw_target_signal.to(device)
    target_signal_spectrogram = preprocess_fn(raw_target_signal)
    model_output = eval_fn(target_signal_spectrogram)
    processed_output = post_process_fn(model_output) if post_process_fn is not None else model_output

    synth.update_cells_from_dict(processed_output)
    pred_final_signal, pred_signals_through_chain = \
        synth.generate_signal(num_sounds_=num_sounds)

    predicted_signal_spectrograms = preprocess_fn(pred_final_signal).squeeze().detach().cpu().numpy()
    target_signal_spectrogram = target_signal_spectrogram.squeeze().detach().cpu().numpy()

    raw_target_signal = raw_target_signal.squeeze().detach().cpu().numpy()
    pred_final_signal = pred_final_signal.squeeze().detach().cpu().numpy()

    metrics['lsd_value'] += np.sum(lsd(target_signal_spectrogram, predicted_signal_spectrograms))
    metrics['pearson_stft'] += np.sum(pearsonr_dist(target_signal_spectrogram,
                                                    predicted_signal_spectrograms,
                                                    input_type='spec'))
    metrics['pearson_fft'] += np.sum(pearsonr_dist(raw_target_signal, pred_final_signal, input_type='audio'))
    metrics['mean_average_error'] += np.sum(mae(target_signal_spectrogram, predicted_signal_spectrograms))
    metrics['mfcc_mae'] += np.sum(mfcc_distance(raw_target_signal, pred_final_signal, sample_rate=cfg.sample_rate))
    metrics['spectral_convergence_value'] += np.sum(spectral_convergence(target_signal_spectrogram,
                                                                         predicted_signal_spectrograms))

    results['target_audio'] = raw_target_signal
    results['predicted_audio'] = pred_final_signal

    results['target_spectrograms'] = target_signal_spectrogram
    results['predicted_spectrograms'] = predicted_signal_spectrograms

    return results, metrics


def discretize_params(operation: str, input_params: dict, synth_cfg):

    params_preset = synth_cfg.all_params_presets.get(operation, {})

    res = {}
    for param_name, param_values in input_params.items():

        if isinstance(param_values, torch.Tensor):
            param_values = param_values.detach().cpu().numpy()
        else:
            param_values = np.asarray(param_values)

        if param_name in ['waveform', 'filter_type']:

            if isinstance(param_values[0], str):
                res[param_name] = param_values
                continue

            idx = np.argmax(param_values, axis=1)
            if param_name == 'waveform':
                res[param_name] = [synth_cfg.wave_type_dic_inv[i] for i in idx]
            else:
                res[param_name] = [synth_cfg.filter_type_dic_inv[i] for i in idx]
            continue

        possible_values = params_preset.get(param_name, None)

        if possible_values is None:
            res[param_name] = param_values
            continue

        idx = np.searchsorted(possible_values, param_values, side="left")
        idx[idx == len(possible_values)] = len(possible_values) - 1
        idx[idx == 0] = 1

        if operation == 'fm' and param_name == 'freq_c':
            below_distance = (param_values / possible_values[idx - 1])
            above_distance = (possible_values[idx] / param_values)
        else:
            below_distance = np.abs(param_values - possible_values[idx - 1])
            above_distance = np.abs(param_values - possible_values[idx])

        idx = idx - (below_distance < above_distance)
        res[param_name] = possible_values[idx]

    return res


def compare_params(target_params, predicted_params):
    res = defaultdict(dict)
    for cell_idx, target_cell_data in target_params.items():

        if target_cell_data['operation'][0] == 'None':
            continue

        target_cell_params = target_cell_data['parameters']
        predicted_cell_params = predicted_params[cell_idx]['parameters']

        for param_name, target_param_values in target_cell_params.items():
            pred_param_values = np.asarray(predicted_cell_params[param_name]).squeeze()
            target_param_values = np.asarray(target_param_values).squeeze()

            assert len(target_param_values) == len(pred_param_values)

            correct_preds = np.sum(target_param_values == pred_param_values)

            res[cell_idx][param_name] = correct_preds

    return res


def lsd(spec1, spec2):
    """ spec1, spec2 one channel - positive values"""

    assert spec1.ndim == 3 and spec2.ndim == 3, "Input must be a batch of 2d spectrograms"

    batch_diff = np.log10(spec1) - np.log10(spec2)
    lsd_val = [np.linalg.norm(x, ord='fro') for x in batch_diff]

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

    abs_diff = np.abs(np.log10(spec1) - np.log10(spec2))
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



