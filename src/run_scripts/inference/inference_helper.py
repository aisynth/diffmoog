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

        num_sounds = len(signal_index)

        # -----------Run Model-----------------
        raw_target_signal = raw_target_signal.to(device)
        target_signal_spectrogram = preprocess_fn(raw_target_signal)
        model_output = eval_fn(target_signal_spectrogram)
        processed_output = post_process_fn(model_output) if post_process_fn is not None else model_output
        predicted_param_dict = helper.clamp_adsr_params(processed_output, synth_cfg, cfg)

        synth.update_cells_from_dict(predicted_param_dict)
        pred_final_signal, pred_signals_through_chain = \
            synth.generate_signal(num_sounds_=num_sounds)

        predicted_signal_spectrograms = preprocess_fn(pred_final_signal)
        results['lsd_value'] += np.sum(lsd(target_signal_spectrogram, predicted_signal_spectrograms))
        results['pearson_stft'] += np.sum(pearsonr_dist(target_signal_spectrogram,
                                                        predicted_signal_spectrograms,
                                                        input_type='spec'))
        results['pearson_fft'] += np.sum(pearsonr_dist(raw_target_signal, pred_final_signal, input_type='audio'))
        results['mean_average_error'] += np.sum(mae(target_signal_spectrogram, predicted_signal_spectrograms))
        results['mfcc_mae'] += np.sum(mfcc_distance(raw_target_signal, pred_final_signal, sample_rate=cfg.sample_rate))
        results['spectral_convergence_value'] += np.sum(spectral_convergence(target_signal_spectrogram,
                                                                             predicted_signal_spectrograms))

        # -----------Discretize output-----------------
        denormalized_discrete_output_params = {cell_idx: {'operation': cell_params['operation'],
                                                          'parameters': discretize_params(cell_params['operation'],
                                                                                          cell_params['parameters'],
                                                                                          synth_cfg)}
                                               for cell_idx, cell_params in processed_output.items()}

        discrete_target_params = {cell_idx: {'operation': cell_params['operation'],
                                             'parameters': discretize_params(cell_params['operation'][0],
                                                                             cell_params['parameters'], synth_cfg)}
                                  for cell_idx, cell_params in target_param_dict.items() if
                                  cell_params['operation'][0] != 'None'}

        # -----------Compare results-----------------
        correct_preds = compare_params(discrete_target_params, denormalized_discrete_output_params)
        for cell_idx, cell_data in correct_preds.items():
            for param_name, correct_preds in cell_data.items():
                results[f'{cell_idx}_{param_name}'] += correct_preds

        n_samples += len(raw_target_signal)
        step += 1

        if step % 100 == 0:
            print(f'processed {step} batches')

    for k, v in results.items():
        results[k] = v / n_samples

    return results


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

    diff = np.log10(spec1) - np.log10(spec2)
    lsd_val = np.linalg.norm(diff, ord='fro')

    return lsd_val


def pearsonr_dist(x1, x2, input_type='spec'):

    if input_type == 'spec':
        x1 = x1.flatten()
        x2 = x2.flatten()
    elif input_type == 'audio':
        x1 = np.abs(fft(x1))
        x2 = np.abs(fft(x2))
    else:
        AttributeError("Unknown input_type")

    pearson_r, _ = pearsonr(x1, x2)

    return pearson_r


def mae(spec1, spec2):

    abs_diff = np.abs(np.log10(spec1) - np.log10(spec2))
    mae_val = abs_diff.mean()

    return mae_val


def mfcc_distance(sound1, sound2, sample_rate):

    mfcc1 = mfcc(sound1, sr=sample_rate, n_mfcc=40)
    mfcc2 = mfcc(sound2, sr=sample_rate, n_mfcc=40)

    abs_diff = np.abs(mfcc1 - mfcc2)
    mfcc_dist = abs_diff.mean()

    return mfcc_dist


def spectral_convergence(target_spec, pred_spec):

    abs_diff = np.abs(target_spec) - np.abs(pred_spec)
    nom = np.linalg.norm(abs_diff, ord='fro')

    denom = np.linalg.norm(np.abs(target_spec), ord='fro')

    sc_val = nom / denom

    return sc_val



