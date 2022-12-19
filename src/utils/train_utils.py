from typing import Sequence, Union, Callable

import numpy as np
import torch
import torchaudio

from pytorch_lightning.loggers import TensorBoardLogger
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import confusion_matrix
from scipy.special import softmax

from synth.synth_constants import synth_constants
from synth.synth_presets import synth_presets_dict
from synth.synth_constants import SynthConstants
from model.loss.spectral_loss_presets import loss_presets


def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent


def log_gradients_in_model(model, writer: SummaryWriter, step):
    for tag, value in model.named_parameters():
        if value.grad is not None:
            grad_val = value.grad.cpu()
            writer.add_histogram(tag + "/grad", grad_val, step)
            if np.linalg.norm(grad_val) < 1e-5 and 'bias' not in tag:
                print(f"Op {tag} gradient approaching 0")


def parse_synth_params(original_params: dict, predicted_params: dict, sample_idx: int) -> (dict, dict):

    pred_res, orig_res = {}, {}

    original_params = to_numpy_recursive(original_params)
    predicted_params = to_numpy_recursive(predicted_params)

    for k, d in predicted_params.items():
        op = d['operation'] if isinstance(d['operation'], str) else d['operation'][0]
        while op in pred_res:
            op = op + '_|'
        pred_res[op] = {}
        orig_res[op] = {}
        for param, vals in d['parameters'].items():

            if param in ['active', 'fm_active']:
                if len(vals.shape) < 2:
                    vals = softmax(vals)
                else:
                    vals = softmax(vals, axis=1)

            if len(vals.shape) == 0:
                pred_res[op][param] = _np_to_str(vals.squeeze(), precision=2)
            else:
                pred_res[op][param] = _np_to_str(vals[sample_idx].squeeze(), precision=2)

            if param in ['waveform', 'filter_type']:
                orig_res[op][param] = original_params[k]['parameters'][param][sample_idx]
            else:
                if len(original_params[k]['parameters'][param].shape) == 0:
                    orig_res[op][param] = _np_to_str(original_params[k]['parameters'][param], precision=2)
                else:
                    orig_res[op][param] = _np_to_str(original_params[k]['parameters'][param][sample_idx], precision=2)

    return orig_res, pred_res


def _np_to_str(val: np.ndarray, precision=2) -> str:

    if val.size == 1:
        return np.format_float_positional(val.squeeze(), precision=precision)

    if val.size > 1:
        return np.array_str(val.squeeze(), precision=precision)

    return ''


def log_dict_recursive(tag: str, data_to_log, writer: TensorBoardLogger, step: int):

    if isinstance(data_to_log, np.float) or isinstance(data_to_log, np.int):
        writer.log_metrics({tag: data_to_log}, step)
        return

    if type(data_to_log) == list:
        data_to_log = np.asarray(data_to_log)

    if type(data_to_log) in [torch.Tensor, np.ndarray, int, float]:
        data_to_log = data_to_log.squeeze()
        if len(data_to_log.shape) == 0 or len(data_to_log) <= 1:
            writer.log_metrics({tag: data_to_log}, step)
        elif len(data_to_log) > 1:
            writer.experiment.add_histogram(tag, data_to_log, step)
        else:
            raise ValueError(f"Unexpected value to log {data_to_log}")
        return

    if not isinstance(data_to_log, dict):
        return

    if 'operation' in data_to_log:
        tag += '_' + data_to_log['operation']

    for k, v in data_to_log.items():
        log_dict_recursive(f'{tag}/{k}', v, writer, step)

    return


def get_param_diffs(predicted_params: dict, target_params: dict, ignore_params: Sequence[str]) -> (dict, dict):

    all_diffs, active_only_diffs = {}, {}
    predicted_params_np = to_numpy_recursive(predicted_params)
    target_params_np = to_numpy_recursive(target_params)

    for op_index, pred_op_dict in predicted_params_np.items():
        all_diffs[op_index] = {}
        active_only_diffs[op_index] = {}
        target_op_dict = target_params_np[op_index]
        if target_op_dict['operation'].ndim > 1:
            if target_op_dict['operation'].shape == (1, 1):
                op_config = synth_constants.param_configs[target_op_dict['operation'].squeeze(axis=0)[0]]
            else:
                op_config = synth_constants.param_configs[target_op_dict['operation'].squeeze()[0]]
        else:
            op_config = synth_constants.param_configs[target_op_dict['operation'][0]]

        for param_name, pred_vals in pred_op_dict['parameters'].items():

            if ignore_params is not None and param_name in ignore_params:
                continue

            target_vals = target_op_dict['parameters'][param_name]
            if pred_vals.ndim == 0 or (pred_vals.ndim == 1 and len(pred_vals) > 1):
                pred_vals = np.expand_dims(pred_vals, 0)

            if param_name == 'waveform':
                target_vals = target_vals.squeeze()
                if target_vals.ndim == 0:
                    waveform_idx = [synth_constants.wave_type_dict[target_vals.item()]]
                    if pred_vals[0][waveform_idx].ndim == 1:
                        diff = (1 - pred_vals[0][waveform_idx])
                    else:
                        diff = (1 - pred_vals[0][waveform_idx]).item()

                else:
                    waveform_idx = [synth_constants.wave_type_dict[wt] for wt in target_vals]
                    diff = [1 - v[idx] for idx, v in zip(waveform_idx, pred_vals)]
                    diff = np.asarray(diff).squeeze()
            elif param_name == 'filter_type':
                if target_vals.ndim == 0:
                    filter_type_idx = [synth_constants.filter_type_dict[target_vals.item()]]
                    diff = (1 - pred_vals[0][filter_type_idx]).item()
                else:
                    filter_type_idx = [synth_constants.filter_type_dict[ft] for ft in target_vals.squeeze()]
                    diff = [1 - v[idx] for idx, v in zip(filter_type_idx, pred_vals)]
                    diff = np.asarray(diff).squeeze()
            # elif param_name in ['attack_t', 'decay_t', 'sustain_t', 'sustain_level', 'release_t']:
            #     continue
            elif param_name == 'envelope':
                diff = [np.linalg.norm(pred_vals[k] - target_vals[k]) for k in range(pred_vals.shape[0])]
            elif param_name in ['active', 'fm_active']:
                active_targets = [0 if f else 1 for f in target_vals]
                softmax_pred_vals = softmax(pred_vals, axis=1)
                active_preds = np.argmax(softmax_pred_vals, axis=1)
                conf_mat = confusion_matrix(active_targets, active_preds, labels=[0, 1])

                true_negative = conf_mat[0][0]
                true_positive = conf_mat[1][1]

                accuracy = (true_negative + true_positive) / len(active_preds)
                all_diffs[op_index][f'{param_name}_accuracy'] = accuracy
                diff = [1 - v[idx] for idx, v in zip(active_targets, softmax_pred_vals)]
            else:
                if pred_vals.shape == (1, 1):
                    diff = np.abs(target_vals - pred_vals.squeeze(axis=0))
                else:
                    diff = np.abs(target_vals.squeeze() - pred_vals.squeeze())


            if param_name not in ['active', 'fm_active'] and pred_op_dict['operation'] not in ['env_adsr',
                                                                                               'lowpass_filter_adsr']:
                if op_config[param_name].get('activity_signal', None) == 'fm_active':
                    activity_signal = target_op_dict['parameters']['fm_active']
                else:
                    activity_signal = target_op_dict['parameters'].get('active', None)

                if activity_signal is not None:
                    active_diff = [v for i, v in enumerate(diff) if activity_signal[i]]
                    if len(active_diff) > 0:
                        active_only_diffs[op_index][param_name] = active_diff
                else:
                    active_only_diffs[op_index][param_name] = diff

            all_diffs[op_index][param_name] = diff

    return all_diffs, active_only_diffs


def get_activation(name, activations_dict: dict):

    def hook(layer, layer_input, layer_output):
        activations_dict[name] = layer_output.detach()

    return hook


def to_numpy_recursive(input_to_convert):

    if isinstance(input_to_convert, (int, float, np.integer, np.floating, str)):
        return np.asarray([input_to_convert])

    if isinstance(input_to_convert, np.ndarray):
        return input_to_convert

    if isinstance(input_to_convert, torch.Tensor):
        return input_to_convert.cpu().detach().numpy()

    if isinstance(input_to_convert, list):
        return np.asarray([to_numpy_recursive(item) for item in input_to_convert])

    if isinstance(input_to_convert, dict):
        return {k: to_numpy_recursive(v) for k, v in input_to_convert.items()}


def to_torch_recursive(input_to_convert, device, ignore_dtypes=None):

    if ignore_dtypes is not None and isinstance(input_to_convert, ignore_dtypes):
        return input_to_convert

    if isinstance(input_to_convert, torch.Tensor):
        return input_to_convert.to(device=device)

    if isinstance(input_to_convert, (int, np.integer)):
        return torch.tensor([input_to_convert], dtype=torch.int32, device=device)

    if isinstance(input_to_convert, (float, np.floating)):
        return torch.tensor([input_to_convert], dtype=torch.float32, device=device)

    if isinstance(input_to_convert, (bool, np.bool, np.bool_)):
        return torch.tensor([input_to_convert], dtype=torch.bool, device=device)

    if isinstance(input_to_convert, np.ndarray):
        return torch.tensor(input_to_convert, device=device)

    if isinstance(input_to_convert, list):
        return torch.tensor([to_torch_recursive(item, device=device, ignore_dtypes=ignore_dtypes)
                             for item in input_to_convert], device=device)

    if isinstance(input_to_convert, dict):
        return {k: to_torch_recursive(v, device, ignore_dtypes) for k, v in input_to_convert.items()}

    raise ValueError(f"Input of unexpected type {type(input_to_convert)}. Please add case to this function.")


def count_unpredicted_params(synth_preset_name, model_preset_name):

    synth_preset = synth_presets_dict[synth_preset_name]
    model_preset = synth_presets_dict[model_preset_name]

    predicted_indices = [cell['index'] for cell in model_preset]

    n_unpredicted_params = 0
    for cell in synth_preset:
        index = cell.get('index')
        operation = cell.get('operation')

        if index in predicted_indices or operation is None:
            continue

        op_params = synth_constants.modular_synth_params[operation]
        if op_params is not None:
            n_unpredicted_params += len(op_params)

    return n_unpredicted_params


def vectorize_unpredicted_params(target_params, model_preset, device):

    predicted_indices = [cell['index'] for cell in model_preset]

    unpredicted_params = []
    for index, params in target_params.items():

        if index in predicted_indices:
            continue

        op = params['operation']
        op_params = params['parameters']
        if op_params is None or op[0] in ['None', 'mix']:
            continue

        for param_name, param_val in op_params.items():
            if param_name == 'waveform':
                waveform_idx = [synth_constants.wave_type_dict[wt] for wt in param_val]
                param_val = torch.tensor(waveform_idx, device=device)
            elif param_name == 'filter_type':
                filter_type_idx = [synth_constants.filter_type_dict[ft] for ft in param_val]
                param_val = torch.tensor(filter_type_idx, device=device)
            else:
                param_val = torch.tensor(param_val, device=device)

            unpredicted_params.append(param_val)

    unpredicted_params_tensor = torch.stack(unpredicted_params, dim=1).float()

    return unpredicted_params_tensor


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


def remove_external_dims(item):

    if isinstance(item, list) and len(item) == 1:
        return remove_external_dims(item[0])
    elif isinstance(item, (np.ndarray, torch.Tensor)) and item.ndim > 0 and len(item) == 1:
        return remove_external_dims(item[0])
    return item


def process_categorical_variable(values: Sequence, map_fn: Callable, batch_size: int, return_one_hot: bool = True):

    if batch_size > 1:
        values = remove_external_dims(values)
    else:
        values = [remove_external_dims(values)]

    assert len(values) == batch_size

    processed_res = []
    for val in values:
        core_val = remove_external_dims(val)
        if map_fn is not None:
            idx = map_fn[core_val] if isinstance(map_fn, dict) else map_fn(core_val)
            if return_one_hot:
                processed_val = np.zeros(len(map_fn), dtype=np.float32)
                processed_val[idx] = 1.0
                processed_res.append(processed_val)
            else:
                processed_res.append(idx)
        else:
            processed_res.append(core_val)

    return processed_res


class MultiSpecTransform:

    def __init__(self, loss_type: str, loss_preset: Union[str, dict], synth_constants: SynthConstants, device='cuda:0'):

        super().__init__()

        self.loss_preset = loss_presets[loss_preset] if isinstance(loss_preset, str) else loss_preset
        self.device = device
        self.sample_rate = synth_constants.sample_rate

        self.spectrogram_ops = {}
        for size in self.loss_preset['fft_sizes']:
            if loss_type == 'BOTH' or loss_type == 'SPECTROGRAM':
                spec_transform = torchaudio.transforms.Spectrogram(n_fft=size, hop_length=int(size / 4), power=2.0).to(self.device)

                self.spectrogram_ops[f'{size}_spectrogram'] = spec_transform

            if loss_type == 'BOTH' or loss_type == 'MEL_SPECTROGRAM':
                mel_spec_transform = torchaudio.transforms.MelSpectrogram(sample_rate=self.sample_rate, n_fft=size,
                                                                          hop_length=int(size / 4), n_mels=256,
                                                                          power=2.0).to(self.device)

                self.spectrogram_ops[f'{size}_mel'] = mel_spec_transform

    def call(self, input_audio):
        if input_audio.shape[1] == 1:
            input_audio = torch.squeeze(input_audio, dim=1)
        max_row_size = 0
        max_col_size = 0
        spectrograms_dict = {}
        for loss_name, loss_op in self.spectrogram_ops.items():
            n_fft = loss_op.n_fft
            output_spec_mag = loss_op(input_audio.float())
            spectrograms_dict[str(n_fft)] = output_spec_mag

            row_size = output_spec_mag.shape[1]
            col_size = output_spec_mag.shape[2]
            if row_size > max_row_size:
                max_row_size = row_size
            if col_size > max_col_size:
                max_col_size = col_size

        self.zero_pad(spectrograms_dict, max_row_size, max_col_size)
        specs_list = []
        for key, spec in spectrograms_dict.items():
            specs_list.append(spec)

        specs_tuple = tuple(specs_list)
        specs_tensor = torch.stack(specs_tuple, dim=1)
        return specs_tensor

    def zero_pad(self, spectrograms_dict, target_row_size, target_col_size):
        for key, spec in spectrograms_dict.items():
            spec_row_size = spec.shape[1]
            spec_col_size = spec.shape[2]
            sizing_tuple = (0, target_col_size - spec_col_size, target_row_size - spec_row_size, 0)
            padding_fn = torch.nn.ConstantPad2d(sizing_tuple, value=0)
            spectrograms_dict[key] = padding_fn(spec)
