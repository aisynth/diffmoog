import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from config import SynthConfig


def log_gradients_in_model(model, writer: SummaryWriter, step):
    for tag, value in model.named_parameters():
        if value.grad is not None:
            grad_val = value.grad.cpu()
            writer.add_histogram(tag + "/grad", grad_val, step)
            if np.linalg.norm(grad_val) < 1e-4 and 'bias' not in tag:
                print(f"Op {tag} gradient approaching 0")


def parse_synth_params(original_params: dict, predicted_params: dict, sample_idx: int) -> (dict, dict):

    pred_res, orig_res = {}, {}

    original_params = to_numpy_recursive(original_params)
    predicted_params = to_numpy_recursive(predicted_params)

    for k, d in predicted_params.items():
        op = d['operation'] if isinstance(d['operation'], str) else d['operation'][0]
        if op in pred_res:
            op = op + '_|'
        pred_res[op] = {}
        orig_res[op] = {}
        for param, vals in d['parameters'].items():

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


def log_dict_recursive(tag: str, data_to_log, writer: SummaryWriter, step: int):

    if type(data_to_log) == list:
        data_to_log = np.asarray(data_to_log)

    if type(data_to_log) in [torch.Tensor, np.ndarray, int, float]:
        data_to_log = data_to_log.squeeze()
        if len(data_to_log.shape) == 0 or len(data_to_log) <= 1:
            writer.add_scalar(tag, data_to_log, step)
        elif len(data_to_log) > 1:
            writer.add_histogram(tag, data_to_log, step)
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


def get_param_diffs(predicted_params: dict, target_params: dict) -> dict:

    all_diffs = {}
    predicted_params = to_numpy_recursive(predicted_params)
    target_params = to_numpy_recursive(target_params)

    for op_index, pred_op_dict in predicted_params.items():
        target_op_dict = target_params[op_index]
        for param_name, pred_vals in pred_op_dict['parameters'].items():

            target_vals = target_op_dict['parameters'][param_name]
            if pred_vals.ndim == 0 or (pred_vals.ndim == 1 and len(pred_vals) > 1):
                pred_vals = np.expand_dims(pred_vals, 0)

            if param_name == 'waveform':
                waveform_idx = [SynthConfig.wave_type_dict[wt] for wt in target_vals]
                diff = [1 - v[idx] for idx, v in zip(waveform_idx, pred_vals)]
                diff = np.asarray(diff).squeeze()
            elif param_name == 'filter_type':
                filter_type_idx = [SynthConfig.filter_type_dict[ft] for ft in target_vals]
                diff = [1 - v[idx] for idx, v in zip(filter_type_idx, pred_vals)]
                diff = np.asarray(diff).squeeze()
            else:
                diff = np.abs(target_vals.squeeze() - pred_vals.squeeze())

            all_diffs[f'{op_index}/{param_name}'] = diff

    return all_diffs


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
