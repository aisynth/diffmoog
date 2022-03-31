import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


def log_gradients_in_model(model, writer: SummaryWriter, step):
    for tag, value in model.named_parameters():
        if value.grad is not None:
            grad_val = value.grad.cpu()
            writer.add_histogram(tag + "/grad", grad_val, step)
            if np.linalg.norm(grad_val) < 1e-4 and 'bias' not in tag:
                print(f"Op {tag} gradient approaching 0")


def parse_synth_params(original_params: dict, predicted_params: dict, sample_idx: int) -> (dict, dict):

    pred_res, orig_res = {}, {}

    for k, d in predicted_params.items():
        op = d['operation']
        pred_res[op] = {}
        orig_res[op] = {}
        for param, vals in d['params'].items():
            pred_res[op][param] = _np_to_str(vals[sample_idx].detach().cpu().numpy().squeeze(), precision=2)

            if param == 'waveform':
                orig_res[op][param] = original_params[k]['parameters'][param][sample_idx]
            else:
                orig_res[op][param] = \
                    _np_to_str(original_params[k]['parameters'][param][sample_idx].detach().cpu().numpy(), precision=2)

    return orig_res, pred_res


def _np_to_str(val: np.ndarray, precision=2) -> str:

    if val.size == 1:
        return np.format_float_positional(val.squeeze(), precision=precision)

    if val.size > 1:
        return np.array_str(val.squeeze(), precision=precision)

    return ''


def log_dict_recursive(tag: str, data_to_log, writer: SummaryWriter, step: int):

    if type(data_to_log) in [torch.Tensor, np.ndarray, int, float]:
        if len(data_to_log) > 1:
            writer.add_histogram(tag, data_to_log, step)
        else:
            writer.add_scalar(tag, data_to_log, step)
        return

    if not isinstance(data_to_log, dict):
        return

    if 'operation' in data_to_log:
        tag += '_' + data_to_log['operation']

    for k, v in data_to_log.items():
        log_dict_recursive(f'{tag}/{k}', v, writer, step)

    return


def get_activation(name, activations_dict: dict):

    def hook(layer, layer_input, layer_output):
        activations_dict[name] = layer_output.detach()

    return hook