import numpy as np
import torch

from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import confusion_matrix
from scipy.special import softmax

from synth.synth_constants import synth_structure
from synth.synth_presets import synth_presets_dict


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


def log_dict_recursive(tag: str, data_to_log, writer: SummaryWriter, step: int):

    if isinstance(data_to_log, np.float) or isinstance(data_to_log, np.int):
        writer.add_scalar(tag, data_to_log, step)
        return

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
    predicted_params_np = to_numpy_recursive(predicted_params)
    target_params_np = to_numpy_recursive(target_params)

    for op_index, pred_op_dict in predicted_params_np.items():
        all_diffs[op_index] = {}
        target_op_dict = target_params_np[op_index]
        for param_name, pred_vals in pred_op_dict['parameters'].items():

            target_vals = target_op_dict['parameters'][param_name]
            if pred_vals.ndim == 0 or (pred_vals.ndim == 1 and len(pred_vals) > 1):
                pred_vals = np.expand_dims(pred_vals, 0)

            if param_name == 'waveform':
                target_vals = target_vals.squeeze()
                if target_vals.ndim == 0:
                    waveform_idx = [synth_structure.wave_type_dict[target_vals.item()]]
                    diff = (1 - pred_vals[0][waveform_idx]).item()
                else:
                    waveform_idx = [synth_structure.wave_type_dict[wt] for wt in target_vals]
                    diff = [1 - v[idx] for idx, v in zip(waveform_idx, pred_vals)]
                    diff = np.asarray(diff).squeeze()
            elif param_name == 'filter_type':
                if target_vals.ndim == 0:
                    filter_type_idx = [synth_structure.filter_type_dict[target_vals.item()]]
                    diff = (1 - pred_vals[0][filter_type_idx]).item()
                else:
                    filter_type_idx = [synth_structure.filter_type_dict[ft] for ft in target_vals.squeeze()]
                    diff = [1 - v[idx] for idx, v in zip(filter_type_idx, pred_vals)]
                    diff = np.asarray(diff).squeeze()
            elif param_name in ['attack_t', 'decay_t', 'sustain_t', 'sustain_level', 'release_t']:
                continue
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
                diff = np.abs(target_vals.squeeze() - pred_vals.squeeze())

            all_diffs[op_index][param_name] = diff

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

        op_params = synth_structure.modular_synth_params[operation]
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
                waveform_idx = [synth_structure.wave_type_dict[wt] for wt in param_val]
                param_val = torch.tensor(waveform_idx, device=device)
            elif param_name == 'filter_type':
                filter_type_idx = [synth_structure.filter_type_dict[ft] for ft in param_val]
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
