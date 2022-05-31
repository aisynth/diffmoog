import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from model.helper import move_to
from synth.synth_modules import make_envelope_shape

from config import Config


class ParametersLoss:
    """This loss compares target and predicted parameters of the modular synthesizer"""

    def __init__(self, cfg: Config, loss_type: str, device='cuda:0'):
        self.device = device
        self.cfg = cfg
        if loss_type == 'L1':
            self.criterion = nn.L1Loss()
        elif loss_type == 'L2':
            self.criterion = nn.MSELoss()
        else:
            raise ValueError("unknown loss type")

    def call(self, predicted_parameters_dict, target_parameters_dict, summary_writer: SummaryWriter,
             global_step: int, log: bool = True):
        """ execute parameters loss computation between two parameter sets

                Args:
                  :param predicted_parameters: predicted audio parameters
                  :param target_parameters: target audio parameters
                  :param summary_writer: tensorboard summary writer
                  :param global_step: global step for summary_writer
                  :param log: log results flag
                """
        total_loss = 0.0
        loss_dict = {}
        for key in predicted_parameters_dict.keys():
            operation = predicted_parameters_dict[key]['operation']
            predicted_parameters = predicted_parameters_dict[key]['parameters']
            target_parameters = target_parameters_dict[key]['parameters']

            for param in predicted_parameters.keys():
                if param == 'waveform':
                    waveform_list = target_parameters['waveform']
                    for idx, waveform in enumerate(waveform_list):
                        if waveform == 'sine':
                            probabilities = [1., 0., 0.]
                        elif waveform == 'square':
                            probabilities = [0., 1., 0.]
                        elif waveform == 'sawtooth':
                            probabilities = [0., 0., 1.]
                        else:
                            probabilities = -1
                            KeyError("Unknown waveform type")

                        waveform_list[idx] = probabilities

                    target_parameters['waveform'] = torch.tensor(waveform_list)
                elif param == 'filter_type':
                    filter_type_list = target_parameters['filter_type']
                    for idx, filter_type in enumerate(filter_type_list):
                        if filter_type == 'low_pass':
                            probabilities = [1., 0.]
                        elif filter_type == 'high_pass':
                            probabilities = [0., 1.]
                        else:
                            probabilities = -1
                            KeyError("Unknown filter_type")
                        filter_type_list[idx] = probabilities

                    target_parameters['filter_type'] = torch.tensor(filter_type_list)
                elif param == 'envelope':
                    attack_t = target_parameters['attack_t']
                    decay_t = target_parameters['decay_t']
                    sustain_t = target_parameters['sustain_t']
                    sustain_level = target_parameters['sustain_level']
                    release_t = target_parameters['release_t']

                    if attack_t.dim() == 0:
                        num_sounds = 1
                    else:
                        num_sounds = attack_t.shape[0]

                    envelope_shape = make_envelope_shape(attack_t,
                                                         decay_t,
                                                         sustain_t,
                                                         sustain_level,
                                                         release_t,
                                                         self.cfg.signal_duration_sec,
                                                         self.cfg.sample_rate,
                                                         self.device,
                                                         num_sounds=num_sounds)
                    target_parameters['envelope'] = envelope_shape
                elif param in ['attack_t', 'decay_t', 'sustain_t', 'sustain_level', 'release_t']:
                    continue

                # move_to(target_parameters[param], self.device)
                target_parameters[param] = target_parameters[param].type(torch.FloatTensor).to(self.device)
                if predicted_parameters[param].dim() > 1 and predicted_parameters[param].shape[0] == 1:
                    predicted_parameters[param] = predicted_parameters[param].squeeze(dim=0)
                else:
                    predicted_parameters[param] = predicted_parameters[param].squeeze()
                if target_parameters[param].dim() > 1:
                    target_parameters[param] = target_parameters[param].squeeze(dim=0)
                loss = self.criterion(predicted_parameters[param], target_parameters[param])
                total_loss += loss

                loss_dict[f"{key}_{operation}_{param}"] = loss

                if log:
                    for loss_name, loss_val in loss_dict.items():
                        summary_writer.add_scalar(f"parameter_sub_losses/{loss_name}",
                                                  loss_val,
                                                  global_step=global_step)
        return total_loss
