from typing import Sequence

import torch
from torch import nn

from synth.synth_constants import SynthConstants


class ParametersLoss(nn.Module):
    """This loss compares target and predicted parameters of the modular synthesizer"""

    def __init__(self, loss_norm: str, synth_constants: SynthConstants, ignore_params: Sequence[str] = None,
                 device='cuda:0'):

        super().__init__()

        self.synth_constants = synth_constants
        self.device = device
        if loss_norm == 'L1':
            self.criterion = nn.L1Loss()
        elif loss_norm == 'L2':
            self.criterion = nn.MSELoss()
        else:
            raise ValueError("unknown loss type")

        self.ignore_params = ignore_params if ignore_params is not None else []

        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def call(self, predicted_parameters_dict, target_parameters_dict):
        """ execute parameters loss computation between two parameter sets

                Args:
                  :param predicted_parameters_dict: predicted audio parameters
                  :param target_parameters_dict: target audio parameters
                """
        total_loss = 0.0
        loss_dict = {}
        for key in predicted_parameters_dict.keys():
            operation = predicted_parameters_dict[key]['operation']
            op_config = self.synth_constants.param_configs[operation]

            predicted_parameters = predicted_parameters_dict[key]['parameters']
            target_parameters = target_parameters_dict[key]['parameters']

            for param in predicted_parameters.keys():

                if param in self.ignore_params:
                    continue

                if param == 'waveform':
                    waveform_list = [self.synth_constants.wave_type_dict[waveform] for waveform
                                     in target_parameters[param]]
                    target = torch.tensor(waveform_list)
                elif param == 'filter_type':
                    filter_type_list = [self.synth_constants.filter_type_dict[filter_type] for
                                        filter_type in target_parameters[param]]
                    target = torch.tensor(filter_type_list)
                elif param in ['active', 'fm_active']:
                    active_list = [0 if is_active else 1 for is_active in target_parameters[param]]
                    target = torch.tensor(active_list)
                else:
                    target = target_parameters[param].clone()

                target = target.type(torch.FloatTensor).to(self.device)
                if predicted_parameters[param].dim() > 1 and predicted_parameters[param].shape[0] == 1:
                    pred = predicted_parameters[param].squeeze(dim=0)
                else:
                    pred = predicted_parameters[param].squeeze()

                # if param not in ['active', 'fm_active'] and operation not in ['env_adsr']:
                #     if op_config[param].get('activity_signal', None) == 'fm_active':
                #         activity_signal = target_parameters['fm_active']
                #     elif 'active' in target_parameters:
                #         activity_signal = target_parameters['active']
                #     else:
                #         activity_signal = None
                #
                #     if activity_signal is not None:
                #         if pred.dim() > 1:
                #             pred = pred * activity_signal.long().squeeze().unsqueeze(-1)
                #         else:
                #             pred = pred * activity_signal.long().squeeze()

                if target.dim() > 1:
                    target = target.squeeze(dim=0)

                if param in ['waveform', 'filter_type', 'active', 'fm_active']:
                    target = target.to(self.device)

                    if param == 'waveform':
                        target = target.type(torch.LongTensor).to(self.device)
                        if pred.dim() == 1:
                            pred = pred.unsqueeze(dim=0)
                        loss = self.ce_loss(pred, target)
                    else:
                        loss = self.bce_loss(pred, target)
                else:
                    loss = self.criterion(pred, target)

                total_loss += loss

                loss_dict[f"{key}_{operation}_{param}"] = loss

        return total_loss, loss_dict
