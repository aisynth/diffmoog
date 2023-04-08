import copy
from collections import defaultdict
from typing import Any, Tuple, Optional

import numpy as np
import torch
import torchaudio
import ast
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.optim.lr_scheduler import ConstantLR, ReduceLROnPlateau

from model.loss.parameters_loss import ParametersLoss
from model.loss.spectral_loss import SpectralLoss, ControlSpectralLoss
from model.model import SynthNetwork
from synth.parameters_normalizer import Normalizer
from synth.synth_architecture import SynthModular
from synth.synth_constants import synth_constants
from utils.metrics import lsd, pearsonr_dist, mae, mfcc_distance, spectral_convergence, paper_lsd
from utils.train_utils import log_dict_recursive, parse_synth_params, get_param_diffs, to_numpy_recursive, MultiSpecTransform
from utils.visualization_utils import visualize_signal_prediction


class LitModularSynth(LightningModule):

    def __init__(self, train_cfg, device, tuning_mode=False):

        super().__init__()

        self.cfg = train_cfg
        self.tuning_mode = tuning_mode
        self.synth = SynthModular(preset_name=train_cfg.synth.preset, synth_constants=synth_constants,
                                  device=device)

        self.ignore_params = train_cfg.synth.get('ignore_params', None)

        self.synth_net = SynthNetwork(cfg=self.cfg,
                                      synth_preset=train_cfg.model.preset,
                                      loss_preset=train_cfg.loss.preset,
                                      device=device,
                                      backbone=train_cfg.model.backbone
                                      )
        self.normalizer = Normalizer(train_cfg.synth.note_off_time, train_cfg.synth.signal_duration, synth_constants)

        self.use_multi_spec_input = train_cfg.synth.use_multi_spec_input

        if train_cfg.synth.transform.lower() == 'mel':
            self.signal_transform = torchaudio.transforms.MelSpectrogram(sample_rate=synth_constants.sample_rate,
                                                                         n_fft=1024, hop_length=256, n_mels=128,
                                                                         power=2.0, f_min=0, f_max=8000).to(device)
        elif train_cfg.synth.transform.lower() == 'spec':
            self.signal_transform = torchaudio.transforms.Spectrogram(n_fft=512, power=2.0).to(device)
        else:
            raise NotImplementedError(f'Input transform {train_cfg.transform} not implemented.')

        self.multi_spec_transform = MultiSpecTransform(loss_type=train_cfg.loss.spec_loss_type,
                                                       loss_preset=train_cfg.loss.preset,
                                                       synth_constants=synth_constants, device=device)

        self.spec_loss = SpectralLoss(loss_type=train_cfg.loss.spec_loss_type,
                                      loss_preset=train_cfg.loss.preset,
                                      synth_constants=synth_constants, device=device)

        self.control_spec_loss = ControlSpectralLoss(signal_duration=train_cfg.synth.signal_duration,
                                                     loss_type=train_cfg.loss.spec_loss_type,
                                                     preset_name=train_cfg.loss.control_spec_preset,
                                                     synth_constants=synth_constants,
                                                     device=device)

        self.params_loss = ParametersLoss(loss_type=train_cfg.loss.parameters_loss_type,
                                          synth_constants=synth_constants, ignore_params=self.ignore_params,
                                          device=device)

        self.epoch_param_diffs = defaultdict(list)
        self.epoch_vals_raw = defaultdict(list)
        self.epoch_vals_normalized = defaultdict(list)
        self.epoch_param_active_diffs = defaultdict(list)

        self.val_epoch_param_diffs = defaultdict(list)
        self.val_epoch_param_active_diffs = defaultdict(list)
        self.tb_logger = None

    def forward(self, raw_signal: torch.Tensor, *args, **kwargs) -> Any:

        assert len(raw_signal.shape) == 2, f'Expected tensor of dimensions [batch_size, signal_length]' \
                                           f' but got shape {raw_signal.shape}'

        raw_signal = torch.unsqueeze(raw_signal, 1)

        # Transform raw signal to spectrogram
        if self.use_multi_spec_input:
            spectrograms = self.multi_spec_transform.call(raw_signal)
        else:
            spectrograms = self.signal_transform(raw_signal)

        # Run NN model and convert predicted params from (0, 1) to original range
        predicted_parameters_unit_range = self.synth_net(spectrograms)
        predicted_params_full_range = self.normalizer.denormalize(predicted_parameters_unit_range)

        return predicted_parameters_unit_range, predicted_params_full_range

    def generate_synth_sound(self, full_range_synth_params: dict, batch_size: int) -> Tuple[torch.Tensor, dict]:

        # Inject synth parameters to the modular synth
        self.synth.update_cells_from_dict(full_range_synth_params)

        # Generate sound
        pred_final_signal, pred_signals_through_chain = \
            self.synth.generate_signal(signal_duration=self.cfg.synth.signal_duration, batch_size=batch_size)

        return pred_final_signal, pred_signals_through_chain

    def in_domain_step(self, batch, log: bool = False, return_metrics=False):

        target_signal, target_params_full_range, signal_index = batch
        batch_size = len(signal_index)

        predicted_params_unit_range, predicted_params_full_range = self(target_signal)
        target_params_unit_range = self.normalizer.normalize(target_params_full_range)

        if self.ignore_params is not None:
            predicted_params_unit_range = self._update_param_dict(target_params_unit_range, predicted_params_unit_range)
            predicted_params_full_range = self._update_param_dict(target_params_full_range, predicted_params_full_range)

        total_params_loss, per_parameter_loss = self.params_loss.call(predicted_params_unit_range,
                                                                      target_params_unit_range)
        pred_final_signal = None
        if self.global_step < self.cfg.loss.spectrogram_loss_warmup and not return_metrics:
            spec_loss = torch.tensor(0)
        else:
            pred_final_signal, pred_signals_through_chain = self.generate_synth_sound(predicted_params_full_range,
                                                                                      batch_size)
            if self.cfg.loss.use_chain_loss:
                _, target_signals_through_chain = self.generate_synth_sound(target_params_full_range, batch_size)
                spec_loss = self._calculate_spectrogram_chain_loss(target_signals_through_chain,
                                                                   pred_signals_through_chain, log=True)
            else:
                spec_loss, per_op_loss, per_op_weighted_loss = self.spec_loss.call(target_signal, pred_final_signal,
                                                                                   step=self.global_step)
                self._log_recursive(per_op_weighted_loss, f'final_spec_losses_weighted')

        loss_total, weighted_params_loss, weighted_spec_loss = self._balance_losses(total_params_loss, spec_loss, log)
        param_diffs, active_only_diffs = get_param_diffs(predicted_params_full_range.copy(),
                                                         target_params_full_range.copy(), self.ignore_params)

        step_losses = {'raw_params_loss': total_params_loss.detach(), 'raw_spec_loss': spec_loss.detach(),
                       'weighted_params_loss': weighted_params_loss.detach(),
                       'weighted_spec_loss': weighted_spec_loss.detach(), 'loss_total': loss_total.item()}

        step_artifacts = {'raw_predicted_parameters': predicted_params_unit_range,
                          'full_range_predicted_parameters': predicted_params_full_range, 'param_diffs': param_diffs,
                          'active_only_diffs': active_only_diffs}

        if return_metrics:
            step_metrics = self._calculate_audio_metrics(target_signal, pred_final_signal)
            return loss_total, step_losses, step_metrics, step_artifacts

        if self.tuning_mode:
            if pred_final_signal is None:
                pred_final_signal, pred_signals_through_chain = self.generate_synth_sound(predicted_params_full_range,
                                                                                          batch_size)
            lsd_val = paper_lsd(target_signal, pred_final_signal)

            step_losses['train_lsd'] = lsd_val

        return loss_total, step_losses, step_artifacts

    def out_of_domain_step(self, batch, return_metrics=False):

        target_final_signal, signal_index = batch
        batch_size = len(signal_index)

        predicted_params_unit_range, predicted_params_full_range = self(target_final_signal)
        pred_final_signal, _ = self.generate_synth_sound(predicted_params_full_range, batch_size)

        loss, per_op_loss, per_op_weighted_loss = self.spec_loss.call(target_final_signal, pred_final_signal,
                                                                      step=self.global_step)

        weighted_spec_loss = self.cfg.loss.spectrogram_loss_weight * loss
        step_losses = {'raw_spec_loss': loss.detach(), 'weighted_spec_loss': weighted_spec_loss.detach()}

        step_artifacts = {'raw_predicted_parameters': predicted_params_unit_range,
                          'full_range_predicted_parameters': predicted_params_full_range,
                          'per_op_spec_loss_raw': per_op_loss, 'per_op_spec_loss_weighted': per_op_weighted_loss}

        if return_metrics:
            step_metrics = self._calculate_audio_metrics(target_final_signal, pred_final_signal)
            return weighted_spec_loss, step_losses, step_metrics, step_artifacts

        return weighted_spec_loss, step_losses, step_artifacts

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:

        if batch_idx == 0:
            target_params = batch[1] if len(batch) == 3 else None
            self._log_sounds_batch(batch[0], target_params, f'samples_train')

        if self.cfg.loss.in_domain_epochs < self.current_epoch:
            assert len(batch) == 2, "Tried to run OOD step on in domain batch"
            loss, step_losses, step_artifacts = self.out_of_domain_step(batch)
        else:
            loss, step_losses, step_artifacts = self.in_domain_step(batch, log=True)
            self._accumulate_batch_values(self.epoch_param_diffs, step_artifacts['param_diffs'])
            self._accumulate_batch_values(self.epoch_param_active_diffs, step_artifacts['active_only_diffs'])

        self._log_recursive(step_losses, f'train_losses')

        self._accumulate_batch_values(self.epoch_vals_raw, step_artifacts['raw_predicted_parameters'])
        self._accumulate_batch_values(self.epoch_vals_normalized, step_artifacts['full_range_predicted_parameters'])
        step_losses['loss'] = loss
        return step_losses

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, dataloader_idx) -> Optional[STEP_OUTPUT]:
        val_name = 'in_domain_validation' if dataloader_idx == 0 else 'nsynth_validation'

        if batch_idx == 0:
            target_params = batch[1] if dataloader_idx == 0 else None
            self._log_sounds_batch(batch[0], target_params, val_name)

        self.synth_net.train()
        if 'in_domain' in val_name:
            loss, step_losses, step_metrics, step_artifacts = self.in_domain_step(batch, return_metrics=True)
        else:
            loss, step_losses, step_metrics, step_artifacts = self.out_of_domain_step(batch, return_metrics=True)

        self._log_recursive(step_losses, f'{val_name}_losses')
        self._log_recursive(step_metrics, f'{val_name}_metrics')

        if dataloader_idx == 0:
            self._accumulate_batch_values(self.val_epoch_param_diffs, step_artifacts['param_diffs'])
            self._accumulate_batch_values(self.val_epoch_param_active_diffs, step_artifacts['active_only_diffs'])

        return loss

    @torch.no_grad()
    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        target_final_signal = batch[0]
        batch_size = len(target_final_signal)

        predicted_params_unit_range, predicted_params_full_range = self(target_final_signal)
        pred_final_signal, pred_signals_through_chain = self.generate_synth_sound(predicted_params_full_range,
                                                                                  batch_size)

        return pred_final_signal, predicted_params_full_range

    def on_train_epoch_end(self) -> None:
        self._log_recursive(self.epoch_param_diffs, 'param_diff')
        self._log_recursive(self.epoch_param_active_diffs, 'active_param_diff')
        self._log_recursive(self.epoch_vals_raw, 'param_values_raw')
        self._log_recursive(self.epoch_vals_normalized, 'param_values_normalized')

        self.epoch_param_diffs = defaultdict(list)
        self.epoch_vals_raw = defaultdict(list)
        self.epoch_vals_normalized = defaultdict(list)
        self.epoch_param_active_diffs = defaultdict(list)

        return

    def on_validation_epoch_end(self) -> None:
        self._log_recursive(self.val_epoch_param_diffs, 'validation_param_diff')
        self._log_recursive(self.val_epoch_param_active_diffs, 'validation_active_param_diff')
        self.val_epoch_param_diffs = defaultdict(list)
        self.val_epoch_param_active_diffs = defaultdict(list)

    def _calculate_spectrogram_chain_loss(self, target_signals_through_chain: dict, pred_signals_through_chain: dict,
                                          log=False):

        chain_losses = torch.zeros(len(target_signals_through_chain))
        for i, op_index in enumerate(target_signals_through_chain.keys()):
            op_index = str(op_index)

            op_index_tuple = ast.literal_eval(op_index)
            current_layer = int(op_index_tuple[1])
            current_channel = int(op_index_tuple[0])
            c_target_operation = self.synth.synth_matrix[current_channel][current_layer].operation

            if self.cfg.loss.use_gradual_chain_loss:
                layer_warmup_factor = self.cfg.loss.chain_warmup_factor * current_layer + self.cfg.loss.spectrogram_loss_warmup

                if self.global_step < layer_warmup_factor:
                    continue

            c_pred_signal = pred_signals_through_chain[op_index]
            if c_pred_signal is None:
                continue
            c_target_signal = target_signals_through_chain[op_index]

            if 'lfo' in c_target_operation:
                loss, per_op_loss, per_op_weighted_loss, ret_spectrograms = \
                    self.control_spec_loss.call(c_target_signal,
                                                c_pred_signal,
                                                step=self.global_step,
                                                return_spectrogram=True)
            else:
                loss, per_op_loss, per_op_weighted_loss, ret_spectrograms = \
                    self.spec_loss.call(c_target_signal,
                                        c_pred_signal,
                                        step=self.global_step,
                                        return_spectrogram=True)
            chain_losses[i] = loss

            if log:
                self._log_recursive(per_op_weighted_loss, f'weighted_chain_spec_loss/{op_index}')

        spectrogram_loss = chain_losses.mean() * self.cfg.loss.chain_loss_weight

        return spectrogram_loss

    def _balance_losses(self, parameters_loss, spectrogram_loss, log=False):

        step = self.global_step
        cfg = self.cfg.loss

        parameters_loss_decay_factor = cfg.min_parameters_loss_decay
        spec_loss_rampup_factor = 1

        if step < cfg.spectrogram_loss_warmup:
            parameters_loss_decay_factor = 1.0
            spec_loss_rampup_factor = 0
        elif step < (cfg.spectrogram_loss_warmup + cfg.loss_switch_steps):
            linear_mix_factor = (step - cfg.spectrogram_loss_warmup) / cfg.loss_switch_steps
            parameters_loss_decay_factor = max(1 - linear_mix_factor, cfg.min_parameters_loss_decay)
            spec_loss_rampup_factor = linear_mix_factor

        weighted_params_loss = parameters_loss * cfg.parameters_loss_weight * parameters_loss_decay_factor
        weighted_spec_loss = cfg.spectrogram_loss_weight * spectrogram_loss * spec_loss_rampup_factor

        loss_total = weighted_params_loss + weighted_spec_loss

        if log:
            self.tb_logger.add_scalar('loss/parameters_decay_factor', parameters_loss_decay_factor, self.global_step)
            self.tb_logger.add_scalar('loss/spec_loss_rampup_factor', spec_loss_rampup_factor, self.global_step)

        return loss_total, weighted_params_loss, weighted_spec_loss

    @torch.no_grad()
    def _calculate_audio_metrics(self, target_signal: torch.Tensor, predicted_signal: torch.Tensor):

        metrics = {}

        target_signal = target_signal.float()
        predicted_signal = predicted_signal.float()

        target_spec = self.signal_transform(target_signal)
        predicted_spec = self.signal_transform(predicted_signal)

        metrics['paper_lsd_value'] = paper_lsd(target_signal, predicted_signal)
        metrics['lsd_value'] = lsd(target_spec, predicted_spec, reduction=torch.mean)
        metrics['pearson_stft'] = pearsonr_dist(target_spec, predicted_spec, input_type='spec', reduction=torch.mean)
        metrics['pearson_fft'] = pearsonr_dist(target_signal, predicted_signal, input_type='audio',
                                               reduction=torch.mean)
        metrics['mean_average_error'] = mae(target_spec, predicted_spec, reduction=torch.mean)
        metrics['mfcc_mae'] = mfcc_distance(target_signal, predicted_signal, sample_rate=synth_constants.sample_rate,
                                            device=predicted_signal.device, reduction=torch.mean)
        metrics['spectral_convergence_value'] = spectral_convergence(target_spec, predicted_spec, reduction=torch.mean)

        return metrics

    def _log_sounds_batch(self, target_signals, target_parameters, tag: str):

        batch_size = len(target_signals)

        predicted_params_unit_range, predicted_params_full_range = self(target_signals)
        pred_final_signal, pred_signals_through_chain = self.generate_synth_sound(predicted_params_full_range,
                                                                                  batch_size)

        for i in range(self.cfg.logging.n_images_to_log):
            if target_parameters is not None:
                sample_params_orig, sample_params_pred = parse_synth_params(target_parameters,
                                                                            predicted_params_full_range, i)
            else:
                sample_params_orig, sample_params_pred = {}, {}

            self.tb_logger.add_audio(f'{tag}/input_{i}_target', target_signals[i],
                                     global_step=self.current_epoch, sample_rate=synth_constants.sample_rate)
            self.tb_logger.add_audio(f'{tag}/input_{i}_pred', pred_final_signal[i],
                                     global_step=self.current_epoch, sample_rate=synth_constants.sample_rate)

            signal_vis = visualize_signal_prediction(target_signals[i], pred_final_signal[i], sample_params_orig,
                                                     sample_params_pred, db=True)
            signal_vis_t = torch.tensor(signal_vis, dtype=torch.uint8, requires_grad=False)

            self.tb_logger.add_image(f'{tag}/{256}_spec/input_{i}', signal_vis_t, global_step=self.current_epoch,
                                     dataformats='HWC')

    def _update_param_dict(self, src_dict: dict, target_dict: dict):

        for op_idx, op_dict in src_dict.items():
            if isinstance(op_dict['parameters'], list) or op_dict['operation'][0] == 'mix':
                continue
            for param_name, param_vals in op_dict['parameters'].items():
                if param_name in self.ignore_params:
                    target_dict[op_idx]['parameters'][param_name] = param_vals

        return target_dict

    def _log_recursive(self, items_to_log: dict, tag: str, on_epoch=False):
        if isinstance(items_to_log, np.float) or isinstance(items_to_log, np.int):
            self.log(tag, items_to_log, on_step=True, on_epoch=on_epoch)
            return

        if type(items_to_log) == list:
            items_to_log = np.asarray(items_to_log)

        if type(items_to_log) in [torch.Tensor, np.ndarray, int, float]:
            items_to_log = items_to_log.squeeze()
            if len(items_to_log.shape) == 0 or len(items_to_log) <= 1:
                if isinstance(items_to_log, (np.ndarray, np.generic)):
                    items_to_log = torch.tensor(items_to_log)
                self.log(tag, items_to_log, batch_size=self.cfg.model.batch_size)
            elif len(items_to_log) > 1:
                self.tb_logger.add_histogram(tag, items_to_log, self.current_epoch)
            else:
                raise ValueError(f"Unexpected value to log {items_to_log}")
            return

        if not isinstance(items_to_log, dict):
            return

        if 'operation' in items_to_log:
            tag += '_' + items_to_log['operation']

        for k, v in items_to_log.items():
            self._log_recursive(v, f'{tag}/{k}', on_epoch)

        return

    @staticmethod
    def _accumulate_batch_values(accumulator: dict, batch_vals: dict):

        batch_vals_np = to_numpy_recursive(batch_vals)

        for op_idx, op_dict in batch_vals_np.items():
            if 'parameters' in op_dict:
                acc_values = op_dict['parameters']
            else:
                acc_values = op_dict
            for param_name, param_vals in acc_values.items():
                accumulator[f'{op_idx}_{param_name}'].extend(param_vals)

    def configure_optimizers(self):

        optimizer_params = self.cfg.model.optimizer

        # Configure optimizer
        if 'optimizer' not in optimizer_params or optimizer_params['optimizer'].lower() == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=optimizer_params.base_lr)
        else:
            raise NotImplementedError(f"Optimizer {self.optimizer_params['optimizer']} not implemented")

        # Configure learning rate scheduler
        if 'scheduler' not in optimizer_params or optimizer_params.scheduler.lower() == 'constant':
            scheduler_config = {"scheduler": ConstantLR(optimizer)}
        elif optimizer_params.scheduler.lower() == 'reduce_on_plateau':
            scheduler_config = {"scheduler": ReduceLROnPlateau(optimizer),
                                "interval": "epoch",
                                "monitor": "val_loss",
                                "frequency": 3,
                                "strict": True}
        elif optimizer_params.scheduler.lower() == 'cosine':
            scheduler_config = {"scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.cfg.model.num_epochs),
                "interval": "epoch"}
        elif optimizer_params.scheduler.lower() == 'cyclic':
            scheduler_config = {"scheduler": torch.optim.lr_scheduler.CyclicLR(
                optimizer, base_lr=self.cfg.model.optimizer.base_lr, max_lr=self.cfg.model.optimizer.max_lr,
                step_size_up=self.cfg.model.optimizer.cyclic_step_size_up),
                "interval": "step"}
        elif optimizer_params.scheduler.lower() == 'exponential':
            scheduler_config = {"scheduler": torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=self.cfg.model.optimizer.gamma),
                "interval": "epoch"}
        else:
            raise NotImplementedError(f"Scheduler {self.optimizer_params['scheduler']} not implemented")

        return {"optimizer": optimizer, "lr_scheduler": scheduler_config}
