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
from utils.train_utils import log_dict_recursive, parse_synth_params, get_param_diffs, to_numpy_recursive, \
    MultiSpecTransform
from utils.visualization_utils import visualize_signal_prediction
from typing import Dict, Any


class LitModularSynth(LightningModule):

    def __init__(self, train_cfg, device, tuning_mode=False):

        super().__init__()

        self.cfg = train_cfg
        self.lr = self.cfg.model.optimizer.base_lr
        self.tuning_mode = tuning_mode
        self.synth = SynthModular(chain_name=train_cfg.synth.chain, synth_constants=synth_constants,
                                  device=device)

        self.ignore_params = train_cfg.synth.get('ignore_params', None)

        self.synth_net = SynthNetwork(cfg=self.cfg,
                                      synth_chain=train_cfg.model.chain,
                                      loss_preset=train_cfg.loss.loss_preset,
                                      device=device,
                                      backbone=train_cfg.model.backbone
                                      )
        self.normalizer = Normalizer(train_cfg.synth.note_off_time, train_cfg.synth.signal_duration, synth_constants)

        self.use_multi_spec_input = train_cfg.synth.use_multi_spec_input

        if train_cfg.synth.transform.lower() == 'mel':
            self.signal_transform = torchaudio.transforms.MelSpectrogram(sample_rate=synth_constants.sample_rate,
                                                                         n_fft=1024, hop_length=256, n_mels=128,
                                                                         power=1.0, center=True).to(device)
        elif train_cfg.synth.transform.lower() == 'spec':
            self.signal_transform = torchaudio.transforms.Spectrogram(n_fft=512, power=2.0).to(device)
        else:
            raise NotImplementedError(f'Input transform {train_cfg.transform} not implemented.')

        self.multi_spec_transform = MultiSpecTransform(loss_preset=train_cfg.loss.loss_preset,
                                                       synth_constants=synth_constants, device=device)

        self.apply_log_transform_to_input = train_cfg.synth.get('apply_log_transform_to_input', False)

        self.spec_loss = SpectralLoss(loss_preset=train_cfg.loss.loss_preset,
                                      synth_constants=synth_constants, device=device)

        self.control_spec_loss = ControlSpectralLoss(loss_preset=train_cfg.loss.control_spec_preset,
                                                     synth_constants=synth_constants,
                                                     device=device)

        self.params_loss = ParametersLoss(loss_norm=train_cfg.loss.parameters_loss_norm,
                                          synth_constants=synth_constants, ignore_params=self.ignore_params,
                                          device=device)

        self.is_running_avg_parameters_loss_initialized = False
        self.is_running_avg_spectrogram_loss_initialized = False
        self.is_control_spectral_loss_initialized = False
        self.running_avg_parameters_loss = torch.tensor(1)
        self.running_avg_spectrogram_loss = torch.tensor(1)
        self.running_avg_control_spectral_loss = torch.tensor(1)

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

        spectrograms = self._preprocess_signal(raw_signal)

        # Run NN encoder model and get predicted parameters
        model_output_dict = self.synth_net(spectrograms)

        return model_output_dict

    def generate_synth_sound(self, full_range_synth_params: dict, batch_size: int) -> Tuple[torch.Tensor, dict]:

        # Inject synth parameters to the modular synth
        self.synth.update_cells_from_dict(full_range_synth_params)

        # Generate sound
        pred_final_signal, pred_signals_through_chain = \
            self.synth.generate_signal(signal_duration=self.cfg.synth.signal_duration, batch_size=batch_size)

        return pred_final_signal, pred_signals_through_chain

    def in_domain_step(self, batch, log: bool = False, return_metrics=False):
        """
        Processes a given batch to compute the model's losses and, optionally, other metrics.

        This function computes the parameter and spectrogram losses, balances and weights them
        based on configuration settings, and can optionally return additional evaluation metrics.
        If `tuning_mode` is enabled, the LSD (log-spectral distance) between the target and predicted signals
        is also computed.

        Parameters:
        - batch (tuple): Contains the target signal, target parameters in full range, and the signal index.
        - log (bool, optional): If True, specific information is logged. Defaults to False.
        - return_metrics (bool, optional): If True, returns additional evaluation metrics. Defaults to False.

        Returns:
        - loss_total - torch.Tensor: The total computed loss for the batch.
        - step_losses - dict:
            Dictionary containing detailed loss values (e.g., raw and weighted parameter/spectrogram losses).
        - step_artifacts - dict (optional):
            Dictionary containing various artifacts, such as predicted parameters and differences.
        - step_metrics - dict (optional, if return_metrics=True): Dictionary containing additional evaluation metrics.

        Notes:
        - If the current epoch is before `self.cfg.loss.spectrogram_loss_warmup_epochs`, the spectrogram loss
          used in the total loss calculation is set to zero.
        - The `return_metrics` flag affects the return type and the computations done. If it's set to True,
          additional metrics are computed, and the function's return type will include the metrics dictionary.
        """

        target_signal, target_params_full_range, signal_index = batch
        batch_size = len(signal_index)

        model_output_dict = self(target_signal)
        predicted_params_unit_range = self.normalizer.post_process_inherent_constraints(model_output_dict)
        predicted_params_full_range = self.normalizer.denormalize(predicted_params_unit_range)

        target_params_unit_range = self.normalizer.normalize(target_params_full_range)

        if self.ignore_params is not None:
            predicted_params_unit_range = self._update_param_dict(target_params_unit_range, predicted_params_unit_range)
            predicted_params_full_range = self._update_param_dict(target_params_full_range, predicted_params_full_range)

        total_params_loss, per_parameter_loss = self.params_loss.call(predicted_params_unit_range,
                                                                      target_params_unit_range)

        pred_final_signal, spec_loss_for_total, spec_loss_for_logging = (
            self._calculate_spec_loss(target_signal,
                                      target_params_full_range,
                                      predicted_params_full_range,
                                      batch_size,
                                      return_metrics))

        if self.training:
            if not self.is_running_avg_parameters_loss_initialized and total_params_loss.item() > 0:
                self._initialize_parameters_loss_running_avg(total_params_loss)

            if not self.is_running_avg_spectrogram_loss_initialized and spec_loss_for_total.item() > 0:
                self._initialize_spectrogram_loss_running_avg(spec_loss_for_total, total_params_loss)

            self._update_running_averages(total_params_loss, spec_loss_for_total, log)

        balanced_parameters_loss, balanced_spectrogram_loss = self._balance_losses(total_params_loss,
                                                                                   spec_loss_for_total, log)

        loss_total = balanced_parameters_loss + balanced_spectrogram_loss

        param_diffs, active_only_diffs = get_param_diffs(predicted_params_full_range.copy(),
                                                         target_params_full_range.copy(), self.ignore_params)

        step_losses = {
            'raw_params_loss': total_params_loss.detach(),
            'raw_spec_loss': spec_loss_for_logging.detach(),
            'weighted_params_loss': balanced_parameters_loss.detach(),
            'weighted_spec_loss': balanced_spectrogram_loss.detach(),
            'loss_total': loss_total.item()
        }

        step_artifacts = {
            'raw_predicted_parameters': predicted_params_unit_range,
            'full_range_predicted_parameters': predicted_params_full_range,
            'param_diffs': param_diffs,
            'active_only_diffs': active_only_diffs
        }

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

        model_output_dict = self(target_final_signal)
        predicted_params_unit_range = self.normalizer.post_process_inherent_constraints(model_output_dict)
        predicted_params_full_range = self.normalizer.denormalize(predicted_params_unit_range)

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
        if self.current_epoch != self.trainer.current_epoch:
            print(f"Current epoch: {self.current_epoch}, trainer epoch: {self.trainer.current_epoch}")
            exit()

        if batch_idx == 0:
            target_params = batch[1] if len(batch) == 3 else None
            self._log_sounds_batch(batch[0], target_params, f'samples_train')

        if self.cfg.loss.in_domain_epochs < self.current_epoch + 1:
            print("Running over out of domain dataset")
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
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
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

    # Gradient check
    # def on_after_backward(self):
    #     # Check the gradients
    #     for name, params in self.named_parameters():
    #         if params.grad is not None:
    #             grad_norm = params.grad.data.norm(2).item()
    #             if grad_norm == 0 or np.isnan(grad_norm):
    #                 print(f"Gradient for {name} has vanished.")
    #             if grad_norm > 1e+5:
    #                 print(f"Gradient for {name} is exploding.")

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint['is_parameters_loss_initialized'] = self.is_running_avg_parameters_loss_initialized
        checkpoint['running_avg_parameters_loss'] = self.running_avg_parameters_loss
        checkpoint['is_spectrogram_loss_initialized'] = self.is_running_avg_spectrogram_loss_initialized
        checkpoint['running_avg_spectrogram_loss'] = self.running_avg_spectrogram_loss

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self.is_running_avg_parameters_loss_initialized = checkpoint['is_parameters_loss_initialized']
        self.running_avg_parameters_loss = checkpoint['running_avg_parameters_loss']
        self.is_running_avg_spectrogram_loss_initialized = checkpoint['is_spectrogram_loss_initialized']
        self.running_avg_spectrogram_loss = checkpoint['running_avg_spectrogram_loss']

    def _preprocess_signal(self, raw_signal: torch.Tensor) -> torch.Tensor:

        # Add channel dimension
        raw_signal = torch.unsqueeze(raw_signal, 1)

        # Transform raw signal to spectrogram
        if self.use_multi_spec_input:
            spectrograms = self.multi_spec_transform.call(raw_signal)
        else:
            spectrograms = self.signal_transform(raw_signal)

        # Apply log transform if required
        if self.apply_log_transform_to_input:
            spectrograms = torch.log(spectrograms + 1e-8)

        return spectrograms

    def _calculate_spec_loss(self, target_signal, target_params_full_range, predicted_params_full_range, batch_size,
                             return_metrics):
        """Calculates the spectrogram loss."""

        # Initial assumption that we don't need to compute the spec loss.
        pred_final_signal = None
        spec_loss = torch.tensor(0.0, dtype=torch.float32)
        spec_loss_for_total = torch.tensor(0.0, dtype=torch.float32)  # Set this to zero initially

        # Only calculate the spec loss if it's time to do so or if metrics are requested
        if self.current_epoch >= self.cfg.loss.spectrogram_loss_warmup_epochs or return_metrics:
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

            if self.current_epoch >= self.cfg.loss.spectrogram_loss_warmup_epochs:
                spec_loss_for_total = spec_loss

        return pred_final_signal, spec_loss_for_total, spec_loss,

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
                layer_warmup_factor = self.cfg.loss.chain_warmup_factor * current_layer + self.cfg.loss.spectrogram_loss_warmup_epochs

                if self.current_epoch < layer_warmup_factor:
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

    def _initialize_parameters_loss_running_avg(self, parameters_loss):
        self.running_avg_parameters_loss = parameters_loss.clone().detach()
        self.is_running_avg_parameters_loss_initialized = True

    def _initialize_spectrogram_loss_running_avg(self, spectrogram_loss, parameters_loss):
        current_normalized_parameter_loss = parameters_loss.clone().detach() / self.running_avg_parameters_loss.clone().detach()
        self.running_avg_spectrogram_loss = spectrogram_loss.clone().detach() / current_normalized_parameter_loss
        self.is_running_avg_spectrogram_loss_initialized = True

    def _update_running_averages(self, parameters_loss, spectrogram_loss, log, alpha=0.99):
        """Update the running averages of the losses."""
        if self.is_running_avg_parameters_loss_initialized:
            new_params_loss_avg = alpha * self.running_avg_parameters_loss + (1 - alpha) * parameters_loss
            self.running_avg_parameters_loss = new_params_loss_avg.detach().clone()

        if self.is_running_avg_spectrogram_loss_initialized:
            new_spec_loss_avg = alpha * self.running_avg_spectrogram_loss + (1 - alpha) * spectrogram_loss
            self.running_avg_parameters_loss = new_spec_loss_avg.detach().clone()

        if log:
            self.tb_logger.add_scalar('normalized_losses/running_avg_parameters_loss', self.running_avg_parameters_loss,
                                      self.global_step)
            self.tb_logger.add_scalar('normalized_losses/running_avg_spectrogram_loss',
                                      self.running_avg_spectrogram_loss, self.global_step)

    def _normalize_losses(self, parameters_loss, spectrogram_loss, log):
        """
        Normalize the provided losses using maintained running averages.

        Args:
            parameters_loss (torch.Tensor): The current value of the parameters loss.
            spectrogram_loss (torch.Tensor): The current value of the spectrogram loss.

        Returns:
            tuple: A tuple containing normalized parameters loss and normalized spectrogram loss.
        """
        normalized_parameters_loss = parameters_loss / self.running_avg_parameters_loss
        normalized_spectrogram_loss = spectrogram_loss / self.running_avg_spectrogram_loss

        if log:
            self.tb_logger.add_scalar('normalized_losses/pre_rampup_normalized_parameters_loss',
                                      normalized_parameters_loss,
                                      self.global_step)
            self.tb_logger.add_scalar('normalized_losses/pre_rampup_normalized_spectrogram_loss',
                                      normalized_spectrogram_loss,
                                      self.global_step)

        return normalized_parameters_loss, normalized_spectrogram_loss

    def _apply_ramp_up(self, normalized_parameters_loss, normalized_spectrogram_loss, log=False):
        """
        Adjust the provided normalized losses using a ramp-up strategy.

        Args:
            normalized_parameters_loss (torch.Tensor): The normalized value of the parameters loss.
            normalized_spectrogram_loss (torch.Tensor): The normalized value of the spectrogram loss.
            log (bool): Whether to log specific values to a logger. Default is False.

        Returns:
            tuple: A tuple containing total loss, weighted parameters loss, and weighted spectrogram loss.
        """
        step = self.current_epoch
        cfg = self.cfg.loss

        parameters_loss_decay_factor = cfg.min_parameters_loss_decay
        spec_loss_rampup_factor = 1

        if step < cfg.spectrogram_loss_warmup_epochs:
            parameters_loss_decay_factor = 1.0
            spec_loss_rampup_factor = 0
        elif step < (cfg.spectrogram_loss_warmup_epochs + cfg.loss_switch_epochs):
            linear_mix_factor = (step - cfg.spectrogram_loss_warmup_epochs) / cfg.loss_switch_epochs
            parameters_loss_decay_factor = max(1 - linear_mix_factor, cfg.min_parameters_loss_decay)
            spec_loss_rampup_factor = linear_mix_factor

        weighted_params_loss = normalized_parameters_loss * parameters_loss_decay_factor
        weighted_spec_loss = normalized_spectrogram_loss * spec_loss_rampup_factor

        if log:
            self.tb_logger.add_scalar('loss/parameters_decay_factor', parameters_loss_decay_factor, self.global_step)
            self.tb_logger.add_scalar('loss/spec_loss_rampup_factor', spec_loss_rampup_factor, self.global_step)
            if log:
                self.tb_logger.add_scalar('normalized_losses/post_rampup_weighted_params_loss',
                                          weighted_params_loss,
                                          self.global_step)
                self.tb_logger.add_scalar('normalized_losses/post_rampup_weighted_spec_loss',
                                          weighted_spec_loss,
                                          self.global_step)

        return weighted_params_loss, weighted_spec_loss

    def _balance_losses(self, parameters_loss, spectrogram_loss, log=False):
        """
        Balance the given losses using normalization ramp-up and configured factorization strategies.

        Args:
            parameters_loss (torch.Tensor): The current value of the parameters loss.
            spectrogram_loss (torch.Tensor): The current value of the spectrogram loss.
            log (bool): Whether to log specific values to a logger. Default is False.

        Returns:
            tuple: A tuple containing total loss, weighted parameters loss, and weighted spectrogram loss.
        """
        params_loss_factor = self.cfg.loss.parameters_loss_weight
        spec_loss_factor = self.cfg.loss.spectrogram_loss_weight

        if self.cfg.loss.apply_loss_normalization:
            parameters_loss, spectrogram_loss = self._normalize_losses(parameters_loss,
                                                                       spectrogram_loss,
                                                                       log)
            params_loss_factor = 1.0
            spec_loss_factor = 1.0

        post_rampup_params_loss, post_rampup_spec_loss = self._apply_ramp_up(parameters_loss,
                                                                             spectrogram_loss,
                                                                             log)

        balanced_parameters_loss = params_loss_factor * post_rampup_params_loss
        balanced_spectrogram_loss = spec_loss_factor * post_rampup_spec_loss

        return balanced_parameters_loss, balanced_spectrogram_loss

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

        model_output_dict = self(target_signals)
        predicted_params_unit_range = self.normalizer.post_process_inherent_constraints(model_output_dict)
        predicted_params_full_range = self.normalizer.denormalize(predicted_params_unit_range)

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
        if isinstance(items_to_log, float) or isinstance(items_to_log, int):
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
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
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
                optimizer, base_lr=self.lr, max_lr=self.cfg.model.optimizer.max_lr,
                step_size_up=self.cfg.model.optimizer.cyclic_step_size_up),
                "interval": "step"}
        elif optimizer_params.scheduler.lower() == 'exponential':
            scheduler_config = {"scheduler": torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=self.cfg.model.optimizer.gamma),
                "interval": "epoch"}
        else:
            raise NotImplementedError(f"Scheduler {self.optimizer_params['scheduler']} not implemented")

        return {"optimizer": optimizer, "lr_scheduler": scheduler_config}
