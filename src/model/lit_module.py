import copy
from typing import Any, Tuple, Optional

import torch
import torchaudio
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT

from model.loss.parameters_loss import ParametersLoss
from model.loss.spectral_loss import SpectralLoss
from model.model import SynthNetwork
from synth.parameters_normalizer import Normalizer, clamp_adsr_params
from synth.synth_architecture import SynthModular
from synth.synth_constants import synth_structure
from utils.metrics import lsd, pearsonr_dist, mae, mfcc_distance, spectral_convergence
from utils.train_utils import log_dict_recursive, parse_synth_params, get_param_diffs
from utils.visualization_utils import visualize_signal_prediction


class LitModularSynth(LightningModule):

    def __init__(self, train_cfg, device: str = 'cuda:0'):

        super().__init__()

        self.cfg = train_cfg

        self.synth = SynthModular(preset_name=train_cfg.synth.preset, synth_structure=synth_structure, device=device)

        self.synth_net = SynthNetwork(train_cfg.model.preset, device, backbone=train_cfg.model.backbone).to(device)
        self.normalizer = Normalizer(train_cfg.synth.note_off_time, synth_structure)

        if train_cfg.transform.lower() == 'mel':
            self.signal_transform = torchaudio.transforms.MelSpectrogram(sample_rate=synth_structure.sample_rate,
                                                                         n_fft=1024, hop_length=256, n_mels=128,
                                                                         power=2.0, f_min=0, f_max=8000)
        elif train_cfg.transform.lower() == 'spec':
            self.signal_transform = torchaudio.transforms.Spectrogram(n_fft=512, power=2.0)
        else:
            raise NotImplementedError(f'Input transform {train_cfg.transform} not implemented.')

        self.spec_loss = SpectralLoss(loss_type=train_cfg.loss.spec_loss_type, preset_name=train_cfg.loss.preset,
                                      sample_rate=synth_structure.sample_rate, device=device)

        self.params_loss = ParametersLoss(loss_type=train_cfg.loss.parameters_loss_type,
                                          synth_structure=synth_structure, device=device)

        self.epoch_param_diffs, self.epoch_vals_raw, self.epoch_vals_normalized = {}, {}, {}

    def forward(self, raw_signal: torch.Tensor, *args, **kwargs) -> Any:

        assert len(raw_signal.shape) == 2, f'Expected tensor of dimensions [batch_size, signal_length] ' \
                                           f'but got shape {raw_signal.shape}'

        spectrograms = self.signal_transform(raw_signal)

        predicted_parameters_unit_range = self.synth_net(spectrograms)
        predicted_params_full_range = self.normalizer.denormalize(predicted_parameters_unit_range)
        predicted_params_full_range = clamp_adsr_params(predicted_params_full_range, synth_structure)

        return predicted_parameters_unit_range, predicted_params_full_range

    def generate_synth_sound(self, full_range_synth_params: dict, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        self.synth.update_cells_from_dict(full_range_synth_params)
        pred_final_signal, pred_signals_through_chain = \
            self.synth.generate_signal(signal_duration=self.train_cfg.synth.signal_duration, batch_size=batch_size)
        return pred_final_signal, pred_signals_through_chain

    def in_domain_step(self, batch, batch_idx, tag: str):
        if batch_idx == 0:
            self.log_sounds_batch(batch, f'samples_{tag}')

        raw_signal, target_params_full_range, signal_index = batch
        batch_size = len(signal_index)

        predicted_params_unit_range, predicted_params_full_range = self(raw_signal)
        target_params_unit_range = self.normalizer.normalize(target_params_full_range)

        total_params_loss, per_parameter_loss = self.params_loss.call(predicted_params_unit_range,
                                                                      target_params_unit_range)

        spec_loss, predicted_signal = self.calculate_spectrogram_loss(raw_signal, target_params_full_range,
                                                                      predicted_params_full_range,
                                                                      self.train_cfg.loss.use_chain_loss,
                                                                      batch_size, log=True)

        loss_total, weighted_params_loss, weighted_spec_loss = self.balance_losses(total_params_loss, spec_loss)
        param_diffs = get_param_diffs(predicted_params_full_range.copy(), target_params_full_range.copy())

        step_losses = {'raw_params_loss': total_params_loss, 'raw_spec_loss': spec_loss,
                       'weighted_params_loss': weighted_params_loss, 'weighted_spec_loss': weighted_spec_loss}

        step_metrics = self.calculate_audio_metrics(raw_signal, predicted_signal)

        step_artifacts = {'raw_predicted_parameters': predicted_params_unit_range,
                          'full_range_predicted_parameters': predicted_params_full_range, 'param_diffs': param_diffs}

        # Log step stats
        self._log_recursive(step_losses, f'losses/{tag}')
        self._log_recursive(step_metrics, f'metrics/{tag}')
        self._log_recursive(per_parameter_loss, f'per_param_losses/{tag}')

        return loss_total, step_artifacts

    def out_of_domain_step(self, batch, batch_idx, tag: str):

        target_final_signal, _, signal_index = batch
        batch_size = len(signal_index)

        predicted_params_unit_range, predicted_params_full_range = self(target_final_signal)
        pred_final_signal, pred_signals_through_chain = self.generate_synth_sound(predicted_params_full_range,
                                                                                  batch_size)

        loss, per_op_loss, per_op_weighted_loss, ret_spectrograms = \
            self.spec_loss.call(target_final_signal, pred_final_signal, return_spectrogram=True, step=self.global_step)

        weighted_spec_loss = self.cfg.loss.spectrogram_loss_weight * loss
        step_losses = {'raw_spec_loss': loss, 'weighted_spec_loss': weighted_spec_loss}

        step_metrics = self.calculate_audio_metrics(target_final_signal, pred_final_signal)

        step_artifacts = {'raw_predicted_parameters': predicted_params_unit_range,
                          'full_range_predicted_parameters': predicted_params_full_range}

        # Log step stats
        self._log_recursive(step_losses, f'losses/{tag}')
        self._log_recursive(step_metrics, f'metrics/{tag}')
        self._log_recursive(per_op_loss, f'spec_loss/{tag}')
        self._log_recursive(per_op_weighted_loss, f'spec_loss_weighted/{tag}')

        return weighted_spec_loss, step_artifacts

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:

        loss, step_artifacts = self.in_domain_step(batch, batch_idx, 'train')

        self._accumulate_batch_values(self.epoch_vals_raw, step_artifacts['raw_predicted_parameters'])
        self._accumulate_batch_values(self.epoch_vals_normalized, step_artifacts['full_range_predicted_parameters'])
        self._accumulate_batch_values(self.epoch_param_diffs, step_artifacts['param_diffs'])

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, dataloader_idx) -> Optional[STEP_OUTPUT]:
        val_name = 'in_domain_validation' if dataloader_idx == 0 else 'nsynth_validation'

        if batch_idx == 0:
            self.log_sounds_batch(batch, val_name)

        if 'in_domain' in val_name:
            loss, step_artifacts = self.in_domain_step(batch, batch_idx, val_name)
        else:
            loss, step_artifacts = self.out_of_domain_step(batch, batch_idx, val_name)

        return loss

    @torch.no_grad()
    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        target_final_signal, _, signal_index = batch
        batch_size = len(signal_index)

        predicted_params_unit_range, predicted_params_full_range = self(target_final_signal)
        pred_final_signal, pred_signals_through_chain = self.generate_synth_sound(predicted_params_full_range,
                                                                                  batch_size)

        return pred_final_signal, predicted_params_full_range

    def on_train_epoch_end(self) -> None:
        self._log_recursive(self.epoch_param_diffs, 'param_diff')
        self._log_recursive(self.epoch_vals_raw, 'param_values_raw')
        self._log_recursive(self.epoch_vals_normalized, 'param_values_normalized')

        self.epoch_param_diffs, self.epoch_vals_raw, self.epoch_vals_normalized = {}, {}, {}

        return

    def log_sounds_batch(self, batch, tag: str):
        raw_target_signal, target_params_full_range, signal_index = batch
        batch_size = len(signal_index)

        predicted_params_unit_range, predicted_params_full_range = self(raw_target_signal)
        pred_final_signal, pred_signals_through_chain = self.generate_synth_sound(predicted_params_full_range,
                                                                                  batch_size)

        for i in range(self.cfg.n_images_to_log):
            sample_params_orig, sample_params_pred = parse_synth_params(target_params_full_range,
                                                                        predicted_params_full_range, i)
            self.logger.experiment.add_audio(f'{tag}/input_{i}_target', raw_target_signal[i],
                                             global_step=self.current_epoch, sample_rate=synth_structure.sample_rate)
            self.logger.experiment.add_audio(f'{tag}/input_{i}_pred', pred_final_signal[i],
                                             global_step=self.current_epoch, sample_rate=synth_structure.sample_rate)

            signal_vis = visualize_signal_prediction(raw_target_signal[i], pred_final_signal[i], sample_params_orig,
                                                     sample_params_pred, db=True)
            signal_vis_t = torch.tensor(signal_vis, dtype=torch.uint8, requires_grad=False)

            self.logger.experiment.add_image(f'{tag}/{256}_spec/input_{i}',
                                             signal_vis_t,
                                             global_step=self.current_epoch,
                                             dataformats='HWC')

    def calculate_spectrogram_loss(self, target_signal: torch.Tensor, target_params: dict, predicted_params: dict,
                                   use_chain_loss: bool, batch_size: int = 1, log=False):

        params_for_pred_signal_generation = copy.copy(target_params)
        params_for_pred_signal_generation.update(predicted_params)
        pred_final_signal, pred_signals_through_chain = self.generate_synth_sound(params_for_pred_signal_generation,
                                                                                  batch_size)

        if use_chain_loss:
            target_final_signal, target_signals_through_chain = self.generate_synth_sound(target_params, batch_size)

            chain_losses = torch.zeros(len(target_signals_through_chain))
            for i, op_index in enumerate(predicted_params.keys()):
                op_index = str(op_index)

                # current_layer = int(op_index[2])
                # layer_warmup_factor = cfg.chain_warmup_factor * current_layer
                #
                # if epoch - cfg.spectrogram_loss_warmup / num_iters < layer_warmup_factor:
                #     continue

                c_pred_signal = pred_signals_through_chain[op_index]
                if c_pred_signal is None:
                    continue

                c_target_signal = target_signals_through_chain[op_index]
                loss, per_op_loss, per_op_weighted_loss, ret_spectrograms = \
                    self.spec_loss.call(c_target_signal, c_pred_signal, step=self.global_step, return_spectrogram=True)
                chain_losses[i] = loss

                if log:
                    self._log_recursive(per_op_weighted_loss, f'{op_index}_weighted')

            spectrogram_loss = chain_losses.mean() / self.train_cfg.loss.chain_loss_weight
        else:
            loss, per_op_loss, per_op_weighted_loss, ret_spectrograms = \
                self.spec_loss.call(target_signal, pred_final_signal, return_spectrogram=True, step=self.global_step)
            spectrogram_loss = loss

            if log:
                self._log_recursive(per_op_weighted_loss, f'final_spec_losses_weighted')

        return spectrogram_loss, pred_final_signal

    def balance_losses(self, parameters_loss, spectrogram_loss):

        step = self.global_step
        cfg = self.cfg.loss

        if step < cfg.spectrogram_loss_warmup:
            weighted_params_loss = parameters_loss * cfg.parameters_loss_weight
            weighted_spec_loss = 0
        elif step < (cfg.spectrogram_loss_warmup + cfg.loss_switch_steps):
            parameters_loss_decay_factor = 1 - ((step - cfg.spectrogram_loss_warmup) / cfg.loss_switch_steps)
            spec_loss_increase_factor = ((step - cfg.spectrogram_loss_warmup) / cfg.loss_switch_steps)

            parameters_loss_decay_factor = max(parameters_loss_decay_factor, cfg.min_parameters_loss_decay)
            weighted_params_loss = parameters_loss * cfg.parameters_loss_weight * parameters_loss_decay_factor
            weighted_spec_loss = cfg.spectrogram_loss_weight * spectrogram_loss * spec_loss_increase_factor
        else:
            parameters_loss_decay_factor = cfg.min_parameters_loss_decay
            weighted_params_loss = parameters_loss * cfg.parameters_loss_weight * parameters_loss_decay_factor
            weighted_spec_loss = cfg.spectrogram_loss_weight * spectrogram_loss

        loss_total = weighted_params_loss + weighted_spec_loss

        # summary_writer.add_scalar('loss/parameters_decay_factor', parameters_loss_decay_factor, step)
        # summary_writer.add_scalar('loss/spec_loss_rampup_factor', spec_loss_increase_factor, step)

        return loss_total, weighted_params_loss, weighted_spec_loss

    @torch.no_grad()
    def calculate_audio_metrics(self, target_signal: torch.Tensor, predicted_signal: torch.Tensor):
        metrics = {}

        target_spec = self.signal_transform(target_signal).cpu().numpy()
        predicted_spec = self.signal_transform(predicted_signal).cpu().numpy()

        target_signal = target_signal.cpu().numpy()
        predicted_signal = predicted_signal.cpu().numpy()

        metrics['lsd_value'] += lsd(target_spec, predicted_spec)
        metrics['pearson_stft'] += pearsonr_dist(target_spec, predicted_spec, input_type='spec')
        metrics['pearson_fft'] += pearsonr_dist(target_signal, predicted_signal, input_type='audio')
        metrics['mean_average_error'] += mae(target_spec, predicted_spec)
        metrics['mfcc_mae'] += mfcc_distance(target_signal, predicted_signal, sample_rate=synth_structure.sample_rate)
        metrics['spectral_convergence_value'] += spectral_convergence(target_spec, predicted_spec)

        return metrics

    def _log_recursive(self, items_to_log: dict, tag: str, on_epoch=False):
        interval = self.current_epoch if on_epoch else self.global_step
        log_dict_recursive(tag, items_to_log, self.logger.experiment, interval)

    @staticmethod
    def _accumulate_batch_values(accumulator: dict, batch_vals: dict):
        for op_idx, op_dict in batch_vals.items():
            for param_name, param_vals in op_dict['parameters'].items():
                accumulator[f'{op_idx}_{param_name}'].extend(param_vals.cpu().detach().numpy())
