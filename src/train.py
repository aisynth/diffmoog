import os.path
from collections import defaultdict

import torch
from torch import nn

import numpy as np

from torch.utils.tensorboard import SummaryWriter

from config import Config, SynthConfig, DatasetConfig, ModelConfig, configure_experiment
from ai_synth_dataset import AiSynthDataset, create_data_loader
from inference import visualize_signal_prediction
from model import BigSynthNetwork, SimpleSynthNetwork, DecoderOnlyNetwork
from synth.synth_architecture import SynthModular, SynthModularCell
import helper
import time
from pathlib import Path
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from tqdm import tqdm

from train_helper import *


def train_single_epoch(model,
                       epoch,
                       data_loader,
                       dataset,
                       transform,
                       optimizer,
                       scheduler,
                       loss_handler,
                       device,
                       modular_synth,
                       normalizer,
                       synth_cfg,
                       cfg,
                       summary_writer: SummaryWriter):

    sum_epoch_loss = 0
    num_of_mini_batches = 0
    data_loading_time_start = time.time()
    epoch_param_diffs = defaultdict(list)
    epoch_param_vals_raw, epoch_param_vals = defaultdict(list), defaultdict(list)

    # with tqdm(data_loader, unit="batch") as tepoch:
    #     for target_signal, target_param_dict, signal_index in tepoch:

    target_signal, target_param_dict, signal_index = dataset[0]
    if epoch == 0:
        print(target_param_dict)

    for cell_idx, cell_params in target_param_dict.items():
        if cell_params['operation'] == 'None':
            continue
        for param_name, param_val in cell_params['parameters'].items():
            if isinstance(param_val, str):
                continue
            target_param_dict[cell_idx]['parameters'][param_name] = torch.tensor(param_val)

    data_loading_time_end = time.time()
    step = epoch * len(data_loader) + num_of_mini_batches

    # tepoch.set_description(f"Epoch {epoch}")
    batch_start_time = time.time()

    # set_to_none as advised in page 6:
    # https://tigress-web.princeton.edu/~jdh4/PyTorchPerformanceTuningGuide_GTC2021.pdf
    model.zero_grad(set_to_none=True)

    input_transform_start = time.time()
    target_signal = target_signal.to(device)
    transformed_signal = transform(target_signal)
    input_transform_end = time.time()

    # Log model to tensorboard
    # if epoch == 0 and num_of_mini_batches == 0:
    #     summary_writer.add_graph(model, transformed_signal)

    # -------------------------------------
    # -----------Run Model-----------------
    # -------------------------------------
    model_start_time = time.time()

    output_dic = model()

    for op_idx, op_dict in output_dic.items():
        for param_name, param_vals in op_dict['params'].items():
            epoch_param_vals_raw[f'{op_idx}_{param_name}'].append(param_vals.cpu().detach().numpy().squeeze())

    denormalized_output_dict = normalizer.denormalize(output_dic)
    predicted_param_dict = helper.clamp_adsr_params(denormalized_output_dict, synth_cfg, cfg)

    # predicted_param_dict = output_dic

    for op_idx, op_dict in predicted_param_dict.items():
        for param_name, param_vals in op_dict['params'].items():
            epoch_param_vals[f'{op_idx}_{param_name}'].append(param_vals.cpu().detach().numpy().squeeze())

    batch_param_diffs = get_param_diffs(predicted_param_dict, target_param_dict)
    for op_idx, diff_vals in batch_param_diffs.items():
        epoch_param_diffs[op_idx].append(diff_vals)

    model_end_time = time.time()
    synth_start_time = time.time()

    # ------------------------------------------------------------
    # -------------Generate Signal-------------------------------
    # ------------------------------------------------------------
    # --------------Predicted-------------------------------------
    update_params = []
    for index, operation_dict in predicted_param_dict.items():
        synth_modular_cell = SynthModularCell(index=index, parameters=operation_dict['params'])
        update_params.append(synth_modular_cell)

    modular_synth.update_cells(update_params)

    pred_final_signal, pred_signals_through_chain = \
        modular_synth.generate_signal(num_sounds=len(transformed_signal))

    # --------------Target-------------------------------------
    update_params = []
    for op_index in output_dic.keys():
        operation_dict = target_param_dict[op_index]
        synth_modular_cell = SynthModularCell(index=op_index, parameters=operation_dict['parameters'])
        update_params.append(synth_modular_cell)

    modular_synth.update_cells(update_params)
    target_final_signal, target_signals_through_chain = \
        modular_synth.generate_signal(num_sounds=len(transformed_signal))

    modular_synth.signal = helper.move_to(modular_synth.signal, device)

    synth_end_time = time.time()
    loss_start_time = time.time()

    target_signal = target_signal.squeeze()

    loss_total = 0
    for op_index in output_dic.keys():
        op_index = str(op_index)

        pred_signal = pred_signals_through_chain[op_index]
        if pred_signal is None:
            continue

        target_signal = target_signals_through_chain[op_index]
        loss, ret_spectrograms = loss_handler.call(target_signal,
                                                   pred_signal,
                                                   summary_writer,
                                                   op_index,
                                                   step,
                                                   return_spectrogram=True)
        loss_total += loss
    summary_writer.add_scalar('loss/train_multi_spectral', loss_total, step)

    loss_end_time = time.time()

    if epoch % 100 == 1:
        for i in range(1):
            sample_params_orig, sample_params_pred = parse_synth_params(target_param_dict, predicted_param_dict, i)
            summary_writer.add_audio(f'input_{i}_target', target_final_signal[i], global_step=epoch,
                                     sample_rate=cfg.sample_rate)
            summary_writer.add_audio(f'input_{i}_pred', pred_final_signal[i], global_step=epoch,
                                     sample_rate=cfg.sample_rate)

            signal_vis = visualize_signal_prediction(target_final_signal[i], pred_final_signal[i],
                                                     sample_params_orig, sample_params_pred, db=True)
            signal_vis_t = torch.tensor(signal_vis, dtype=torch.uint8, requires_grad=False)

            summary_writer.add_image(f'{256}_spec/input_{i}',
                                     signal_vis_t,
                                     global_step=epoch,
                                     dataformats='HWC')

    backward_start_time = time.time()

    num_of_mini_batches += 1
    sum_epoch_loss += loss_total.item()
    loss_total.backward()
    optimizer.step()
    scheduler.step()

    backward_end_time = time.time()

    summary_writer.add_scalar('lr_adam', optimizer.param_groups[0]['lr'], step)

    if num_of_mini_batches % 100 == 0:
        log_gradients_in_model(model, summary_writer, step)

    batch_end_time = time.time()

    if cfg.print_train_batch_stats:
        print(f"MSE batch loss: {round(loss_total.item(), 7)},\n")
        if cfg.print_timings:
            print(
                f"total batch processing time: {round(batch_end_time - batch_start_time, 2)}s, \n"
                f"model processing time: {round(model_end_time - model_start_time, 2)}s, \n"
                f"synth processing time: {round(synth_end_time - synth_start_time, 2)}s, \n"
                f"backward processing time: {round(backward_end_time - backward_start_time, 2)}s, \n"
                f"loss processing time: {round(loss_end_time - loss_start_time, 2)}s\n"
                f"input transform time: {round(input_transform_end - input_transform_start, 2)}s\n"
                f"data loading time: {round(data_loading_time_end - data_loading_time_start, 2)}s\n"
                f"gradient logging time: {round(batch_end_time - backward_end_time, 2)}s\n")

            if cfg.print_synth_param_stats:
                helper.print_synth_param_stats(predicted_param_dict, target_param_dict, synth_cfg, device)

    # tepoch.set_postfix(loss=loss_total.item())
    data_loading_time_start = time.time()

    avg_epoch_loss = sum_epoch_loss / num_of_mini_batches
    summary_writer.add_scalar('loss/train_multi_spectral_epoch', avg_epoch_loss, epoch)
    log_dict_recursive('param_diff', epoch_param_diffs, summary_writer, epoch)
    log_dict_recursive('param_values_raw', epoch_param_vals_raw, summary_writer, epoch)
    log_dict_recursive('param_values_normalized', epoch_param_vals, summary_writer, epoch)

    return avg_epoch_loss


def train(model,
          modular_synth,
          data_loader,
          dataset,
          transform,
          optimizer,
          device,
          start_epoch: int,
          num_epochs: int,
          cfg: Config,
          model_cfg: ModelConfig,
          synth_cfg: SynthConfig,
          summary_writer: SummaryWriter):

    # Initializations
    model.train()
    torch.autograd.set_detect_anomaly(True)

    loss_list = []

    # scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer=optimizer, factor=1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=num_epochs * len(data_loader))
    # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, model_cfg.learning_rate / 10, model_cfg.learning_rate,
    #                                               mode='triangular2', cycle_momentum=False)
    normalizer = helper.Normalizer(cfg.signal_duration_sec, synth_cfg)

    if cfg.spectrogram_loss_type == 'MULTI-SPECTRAL':
        loss_handler = helper.SpectralLoss(cfg=cfg,
                                           fft_sizes=cfg.fft_sizes,
                                           loss_type=cfg.multi_spectral_loss_type,
                                           mag_weight=cfg.multi_spectral_mag_weight,
                                           delta_time_weight=cfg.multi_spectral_delta_time_weight,
                                           delta_freq_weight=cfg.multi_spectral_delta_freq_weight,
                                           cumsum_freq_weight=cfg.multi_spectral_cumsum_freq_weight,
                                           cumsum_time_weight=cfg.multi_spectral_cumsum_time_weight,
                                           logmag_weight=cfg.multi_spectral_logmag_weight,
                                           normalize_by_size=cfg.normalize_loss_by_nfft,
                                           device=device)
    else:
        raise ValueError("SYNTH_TYPE 'MODULAR' supports only SPECTROGRAM_LOSS_TYPE of type 'MULTI-SPECTRAL'")

    for epoch in range(num_epochs):
        cur_epoch = start_epoch + epoch
        avg_epoch_loss = \
            train_single_epoch(model=model,
                               epoch=cur_epoch,
                               data_loader=data_loader,
                               dataset=dataset,
                               transform=transform,
                               optimizer=optimizer,
                               scheduler=scheduler,
                               device=device,
                               modular_synth=modular_synth,
                               normalizer=normalizer,
                               loss_handler=loss_handler,
                               synth_cfg=synth_cfg,
                               cfg=cfg,
                               summary_writer=summary_writer)

        # Sum stats over multiple epochs
        loss_list.append(avg_epoch_loss)

        # save model checkpoint
        if cur_epoch % cfg.num_epochs_to_save_model == 0:
            ckpt_path = os.path.join(cfg.ckpts_dir, f'synth_net_epoch{cur_epoch}.pt')
            helper.save_model(cur_epoch,
                              model,
                              optimizer,
                              avg_epoch_loss,
                              loss_list,
                              ckpt_path,
                              cfg.txt_path,
                              cfg.numpy_path)

    print("Finished training")


def run(args):
    exp_name = args.experiment
    dataset_name = args.dataset
    cfg, model_cfg, synth_cfg, dataset_cfg = configure_experiment(exp_name, dataset_name)
    summary_writer = SummaryWriter(cfg.tensorboard_logdir)

    device = helper.get_device(args.gpu_index)

    transforms = {'mel': helper.mel_spectrogram_transform(cfg.sample_rate).to(device),
                  'spec': helper.spectrogram_transform().to(device)}
    transform = transforms[args.transform]

    ai_synth_dataset = AiSynthDataset(dataset_cfg.train_parameters_file, dataset_cfg.train_audio_dir, device)
    train_dataloader = create_data_loader(ai_synth_dataset, model_cfg.batch_size, ModelConfig.num_workers)

    # init modular synth
    modular_synth = SynthModular(synth_cfg=synth_cfg,
                                 sample_rate=cfg.sample_rate,
                                 signal_duration_sec=cfg.signal_duration_sec,
                                 num_sounds=1,
                                 device=device,
                                 preset=synth_cfg.preset)

    modular_synth.generate_random_params(synth_cfg)
    init_params = modular_synth.collect_params()

    # Denormalize init params
    normalizer = helper.Normalizer(signal_duration_sec=cfg.signal_duration_sec, synth_cfg=synth_cfg)
    denormalized_init_params = normalizer.normalize(init_params)

    # construct model and assign it to device
    if model_cfg.model_type == 'simple':
        synth_net = SimpleSynthNetwork(synth_cfg, device, backbone=model_cfg.backbone).to(device)
    elif model_cfg.model_type == 'decoder_only':
        synth_net = DecoderOnlyNetwork(synth_cfg, device)
        synth_net.apply_params(denormalized_init_params)
    else:
        synth_net = BigSynthNetwork(synth_cfg, device).to(device)

    # for name, layer in synth_net.named_modules():
    #     layer.register_forward_hook(get_activation(name, activations_dict))

    optimizer = torch.optim.SGD(synth_net.parameters(), lr=model_cfg.learning_rate,
                                 weight_decay=model_cfg.optimizer_weight_decay)

    print(f"Training model start")

    if cfg.use_loaded_model:
        print(f"Use Loaded model {cfg.load_model_path.name}")
        checkpoint = torch.load(cfg.load_model_path)
        synth_net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        cur_epoch = checkpoint['epoch'] + 1
        loss = checkpoint['loss']
    else:
        cur_epoch = 0

    # train model
    train(model=synth_net,
          modular_synth=modular_synth,
          data_loader=train_dataloader,
          dataset=ai_synth_dataset,
          transform=transform,
          optimizer=optimizer,
          device=device,
          start_epoch=cur_epoch,
          num_epochs=model_cfg.num_epochs,
          cfg=cfg,
          synth_cfg=synth_cfg,
          model_cfg=model_cfg,
          summary_writer=summary_writer)

    # save model
    torch.save(synth_net.state_dict(), cfg.save_model_path)
    print("Final trained synth net saved at trained_synth_net.pt")


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            description='Train AI Synth')
    parser.add_argument('-g', '--gpu_index', help='index of gpu (if exist, torch indexing) -1 for cpu',
                        type=int, default=0)
    parser.add_argument('-t', '--transform', choices=['mel', 'spec'],
                        help='mel: Mel Spectrogram, spec: Spectrogram', default='mel')
    parser.add_argument('-e', '--experiment', required=True,
                        help='Experiment name', type=str)
    parser.add_argument('-d', '--dataset', required=True, type=str,
                        help='Dataset name')

    args = parser.parse_args()
    run(args)
