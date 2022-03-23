import os.path

import torch
from torch import nn

import numpy as np

from torch.utils.tensorboard import SummaryWriter

from config import Config, SynthConfig, DatasetConfig, ModelConfig, configure_experiment
from ai_synth_dataset import AiSynthDataset, create_data_loader
from model import BigSynthNetwork
from synth.synth_architecture import SynthModular, SynthModularCell
import helper
import time
from pathlib import Path
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from tqdm import tqdm


def train_single_epoch(model,
                       epoch,
                       data_loader,
                       transform,
                       optimizer,
                       scheduler,
                       device,
                       modular_synth,
                       normalizer,
                       synth_cfg,
                       cfg,
                       activations_dict: dict,
                       summary_writer: SummaryWriter):
    sum_epoch_loss = 0
    num_of_mini_batches = 0
    with tqdm(data_loader, unit="batch") as tepoch:
        for target_signal, target_param_dic, signal_index in tepoch:

            step = epoch * len(data_loader) + num_of_mini_batches

            tepoch.set_description(f"Epoch {epoch}")
            batch_start_time = time.time()

            target_signal = target_signal.to(device)

            # set_to_none as advised in page 6:
            # https://tigress-web.princeton.edu/~jdh4/PyTorchPerformanceTuningGuide_GTC2021.pdf
            model.zero_grad(set_to_none=True)
            transformed_signal = transform(target_signal)

            # Log model to tensorboard
            # if epoch == 0 and num_of_mini_batches == 0:
            #     summary_writer.add_graph(model, transformed_signal)

            # -------------------------------------
            # -----------Run Model-----------------
            # -------------------------------------
            model_start_time = time.time()

            output_dic = model(transformed_signal)
            log_dict_recursive('raw_output', output_dic, summary_writer, step)

            model_end_time = time.time()
            synth_start_time = time.time()

            denormalized_output_dict = normalizer.denormalize(output_dic)
            log_dict_recursive('denormalized_output', denormalized_output_dict, summary_writer, step)

            predicted_param_dict = helper.clamp_regression_params(denormalized_output_dict, synth_cfg, cfg)
            log_dict_recursive('clamped_output', predicted_param_dict, summary_writer, step)

            update_params = []
            for index, operation_dict in predicted_param_dict.items():
                synth_modular_cell = SynthModularCell(index=index, parameters=operation_dict['params'])
                update_params.append(synth_modular_cell)

            modular_synth.update_cells(update_params)
            modular_synth.generate_signal(num_sounds=len(transformed_signal))

            modular_synth.signal = helper.move_to(modular_synth.signal, device)

            synth_end_time = time.time()
            loss_start_time = time.time()

            if cfg.spectrogram_loss_type == 'MULTI-SPECTRAL':
                multi_spec_loss = helper.SpectralLoss(cfg=cfg,
                                                      loss_type=cfg.multi_spectral_loss_type,
                                                      mag_weight=cfg.multi_spectral_mag_weight,
                                                      delta_time_weight=cfg.multi_spectral_delta_time_weight,
                                                      delta_freq_weight=cfg.multi_spectral_delta_freq_weight,
                                                      cumsum_freq_weight=cfg.multi_spectral_cumsum_freq_weight,
                                                      logmag_weight=cfg.multi_spectral_logmag_weight,
                                                      device=device)
                target_signal = target_signal.squeeze()
                loss = multi_spec_loss.call(target_signal, modular_synth.signal, summary_writer, step)
                summary_writer.add_scalar('loss/train_multi_spectral', loss, step)
            else:
                ValueError("SYNTH_TYPE 'MODULAR' supports only SPECTROGRAM_LOSS_TYPE of type 'MULTI-SPECTRAL'")

            loss_end_time = time.time()
            backward_start_time = time.time()

            num_of_mini_batches += 1
            sum_epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()

            summary_writer.add_scalar('lr_adam', optimizer.param_groups[0]['lr'], step)
            log_gradients_in_model(model, summary_writer, step)

            backward_end_time = time.time()
            batch_end_time = time.time()

            if cfg.print_train_batch_stats:
                print(f"MSE batch loss: {round(loss.item(), 7)},\n")
                if cfg.print_timings:
                    print(
                        f"total batch processing time: {round(batch_end_time - batch_start_time, 2)}s, \n"
                        f"model processing time: {round(model_end_time - model_start_time, 2)}s, \t"
                        f"synth processing time: {round(synth_end_time - synth_start_time, 2)}s, \t"
                        f"backward processing time: {round(backward_end_time - backward_start_time, 2)}s, \t"
                        f"loss processing time: {round(loss_end_time - loss_start_time, 2)}s\n")
                if cfg.print_synth_param_stats:
                    helper.print_synth_param_stats(predicted_param_dict, target_param_dic, synth_cfg, device)

            tepoch.set_postfix(loss=loss.item())

    avg_epoch_loss = sum_epoch_loss / num_of_mini_batches
    summary_writer.add_scalar('loss/train_multi_spectral_epoch', avg_epoch_loss, epoch)

    return avg_epoch_loss


def train(model,
          data_loader,
          transform,
          optimizer,
          device,
          start_epoch: int,
          num_epochs: int,
          cfg: Config,
          model_cfg: ModelConfig,
          synth_cfg: SynthConfig,
          activations_dict: dict,
          summary_writer: SummaryWriter):

    # Initializations
    model.train()
    torch.autograd.set_detect_anomaly(True)

    loss_list = []

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=num_epochs * len(data_loader))
    normalizer = helper.Normalizer(cfg.signal_duration_sec, synth_cfg)

    # init modular synth
    modular_synth = SynthModular(synth_cfg=synth_cfg,
                                 sample_rate=cfg.sample_rate,
                                 signal_duration_sec=cfg.signal_duration_sec,
                                 num_sounds=1,
                                 device=device,
                                 preset=synth_cfg.preset
                                 )

    for epoch in range(num_epochs):
        cur_epoch = start_epoch + epoch
        avg_epoch_loss = \
            train_single_epoch(model=model,
                               epoch=cur_epoch,
                               data_loader=data_loader,
                               transform=transform,
                               optimizer=optimizer,
                               scheduler=scheduler,
                               device=device,
                               modular_synth=modular_synth,
                               normalizer=normalizer,
                               synth_cfg=synth_cfg,
                               cfg=cfg,
                               activations_dict=activations_dict,
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


def log_gradients_in_model(model, logger, step):
    for tag, value in model.named_parameters():
        if value.grad is not None:
            grad_val = value.grad.cpu()
            logger.add_histogram(tag + "/grad", grad_val, step)
            if np.linalg.norm(grad_val) < 1e-4 and 'bias' not in tag:
                print(f"Op {tag} gradient approaching 0")


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


def run(exp_name: str, dataset_name: str):

    cfg, model_cfg, synth_cfg, dataset_cfg = configure_experiment(exp_name, dataset_name)
    summary_writer = SummaryWriter(cfg.tensorboard_logdir)

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            description='Train AI Synth')
    parser.add_argument('-g', '--gpu_index', help='index of gpu (if exist, torch indexing) -1 for cpu',
                        type=int, default=0)
    parser.add_argument('-t', '--transform', choices=['mel', 'spec'],
                        help='mel: Mel Spectrogram, spec: Spectrogram', default='mel')

    args = parser.parse_args()
    device = helper.get_device(args.gpu_index)

    transforms = {'mel': helper.mel_spectrogram_transform(cfg.sample_rate).to(device),
                  'spec': helper.spectrogram_transform().to(device)}
    transform = transforms[args.transform]

    ai_synth_dataset = AiSynthDataset(dataset_cfg.train_parameters_file, dataset_cfg.train_audio_dir, device)
    train_dataloader = create_data_loader(ai_synth_dataset, model_cfg.batch_size, ModelConfig.num_workers)

    # construct model and assign it to device
    synth_net = BigSynthNetwork(synth_cfg, device).to(device)

    activations_dict = {}
    # for name, layer in synth_net.named_modules():
    #     layer.register_forward_hook(get_activation(name, activations_dict))

    optimizer = torch.optim.Adam(synth_net.parameters(),
                                 lr=ModelConfig.learning_rate,
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
          data_loader=train_dataloader,
          transform=transform,
          optimizer=optimizer,
          device=device,
          start_epoch=cur_epoch,
          num_epochs=model_cfg.num_epochs,
          cfg=cfg,
          model_cfg=model_cfg,
          synth_cfg=synth_cfg,
          activations_dict=activations_dict,
          summary_writer=summary_writer)

    # save model
    torch.save(synth_net.state_dict(), cfg.save_model_path)
    print("Final trained synth net saved at trained_synth_net.pt")


if __name__ == "__main__":
    run('test', 'toy_data')
