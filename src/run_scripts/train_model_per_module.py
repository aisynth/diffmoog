import os.path
from collections import defaultdict

from config import Config, ModelConfig, configure_experiment
from dataset.ai_synth_dataset import AiSynthDataset, create_data_loader
from run_scripts.inference.inference import visualize_signal_prediction
from model.model import SimpleSynthNetwork
from synth.synth_architecture import SynthModular
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from tqdm import tqdm

from train_helper import *


def train_single_epoch(model,
                       lfo_net,
                       epoch,
                       data_loader,
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
    epoch_param_diffs = defaultdict(list)
    epoch_param_vals_raw, epoch_param_vals = defaultdict(list), defaultdict(list)
    with tqdm(data_loader, unit="batch") as tepoch:
        for target_signal, target_param_dict, signal_index in tepoch:

            step = epoch * len(data_loader) + num_of_mini_batches

            tepoch.set_description(f"Epoch {epoch}")

            # set_to_none as advised in page 6:
            # https://tigress-web.princeton.edu/~jdh4/PyTorchPerformanceTuningGuide_GTC2021.pdf
            model.zero_grad(set_to_none=True)

            target_signal = target_signal.to(device)
            transformed_signal = transform(target_signal)

            # -----------Run Model-----------------
            output_params = model(transformed_signal)

            for op_idx, op_dict in output_params.items():
                for param_name, param_vals in op_dict['parameters'].items():
                    epoch_param_vals_raw[f'{op_idx}_{param_name}'].extend(param_vals.cpu().detach().numpy())

            denormalized_output_dict = normalizer.denormalize(output_params)
            predicted_param_dict = helper.clamp_adsr_params(denormalized_output_dict, synth_cfg, cfg)

            for op_idx, op_dict in predicted_param_dict.items():
                for param_name, param_vals in op_dict['parameters'].items():
                    epoch_param_vals[f'{op_idx}_{param_name}'].extend(param_vals.cpu().detach().numpy())

            batch_param_diffs = get_param_diffs(predicted_param_dict, target_param_dict)
            for op_idx, diff_vals in batch_param_diffs.items():
                epoch_param_diffs[op_idx].extend(diff_vals)

            # -------------Generate Signal-------------------------------
            # --------------Predicted-------------------------------------
            params_for_pred_signal_generation = target_param_dict.copy()
            params_for_pred_signal_generation.update(predicted_param_dict)
            modular_synth.update_cells_from_dict(params_for_pred_signal_generation)
            pred_final_signal, pred_signals_through_chain = \
                modular_synth.generate_signal(batch_size=len(transformed_signal))

            # --------------Target-------------------------------------
            modular_synth.update_cells_from_dict(target_param_dict)
            target_final_signal, target_signals_through_chain = \
                modular_synth.generate_signal(batch_size=len(transformed_signal))

            modular_synth.signal = helper.move_to(modular_synth.signal, device)
            target_signal = target_signal.squeeze()

            # Compute loss and backprop
            loss_total = 0
            for op_index in output_params.keys():
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

            num_of_mini_batches += 1
            sum_epoch_loss += loss_total.item()
            loss_total.backward()
            optimizer.step()
            scheduler.step()

            # Log step stats
            summary_writer.add_scalar('loss/train_multi_spectral', loss_total, step)
            summary_writer.add_scalar('lr_adam', optimizer.param_groups[0]['lr'], step)

            if num_of_mini_batches == 1:
                for i in range(5):
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

            if num_of_mini_batches % 100 == 0:
                log_gradients_in_model(model, summary_writer, step)

            tepoch.set_postfix(loss=loss_total.item())

    # Log epoch stats
    avg_epoch_loss = sum_epoch_loss / num_of_mini_batches
    summary_writer.add_scalar('loss/train_multi_spectral_epoch', avg_epoch_loss, epoch)
    log_dict_recursive('param_diff', epoch_param_diffs, summary_writer, epoch)
    log_dict_recursive('param_values_raw', epoch_param_vals_raw, summary_writer, epoch)
    log_dict_recursive('param_values_normalized', epoch_param_vals, summary_writer, epoch)

    return avg_epoch_loss


def train(model,
          lfo_net,
          data_loader,
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

    # init modular synth
    modular_synth = SynthModular(synth_cfg=synth_cfg,
                                 sample_rate=cfg.sample_rate,
                                 signal_duration_sec=cfg.signal_duration_sec,
                                 num_sounds=1,
                                 device=device,
                                 preset_name=synth_cfg.preset)

    for epoch in range(num_epochs):
        cur_epoch = start_epoch + epoch
        avg_epoch_loss = \
            train_single_epoch(model=model,
                               lfo_net=lfo_net,
                               epoch=cur_epoch,
                               data_loader=data_loader,
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

    # Hard coded lfo predictor net
    lfo_net = SimpleSynthNetwork('LFO', synth_cfg, cfg, device, backbone=model_cfg.backbone).to(device)
    checkpoint = torch.load(cfg.lfo_model_path)
    lfo_net.load_state_dict(checkpoint['model_state_dict'])
    lfo_net.eval()
    lfo_net.requires_grad_(False)

    # construct model and assign it to device
    if model_cfg.model_type == 'simple':
        synth_net = SimpleSynthNetwork(model_cfg.preset, synth_cfg, cfg, device, backbone=model_cfg.backbone).to(device)
    else:
        raise NotImplementedError("only SimpleSynthNetwork supported at the moment")

    optimizer = torch.optim.Adam(synth_net.parameters(),
                                 lr=model_cfg.learning_rate,
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
          lfo_net=lfo_net,
          data_loader=train_dataloader,
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
