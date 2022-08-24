import copy
import os.path
import sys

from run_scripts.inference.inference_helper import lsd, inference_loop

sys.path.append("..")

from collections import defaultdict

from config import Config, ModelConfig, configure_experiment
from dataset.ai_synth_dataset import AiSynthDataset, create_data_loader
from run_scripts.inference.inference import visualize_signal_prediction
from model.model import SimpleSynthNetwork
from model.spectral_loss import SpectralLoss
from model.parameters_loss import ParametersLoss
from synth.synth_architecture import SynthModular
from model import helper
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from tqdm import tqdm
from collections.abc import Iterable

from train_helper import *

sys.path.append(".")

def train_single_epoch(model,
                       epoch,
                       train_data_loader,
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
    with tqdm(train_data_loader, unit="batch") as tepoch:
        for target_signal, target_param_dict, signal_index in tepoch:

            num_sounds = len(signal_index)
            step = epoch * len(train_data_loader) + num_of_mini_batches

            tepoch.set_description(f"Epoch {epoch}")

            # set_to_none as advised in page 6:
            # https://tigress-web.princeton.edu/~jdh4/PyTorchPerformanceTuningGuide_GTC2021.pdf
            model.zero_grad(set_to_none=True)

            target_param_dict = helper.move_to(target_param_dict, device)
            denormalized_target_params = normalizer.normalize(target_param_dict.copy())

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

            target_param_dict = helper.build_envelope_from_adsr(target_param_dict,
                                                                cfg,
                                                                device)

            batch_param_diffs = get_param_diffs(predicted_param_dict.copy(), target_param_dict.copy())
            for op_idx, diff_vals in batch_param_diffs.items():
                # print(op_idx, diff_vals)
                if isinstance(diff_vals, Iterable):
                    epoch_param_diffs[op_idx].extend(diff_vals)
                else:
                    epoch_param_diffs[op_idx].append(diff_vals)

            parameters_loss = loss_handler['parameters_loss'].call(predicted_parameters_dict=output_params,
                                                                   target_parameters_dict=denormalized_target_params,
                                                                   summary_writer=summary_writer,
                                                                   global_step=step)

            # -------------Generate Signal-------------------------------
            # --------------Target-------------------------------------
            modular_synth.update_cells_from_dict(target_param_dict)
            target_final_signal, target_signals_through_chain = \
                modular_synth.generate_signal(batch_size=num_sounds)

            # --------------Predicted-------------------------------------
            params_for_pred_signal_generation = copy.copy(target_param_dict)
            params_for_pred_signal_generation.update(predicted_param_dict)
            modular_synth.update_cells_from_dict(params_for_pred_signal_generation)
            pred_final_signal, pred_signals_through_chain = \
                modular_synth.generate_signal(batch_size=num_sounds)

            spectrogram_loss = 0
            for op_index in output_params.keys():
                op_index = str(op_index)

                pred_signal = pred_signals_through_chain[op_index]
                if pred_signal is None:
                    continue

                target_signal = target_signals_through_chain[op_index]
                loss, ret_spectrograms = loss_handler['spectrogram_loss'].call(target_signal,
                                                                               pred_signal,
                                                                               summary_writer,
                                                                               op_index,
                                                                               step,
                                                                               return_spectrogram=True)
                spectrogram_loss += loss

            loss_total = cfg.parameters_loss_weight * parameters_loss + cfg.spectrogram_loss_weight * spectrogram_loss
            lsd_value = np.mean(lsd(transformed_signal.squeeze().detach().cpu().numpy(),
                            transform(pred_final_signal).detach().cpu().numpy()))

            num_of_mini_batches += 1
            sum_epoch_loss += loss_total.item()
            loss_total.backward()
            optimizer.step()
            scheduler.step()

            # Log step stats
            summary_writer.add_scalar('loss/train_parameters_loss', cfg.parameters_loss_weight * parameters_loss, step)
            summary_writer.add_scalar('loss/train_spectral_loss', cfg.spectrogram_loss_weight * spectrogram_loss, step)
            summary_writer.add_scalar('loss/total_loss', loss_total, step)
            summary_writer.add_scalar('metrics/lsd', lsd_value, step)
            summary_writer.add_scalar('lr_adam', optimizer.param_groups[0]['lr'], step)

            if num_of_mini_batches == 1:
                if num_sounds == 1:
                    sample_params_orig, sample_params_pred = parse_synth_params(target_param_dict,
                                                                                predicted_param_dict,
                                                                                0)
                    summary_writer.add_audio(f'input_0_target', target_final_signal, global_step=epoch,
                                             sample_rate=cfg.sample_rate)
                    summary_writer.add_audio(f'input_0_pred', pred_final_signal, global_step=epoch,
                                             sample_rate=cfg.sample_rate)

                    pred_final_signal = pred_final_signal.squeeze()
                    target_final_signal = target_final_signal.squeeze()
                    signal_vis = visualize_signal_prediction(target_final_signal, pred_final_signal,
                                                             sample_params_orig, sample_params_pred, db=True)
                    signal_vis_t = torch.tensor(signal_vis, dtype=torch.uint8, requires_grad=False)

                    summary_writer.add_image(f'{256}_spec/input_0',
                                             signal_vis_t,
                                             global_step=epoch,
                                             dataformats='HWC')
                else:
                    if num_sounds < 5:
                        range_ = range(num_sounds)
                    else:
                        range_ = range(5)
                    for i in range_:
                        sample_params_orig, sample_params_pred = parse_synth_params(target_param_dict,
                                                                                    predicted_param_dict,
                                                                                    i)
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

            if num_of_mini_batches % 50 == 0:
                log_gradients_in_model(model, summary_writer, step)

            tepoch.set_postfix(loss=loss_total.item())

    # Log epoch stats
    avg_epoch_loss = sum_epoch_loss / num_of_mini_batches
    summary_writer.add_scalar('loss/train_loss_epoch', avg_epoch_loss, epoch)
    log_dict_recursive('param_diff', epoch_param_diffs, summary_writer, epoch)
    log_dict_recursive('param_values_raw', epoch_param_vals_raw, summary_writer, epoch)
    log_dict_recursive('param_values_normalized', epoch_param_vals, summary_writer, epoch)

    return avg_epoch_loss


def train(model,
          train_data_loader,
          val_dataloader,
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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=num_epochs * len(train_data_loader))
    # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, model_cfg.learning_rate / 10, model_cfg.learning_rate,
    #                                               mode='triangular2', cycle_momentum=False)
    normalizer = helper.Normalizer(cfg.signal_duration_sec, synth_cfg)

    loss_handler = {}
    if cfg.spectrogram_loss_type == 'MULTI-SPECTRAL':
        loss_preset = cfg.multi_spectral_loss_preset
        total_train_steps = num_epochs * len(train_data_loader)
        loss_handler['spectrogram_loss'] = SpectralLoss(cfg=cfg,
                                                        preset_name=loss_preset,
                                                        device=device,
                                                        total_train_steps=total_train_steps)
    else:
        raise ValueError("SYNTH_TYPE 'MODULAR' supports only SPECTROGRAM_LOSS_TYPE of type 'MULTI-SPECTRAL'")

    if cfg.add_parameters_loss:
        loss_handler['parameters_loss'] = ParametersLoss(cfg=cfg,
                                                         loss_type=cfg.parameters_loss_type,
                                                         device=device)

    # init modular synth
    modular_synth = SynthModular(synth_cfg=synth_cfg,
                                 sample_rate=cfg.sample_rate,
                                 signal_duration_sec=cfg.signal_duration_sec,
                                 num_sounds_=1,
                                 device=device,
                                 preset_name=synth_cfg.preset)

    for epoch in range(num_epochs):
        cur_epoch = start_epoch + epoch
        avg_epoch_loss = \
            train_single_epoch(model=model,
                               epoch=cur_epoch,
                               train_data_loader=train_data_loader,
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

        epoch_val_metrics = inference_loop(cfg=cfg, synth_cfg=synth_cfg, synth=modular_synth,
                                           test_dataloader=val_dataloader, preprocess_fn=transform, eval_fn=model,
                                           post_process_fn=normalizer.denormalize, device=device)
        
        summary_writer.add_scalars('val_metrics', epoch_val_metrics, epoch)

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

    train_dataset = AiSynthDataset(dataset_cfg.train_parameters_file, dataset_cfg.train_audio_dir, device)
    train_dataloader = create_data_loader(train_dataset, model_cfg.batch_size, ModelConfig.num_workers)
    
    val_dataset = AiSynthDataset(dataset_cfg.val_parameters_file, dataset_cfg.val_audio_dir, device)
    val_dataloader = create_data_loader(val_dataset, model_cfg.batch_size, ModelConfig.num_workers)

    # construct model and assign it to device
    if model_cfg.model_type == 'simple':
        synth_net = SimpleSynthNetwork(model_cfg.preset, synth_cfg, cfg, device, backbone=model_cfg.backbone).to(device)
    else:
        raise NotImplementedError("only SimpleSynthNetwork supported at the moment")

    # for name, layer in synth_net.named_modules():
    #     layer.register_forward_hook(get_activation(name, activations_dict))

    optimizer = torch.optim.Adam(synth_net.parameters(),
                                 lr=model_cfg.learning_rate,
                                 weight_decay=model_cfg.optimizer_weight_decay)

    print(f"Training model start")

    if cfg.use_loaded_model:
        print(f"Use Loaded model {cfg.load_model_path}")
        checkpoint = torch.load(cfg.load_model_path)
        synth_net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        cur_epoch = checkpoint['epoch'] + 1
        loss = checkpoint['loss']
    else:
        cur_epoch = 0

    # train model
    train(model=synth_net,
          train_data_loader=train_dataloader,
          val_dataloader=val_dataloader,
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
