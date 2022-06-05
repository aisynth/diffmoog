from collections import defaultdict

from tqdm import tqdm

from config import Config, ModelConfig, configure_experiment
from dataset.ai_synth_dataset import AiSynthDataset
from run_scripts.inference.inference import visualize_signal_prediction
from model.model import DecoderOnlyNetwork
from synth.synth_architecture import SynthModular
from model import helper
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from train_helper import *


def train_single_step(model, step, sample, optimizer, scheduler, loss_handler, device, modular_synth,
                      normalizer, synth_cfg, cfg, summary_writer: SummaryWriter):

    target_signal, target_param_dict, signal_index = sample
    target_signal = target_signal.to(device)

    # set_to_none as advised in page 6:
    # https://tigress-web.princeton.edu/~jdh4/PyTorchPerformanceTuningGuide_GTC2021.pdf
    model.zero_grad(set_to_none=True)

    # Get decoder parameters
    output_dic = model()

    denormalized_output_dict = normalizer.denormalize(output_dic)
    predicted_param_dict = helper.clamp_adsr_params(denormalized_output_dict, synth_cfg, cfg)

    param_diffs = get_param_diffs(predicted_param_dict, target_param_dict)

    # Generate predicted signal
    modular_synth.update_cells_from_dict(predicted_param_dict)
    pred_signal, _ = modular_synth.generate_signal()

    # Compute loss and backprop
    loss, ret_spectrograms = loss_handler.call(target_signal, pred_signal, summary_writer, 'final', step,
                                               return_spectrogram=True)

    loss.backward()
    optimizer.step()
    scheduler.step()

    # Log step statistics
    summary_writer.add_scalar('loss/train_multi_spectral', loss, step)
    summary_writer.add_scalar('lr_adam', optimizer.param_groups[0]['lr'], step)

    if step % 100 == 1:
        sample_params_orig, sample_params_pred = parse_synth_params(target_param_dict, predicted_param_dict, 0)
        summary_writer.add_audio(f'input_{0}_target', target_signal[0], global_step=step,
                                 sample_rate=cfg.sample_rate)
        summary_writer.add_audio(f'input_{0}_pred', pred_signal[0], global_step=step,
                                 sample_rate=cfg.sample_rate)

        signal_vis = visualize_signal_prediction(target_signal[0], pred_signal[0],
                                                 sample_params_orig, sample_params_pred, db=True)
        signal_vis_t = torch.tensor(signal_vis, dtype=torch.uint8, requires_grad=False)

        summary_writer.add_image(f'{256}_spec/input_{0}', signal_vis_t, global_step=step, dataformats='HWC')

    log_gradients_in_model(model, summary_writer, step)

    log_dict_recursive('param_diff', param_diffs, summary_writer, step)
    log_dict_recursive('param_values_raw', output_dic, summary_writer, step)
    log_dict_recursive('param_values_normalized', predicted_param_dict, summary_writer, step)

    return loss.item()


def train(model, modular_synth, sample, optimizer, device, start_step: int, num_steps: int, cfg: Config,
          model_cfg: ModelConfig, synth_cfg: SynthConfig, summary_writer: SummaryWriter):

    # Initializations
    model.train()
    torch.autograd.set_detect_anomaly(True)

    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer=optimizer, factor=1)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=num_epochs * len(data_loader))
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

    for step in tqdm(range(num_steps)):
        cur_step = start_step + step
        train_single_step(model=model, step=cur_step, sample=sample, optimizer=optimizer, scheduler=scheduler,
                          device=device, modular_synth=modular_synth, normalizer=normalizer, loss_handler=loss_handler,
                          synth_cfg=synth_cfg, cfg=cfg, summary_writer=summary_writer)

    print("Finished training")


def run(args):

    # Init experiment
    exp_name = args.experiment
    dataset_name = args.dataset
    cfg, model_cfg, synth_cfg, dataset_cfg = configure_experiment(exp_name, dataset_name)

    sample_idx = 10

    summary_writer = SummaryWriter(cfg.tensorboard_logdir)

    device = helper.get_device(args.gpu_index)

    ai_synth_dataset = AiSynthDataset(dataset_cfg.train_parameters_file, dataset_cfg.train_audio_dir, device)
    sample = ai_synth_dataset[sample_idx]

    # init modular synth
    modular_synth = SynthModular(synth_cfg=synth_cfg, sample_rate=cfg.sample_rate, device=device, num_sounds=1,
                                 signal_duration_sec=cfg.signal_duration_sec, preset=synth_cfg.preset)

    modular_synth.generate_random_params(synth_cfg)
    init_params = modular_synth.collect_params()

    # Denormalize init params
    normalizer = helper.Normalizer(signal_duration_sec=cfg.signal_duration_sec, synth_cfg=synth_cfg)
    denormalized_init_params = normalizer.normalize(init_params)

    # construct model and assign it to device
    synth_net = DecoderOnlyNetwork(synth_cfg, device)
    synth_net.apply_params(denormalized_init_params)

    if args.params_to_freeze is not None:
        target_params = sample[1]
        normalized_target_params = normalizer.normalize(target_params)
        freeze_params = parse_args_to_freeze(args.params_to_freeze, normalized_target_params)
        synth_net.freeze_params(freeze_params)

    optimizer = torch.optim.SGD(synth_net.parameters(), lr=model_cfg.learning_rate,
                                weight_decay=model_cfg.optimizer_weight_decay)

    print(f"Training model start")

    # train model
    train(model=synth_net,
          modular_synth=modular_synth,
          sample=sample,
          optimizer=optimizer,
          device=device,
          start_step=0,
          num_steps=model_cfg.num_epochs,
          cfg=cfg,
          synth_cfg=synth_cfg,
          model_cfg=model_cfg,
          summary_writer=summary_writer)


def parse_args_to_freeze(params_to_freeze, target_vals):
    freeze_params = defaultdict(dict)
    for idx, params in params_to_freeze.items():
        freeze_params[idx]['operation'] = target_vals[idx]['operation']
        freeze_params[idx]['parameters'] = {}
        for param_name in params:
            if param_name == 'waveform':
                waveform_idx = SynthConfig.wave_type_dict[target_vals[idx]['parameters'][param_name]]
                freeze_params[idx]['parameters'][param_name] = [0, 0, 0]
                freeze_params[idx]['parameters'][param_name][waveform_idx] = 1
            elif param_name == 'filter_type':
                filter_type_idx = SynthConfig.filter_type_dict[target_vals[idx]['parameters'][param_name]]
                freeze_params[idx]['parameters'][param_name] = [0, 0]
                freeze_params[idx]['parameters'][param_name][filter_type_idx] = 1
            else:
                freeze_params[idx]['parameters'][param_name] = target_vals[idx]['parameters'][param_name]

    return freeze_params


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

    args.params_to_freeze = {(0, 0): ['freq'], (0, 1): ['waveform', 'mod_index']}

    run(args)
