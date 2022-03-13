import torch
from torch import nn
from config import Config, SynthConfig, DatasetConfig, ModelConfig
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
                       cfg):
    sum_epoch_loss = 0
    num_of_mini_batches = 0
    with tqdm(data_loader, unit="batch") as tepoch:
        for target_signal, target_param_dic in tepoch:

            tepoch.set_description(f"Epoch {epoch}")
            batch_start_time = time.time()

            # set_to_none as advised in page 6:
            # https://tigress-web.princeton.edu/~jdh4/PyTorchPerformanceTuningGuide_GTC2021.pdf
            model.zero_grad(set_to_none=True)
            transformed_signal = transform(target_signal)

            # -------------------------------------
            # -----------Run Model-----------------
            # -------------------------------------
            model_start_time = time.time()

            output_dic = model(transformed_signal)

            model_end_time = time.time()
            synth_start_time = time.time()

            denormalized_output_dict = normalizer.denormalize(output_dic)
            predicted_param_dict = helper.clamp_regression_params(denormalized_output_dict, synth_cfg, cfg)

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
                multi_spec_loss = helper.SpectralLoss(cfg, device=device)
                target_signal = target_signal.squeeze()
                loss = multi_spec_loss.call(target_signal, modular_synth.signal)
            else:
                ValueError("SYNTH_TYPE 'MODULAR' supports only SPECTROGRAM_LOSS_TYPE of type 'MULTI-SPECTRAL'")

            loss_end_time = time.time()
            backward_start_time = time.time()

            num_of_mini_batches += 1
            sum_epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()

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

    return avg_epoch_loss


def train(model,
          data_loader,
          transform,
          optimizer,
          device,
          cur_epoch: int,
          num_epochs: int,
          cfg: Config,
          model_cfg: ModelConfig,
          synth_cfg: SynthConfig):

    # Initializations
    model.train()
    torch.autograd.set_detect_anomaly(True)

    loss_list = []

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
                                                step_size=model_cfg.optimizer_scheduler_lr,
                                                gamma=model_cfg.optimizer_scheduler_gamma)
    normalizer = helper.Normalizer(cfg.signal_duration_sec, synth_cfg)

    # init modular synth
    modular_synth = SynthModular(synth_cfg=synth_cfg,
                                 sample_rate=cfg.sample_rate,
                                 signal_duration_sec=cfg.signal_duration_sec,
                                 num_sounds=1,
                                 device=device,
                                 preset=cfg.preset
                                 )

    for epoch in range(num_epochs):
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
                               cfg=cfg)

        # Sum stats over multiple epochs
        loss_list.append(avg_epoch_loss)

        # save model checkpoint
        helper.save_model(cur_epoch,
                          model,
                          optimizer,
                          avg_epoch_loss,
                          loss_list,
                          cfg.txt_path,
                          cfg.numpy_path)

        cur_epoch += 1

    print("Finished training")


def run():
    cfg = Config()
    synth_cfg = SynthConfig()
    dataset_cfg = DatasetConfig()
    model_cfg = ModelConfig()

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
    train_dataloader = create_data_loader(ai_synth_dataset, model_cfg.batch_size)

    # construct model and assign it to device
    synth_net = BigSynthNetwork(synth_cfg, device).to(device)
    optimizer = torch.optim.Adam(synth_net.parameters(),
                                 lr=model_cfg.learning_rate,
                                 weight_decay=model_cfg.optimizer_weight_decay)

    print(f"Training model start")

    if cfg.use_loaded_model:
        print(f"Use Loaded model {cfg.load_model_path.name}")
        checkpoint = torch.load(cfg.load_model_path)
        synth_net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        cur_epoch = checkpoint['epoch']
        loss = checkpoint['loss']
    else:
        cur_epoch = 0

    # train model
    train(model=synth_net,
          data_loader=train_dataloader,
          transform=transform,
          optimizer=optimizer,
          device=device,
          cur_epoch=cur_epoch,
          num_epochs=model_cfg.num_epochs,
          cfg=cfg,
          model_cfg=model_cfg,
          synth_cfg=synth_cfg)

    # save model
    torch.save(synth_net.state_dict(), cfg.save_model_path)
    print("Final trained synth net saved at trained_synth_net.pt")


if __name__ == "__main__":
    run()
