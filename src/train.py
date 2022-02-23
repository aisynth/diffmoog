import torch
from torch import nn
from config import Config, SynthConfig, DatasetConfig, ModelConfig
from ai_synth_dataset import AiSynthDataset, create_data_loader
from model import BigSynthNetwork
from synth.synth_architecture import SynthModular, SynthModularCell
from synth import synth_modular_presets
from torch.distributions import Categorical
import helper
import time
import os
import hydra
from omegaconf import DictConfig
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

# writer = SummaryWriter()


def train_single_epoch(model, data_loader, transform, optimizer_arg, scheduler_arg, device_arg,
                       modular_synth, normalizer, synth_cfg, cfg, record_number):
    sum_epoch_loss, sum_epoch_accuracy, sum_stats = helper.reset_stats(synth_cfg)
    num_of_mini_batches = 0
    for target_signal, target_param_dic in data_loader:
        batch_start_time = time.time()

        # set_to_none as advised in page 6:
        # https://tigress-web.princeton.edu/~jdh4/PyTorchPerformanceTuningGuide_GTC2021.pdf
        model.zero_grad(set_to_none=True)

        transformed_signal = transform(target_signal)
        # -------------------------------------
        # -----------Run Model-----------------
        # -------------------------------------
        # signal_log_mel_spec.requires_grad = True
        model_start_time = time.time()
        output_dic = model(transformed_signal)
        model_end_time = time.time()

        synth_start_time = time.time()

        # helper.map_classification_params_from_ints(predicted_dic)
        denormalized_output_dict = normalizer.denormalize(output_dic)
        predicted_param_dict = helper.clamp_regression_params(denormalized_output_dict, synth_cfg, cfg)

        update_params = []
        for index, operation_dict in predicted_param_dict.items():
            synth_modular_cell = SynthModularCell(index=index, parameters=operation_dict['params'])
            update_params.append(synth_modular_cell)

        modular_synth.update_cells(update_params)
        modular_synth.generate_signal(num_sounds=len(transformed_signal))

        modular_synth.signal = helper.move_to(modular_synth.signal, device_arg)

        synth_end_time = time.time()

        # predicted_transformed_signal = transform(modular_synth.signal)
        # transformed_signal = torch.squeeze(transformed_signal)
        # for i in range(5, 7):
        #     helper.plot_spectrogram(transformed_signal[i].cpu().detach().numpy(),
        #                             title=f"True MelSpectrogram (dB) for sound {i}",
        #                             ylabel='mel freq')
        #
        #     helper.plot_spectrogram(predicted_transformed_signal[i].cpu().detach().numpy(),
        #                             title=f"Predicted MelSpectrogram (dB) sound {i}",
        #                             ylabel='mel freq')

        loss_start_time = time.time()

        if cfg.spectrogram_loss_type == 'MULTI-SPECTRAL':
            multi_spec_loss = helper.SpectralLoss(cfg, device=device_arg)
            target_signal = target_signal.squeeze()
            loss = multi_spec_loss.call(target_signal, modular_synth.signal)
            # writer.add_scalar("Loss/train", loss, record_number)
        else:
            ValueError("SYNTH_TYPE 'MODULAR' supports only SPECTROGRAM_LOSS_TYPE of type 'MULTI-SPECTRAL'")

        loss_end_time = time.time()

        backward_start_time = time.time()

        num_of_mini_batches += 1
        sum_epoch_loss += loss.item()
        loss.backward()
        optimizer_arg.step()
        scheduler_arg.step()
        record_number += 1
        # writer.flush()

        # Check Accuracy
        # if cfg.architecture == 'SPEC_NO_SYNTH' \
        #         or (cfg.model_frequency_output == 'LOGITS' and cfg.architecture == 'PARAMETERS_ONLY') \
        #         or (cfg.model_frequency_output == 'PROBS' and cfg.architecture == 'REINFORCE'):
        #     correct = 0
        #     correct += (osc_logits_pred.argmax(1) == osc_target_id).type(torch.float).sum().item()
        #     accuracy = correct / len(osc_logits_pred)
        #     sum_epoch_accuracy += accuracy
        #
        # if cfg.architecture == 'FULL' or cfg.architecture == 'SPECTROGRAM_ONLY' or \
        #         (cfg.architecture == 'PARAMETERS_ONLY' and cfg.freq_param_loss_type == 'MSE'):
        #     accuracy, stats = helper.regression_freq_accuracy(output_dic, target_param_dic, device_arg, synth_cfg, cfg)
        #     sum_epoch_accuracy += accuracy
        #     for i in range(len(synth_cfg.osc_freq_list)):
        #         sum_stats[i]['prediction_success'] += stats[i]['prediction_success']
        #         sum_stats[i]['predicted_frequency'] += stats[i]['predicted_frequency']
        #         sum_stats[i]['frequency_model_output'] += stats[i]['frequency_model_output']
        # else:
        #     ValueError("Unknown architecture")

        batch_end_time = time.time()

        if cfg.print_train_stats:

            if cfg.architecture == 'SPECTROGRAM_ONLY' or cfg.architecture == 'SPEC_NO_SYNTH':

                backward_end_time = time.time()

                print(
                    f"MSE loss: {round(loss.item(), 7)},\n"
                    f"batch processing time: {round(batch_end_time - batch_start_time, 2)}s, \n"
                    f"model processing time: {round(model_end_time - model_start_time, 2)}s, \t"
                    f"synth processing time: {round(synth_end_time - synth_start_time, 2)}s, \t"
                    f"backward processing time: {round(backward_end_time - backward_start_time, 2)}s, \t"
                    f"loss processing time: {round(loss_end_time - loss_start_time, 2)}s\n")
                helper.print_modular_stats(predicted_param_dict, target_param_dic, synth_cfg)
                print('---------------------')

            else:
                ValueError("Unknown architecture")


    avg_epoch_loss = sum_epoch_loss / num_of_mini_batches
    avg_epoch_accuracy = sum_epoch_accuracy / num_of_mini_batches
    avg_stats = sum_stats
    for i in range(len(synth_cfg.osc_freq_list)):
        avg_stats[i]['prediction_success'] = sum_stats[i]['prediction_success'] / num_of_mini_batches
        avg_stats[i]['predicted_frequency'] = sum_stats[i]['predicted_frequency'] / num_of_mini_batches
        avg_stats[i]['frequency_model_output'] = sum_stats[i]['frequency_model_output'] / num_of_mini_batches

    return avg_epoch_loss, avg_epoch_accuracy, avg_stats


def train(model, data_loader, transform, optimiser_arg, device_arg, cur_epoch, num_epochs,
          cfg: Config, synth_cfg: SynthConfig, record_number=0):
    model.train()

    loss_list = []
    accuracy_list = []

    multi_epoch_avg_loss, multi_epoch_avg_accuracy, multi_epoch_stats = helper.reset_stats(synth_cfg)

    scheduler = torch.optim.lr_scheduler.StepLR(optimiser_arg, step_size=5000, gamma=0.1)
    # Initializations
    criterion_spectrogram = nn.MSELoss()
    if cfg.freq_param_loss_type == 'CE':
        criterion_osc_freq = nn.CrossEntropyLoss()
    elif cfg.freq_param_loss_type == 'MSE':
        criterion_osc_freq = nn.MSELoss()
    normalizer = helper.Normalizer(cfg.signal_duration_sec, synth_cfg)
    torch.autograd.set_detect_anomaly(True)

    # init modular synth architecture
    modular_synth = SynthModular(synth_cfg=synth_cfg,
                                 num_channels=synth_cfg.num_channels,
                                 num_layers=synth_cfg.num_channels,
                                 sample_rate=cfg.sample_rate,
                                 signal_duration_sec=cfg.signal_duration_sec,
                                 num_sounds=1,
                                 device=device_arg
                                 )

    if cfg.preset == 'BASIC_FLOW':
        preset = synth_modular_presets.BASIC_FLOW
    elif cfg.preset == 'FM':
        preset = synth_modular_presets.FM
    else:
        preset = None
        ValueError("Unknown PRESET")

    modular_synth.apply_architecture(preset)

    for epoch in range(num_epochs):
        if epoch % 100 == 0:
            print(f"Epoch {cur_epoch} start")

        avg_epoch_loss, avg_epoch_accuracy, avg_epoch_stats = \
            train_single_epoch(model, data_loader, transform, optimiser_arg, scheduler, device_arg,
                               modular_synth, normalizer, synth_cfg, cfg, record_number)

        # Sum stats over multiple epochs
        multi_epoch_avg_loss += avg_epoch_loss
        multi_epoch_avg_accuracy += avg_epoch_accuracy
        for j in range(len(synth_cfg.osc_freq_list)):
            multi_epoch_stats[j]['prediction_success'] += avg_epoch_stats[j]['prediction_success']
            multi_epoch_stats[j]['predicted_frequency'] += avg_epoch_stats[j]['predicted_frequency']
            multi_epoch_stats[j]['frequency_model_output'] += avg_epoch_stats[j]['frequency_model_output']

        cur_epoch += 1

        # save model checkpoint
        if (epoch + 1) % cfg.num_epochs_to_save_model == 0 and epoch != 0:
            helper.save_model(cur_epoch, model, optimiser_arg, avg_epoch_loss, loss_list, accuracy_list)

        print("--------------------------------------")
        print(f"Epoch {cur_epoch} end")
        print(f"Average Epoch{cur_epoch} Loss:", round(avg_epoch_loss, 5))
        print(f"Average Epoch{cur_epoch} Accuracy: {round(avg_epoch_accuracy * 100, 2)}%")
        print("--------------------------------------\n")

        loss_list.append(avg_epoch_loss)
        accuracy_list.append(avg_epoch_accuracy * 100)

        # save model checkpoint
        helper.save_model(cur_epoch, model, optimiser_arg, avg_epoch_loss, loss_list, accuracy_list)

    print("Finished training")


# @hydra.main(config_path="conf", config_name="model_config")
def run():
    transforms = {'mel': helper.mel_spectrogram_transform(16000),
                  'spec': helper.spectrogram_transform}
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            description='Train AI Synth')
    parser.add_argument('-g', '--gpu_index', help='index of gpu (if exist, torch indexing) -1 for cpu',
                        type=int, default=0)
    parser.add_argument('-t', '--transform', choices=transforms.keys(),
                        help='mel: Mel Spectogram, spec: Spectogram', default='mel')

    args = parser.parse_args()
    device = helper.get_device(args.gpu_index)
    transform = transforms[args.transform]

    cfg = Config()
    synth_cfg = SynthConfig()
    dataset_cfg = DatasetConfig()
    model_cfg = ModelConfig()

    ai_synth_dataset = AiSynthDataset(dataset_cfg.train_parameters_file, dataset_cfg.train_audio_dir, device)
    train_dataloader = create_data_loader(ai_synth_dataset, model_cfg.batch_size)

    # construct model and assign it to device
    synth_net = BigSynthNetwork(synth_cfg).to(device)

    # initialize optimizer
    optimizer = torch.optim.Adam(synth_net.parameters(), lr=model_cfg.learning_rate, weight_decay=1e-4)

    print(f"Training model with: \n LOSS_MODE: {cfg.architecture} \n")

    if cfg.use_loaded_model:
        checkpoint = torch.load(cfg.load_model_path)
        synth_net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        cur_epoch = checkpoint['epoch']
        loss = checkpoint['loss']
    else:
        cur_epoch = 0

    if cfg.override_optimizer:
        optimizer = torch.optim.Adam(synth_net.parameters(), lr=model_cfg.learning_rate, weight_decay=1e-4)

    record_number = 0

    # train model
    train(synth_net, train_dataloader, transform, optimizer, device,
          cur_epoch=cur_epoch,
          num_epochs=model_cfg.num_epochs,
          cfg=cfg,
          synth_cfg=synth_cfg,
          record_number=record_number)

    path_parent = os.path.dirname(os.getcwd())

    # save model
    saved_model_path = Path(__file__).parent.parent.joinpath('trained_models', 'trained_synth_net.pth')

    torch.save(synth_net.state_dict(), saved_model_path)
    print("Trained synth net saved at synth_net.pth")


if __name__ == "__main__":
    run()
