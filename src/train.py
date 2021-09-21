import torch
from torch import nn
from torch.utils.data import DataLoader
from src.config import BATCH_SIZE, EPOCHS, LEARNING_RATE, DEBUG_MODE, REGRESSION_LOSS_FACTOR, \
    SPECTROGRAM_LOSS_FACTOR, PRINT_TRAIN_STATS, DATASET_MODE, LOSS_MODE
from src.ai_synth_dataset import AiSynthDataset
from src.config import PARAMETERS_FILE
from synth_model import SynthNetwork
from sound_generator import SynthBasicFlow
import synth
import helper
import time


def create_data_loader(train_data, batch_size):
    dataloader_obj = DataLoader(train_data, batch_size=batch_size)
    return dataloader_obj


def train_single_epoch(model, data_loader, optimizer_arg, device_arg):
    normalizer = helper.Normalizer()

    # Init criteria
    criterion_spectrogram = nn.MSELoss()

    for signal_log_mel_spec, target_params_dic in data_loader:
        start = time.time()

        if DEBUG_MODE:
            helper.plot_spectrogram(signal_log_mel_spec[0][0].cpu(), title="MelSpectrogram (dB)", ylabel='mel freq')

        optimizer_arg.zero_grad()
        # -------------------------------------
        # -----------Run Model-----------------
        # -------------------------------------
        output_dic = model(signal_log_mel_spec)

        helper.map_classification_params_from_ints(predicted_dic)
        normalizer.denormalize(predicted_dic)
        helper.clamp_regression_params(predicted_dic)

        # -------------------------------------
        # -----------Run Synth-----------------
        # -------------------------------------
        overall_synth_obj = SynthBasicFlow(parameters_dict=predicted_dic, num_sounds=len(signal_log_mel_spec))

        overall_synth_obj.signal = helper.move_to(overall_synth_obj.signal, device_arg)

        predicted_log_mel_spec_sound_signal = helper.log_mel_spec_transform(overall_synth_obj.signal)
        predicted_log_mel_spec_sound_signal = helper.move_to(predicted_log_mel_spec_sound_signal, device_arg)

        signal_log_mel_spec = torch.squeeze(signal_log_mel_spec)

        loss_spectrogram_total2 = criterion_spectrogram(predicted_log_mel_spec_sound_signal,
                                                        signal_log_mel_spec)

        # todo: remove the individual synth inference code
        # -----------------------------------------------
        # -----------Compute sound individually----------
        # -----------------------------------------------
        loss_spectrogram_total = 0
        current_predicted_dic = {}
        for i in range(len(signal_log_mel_spec)):
            for key, value in predicted_dic.items():
                if torch.is_tensor(predicted_dic[key][i]):
                    current_predicted_dic[key] = predicted_dic[key][i].item()
                else:
                    current_predicted_dic[key] = predicted_dic[key][i]

            # Generate sound from predicted parameters
            synth_obj = SynthBasicFlow(parameters_dict=current_predicted_dic)

            synth_obj.signal = helper.move_to(synth_obj.signal, device_arg)
            predicted_log_mel_spec_sound_signal = helper.log_mel_spec_transform(synth_obj.signal)

            predicted_log_mel_spec_sound_signal = helper.move_to(predicted_log_mel_spec_sound_signal, device_arg)

            signal_log_mel_spec = torch.squeeze(signal_log_mel_spec)
            predicted_log_mel_spec_sound_signal = torch.squeeze(predicted_log_mel_spec_sound_signal)
            target_log_mel_spec_sound_signal = signal_log_mel_spec[i]
            current_loss_spectrogram = criterion_spectrogram(predicted_log_mel_spec_sound_signal,
                                                             target_log_mel_spec_sound_signal)
            loss_spectrogram_total = loss_spectrogram_total + current_loss_spectrogram

        loss_spectrogram_total = loss_spectrogram_total / len(signal_log_mel_spec)

        # -----------------------------------------------
        #          -----------End Part----------
        # -----------------------------------------------

        if LOSS_MODE == "FULL":
            classification_target_params = target_params_dic['classification_params']
            regression_target_parameters = target_params_dic['regression_params']
            helper.map_classification_params_to_ints(classification_target_params)
            normalizer.normalize(regression_target_parameters)

            classification_target_params = helper.move_to(classification_target_params, device_arg)
            regression_target_parameters = helper.move_to(regression_target_parameters, device_arg)

            criterion_osc1_freq = criterion_osc1_wave = criterion_lfo1_wave \
                = criterion_osc2_freq = criterion_osc2_wave = criterion_lfo2_wave \
                = criterion_filter_type \
                = nn.CrossEntropyLoss()
            criterion_regression_params = nn.MSELoss()

            loss_osc1_freq = criterion_osc1_freq(output_dic['osc1_freq'], classification_target_params['osc1_freq'])
            loss_osc1_wave = criterion_osc1_wave(output_dic['osc1_wave'], classification_target_params['osc1_wave'])
            loss_lfo1_wave = criterion_lfo1_wave(output_dic['lfo1_wave'], classification_target_params['lfo1_wave'])
            loss_osc2_freq = criterion_osc2_freq(output_dic['osc2_freq'], classification_target_params['osc2_freq'])
            loss_osc2_wave = criterion_osc2_wave(output_dic['osc2_wave'], classification_target_params['osc2_wave'])
            loss_lfo2_wave = criterion_lfo2_wave(output_dic['lfo2_wave'], classification_target_params['lfo2_wave'])
            loss_filter_type = \
                criterion_filter_type(output_dic['filter_type'], classification_target_params['filter_type'])

            # todo: refactor code. the code gets dictionary of tensors
            #  (regression_target_parameters) and return 2d tensor
            regression_target_parameters_tensor = torch.empty((len(regression_target_parameters['osc1_amp']), 1))
            regression_target_parameters_tensor = helper.move_to(regression_target_parameters_tensor, device_arg)
            for key, value in regression_target_parameters.items():
                regression_target_parameters_tensor = \
                    torch.cat([regression_target_parameters_tensor, regression_target_parameters[key].unsqueeze(dim=1)],
                              dim=1)
            regression_target_parameters_tensor = regression_target_parameters_tensor[:, 1:]
            regression_target_parameters_tensor = regression_target_parameters_tensor.float()

            loss_classification_params = \
                loss_osc1_freq + loss_osc1_wave + loss_lfo1_wave + \
                loss_osc2_freq + loss_osc2_wave + loss_lfo2_wave + \
                loss_filter_type

            loss_regression_params = \
                criterion_regression_params(output_dic['regression_params'], regression_target_parameters_tensor)

            loss = \
                loss_classification_params \
                + REGRESSION_LOSS_FACTOR * loss_regression_params \
                + SPECTROGRAM_LOSS_FACTOR * loss_spectrogram_total

        if LOSS_MODE == 'SPECTROGRAM_ONLY':
            loss = loss_spectrogram_total
            loss.requires_grad = True

        # backpropogate error and update wights
        loss.backward()
        optimizer_arg.step()

        if PRINT_TRAIN_STATS:
            if LOSS_MODE == 'FULL':
                print("-----Classification params-----")
                print('loss_osc1_freq', loss_osc1_freq)
                print('loss_osc1_wave', loss_osc1_wave)
                print('loss_lfo1_wave', loss_lfo1_wave)
                print('loss_osc2_freq', loss_osc2_freq)
                print('loss_osc2_wave', loss_osc2_wave)
                print('loss_lfo2_wave', loss_lfo2_wave)
                print('loss_filter_type', loss_filter_type)
                print("-----Regression params-----")
                print('loss_regression_params', REGRESSION_LOSS_FACTOR * loss_regression_params)
                print('loss_spectrogram_total', SPECTROGRAM_LOSS_FACTOR * loss_spectrogram_total)
                print('\n')
            print("-----Total Loss-----")
            print(f"loss: {loss.item()}")
            print("--------------------")

        if DEBUG_MODE:
            print("osc1_freq",
                  torch.argmax(output_dic['osc1_freq'], dim=1), classification_target_params['osc1_freq'])
            print("osc1_wave",
                  torch.argmax(output_dic['osc1_wave'], dim=1), classification_target_params['osc1_wave'])
            print("lfo1_wave",
                  torch.argmax(output_dic['lfo1_wave'], dim=1), classification_target_params['lfo1_wave'])
            print("osc2_freq",
                  torch.argmax(output_dic['osc2_freq'], dim=1), classification_target_params['osc2_freq'])
            print("osc2_wave",
                  torch.argmax(output_dic['osc2_wave'], dim=1), classification_target_params['osc2_wave'])
            print("lfo2_wave",
                  torch.argmax(output_dic['lfo2_wave'], dim=1), classification_target_params['lfo2_wave'])
            print("filter_type",
                  torch.argmax(output_dic['filter_type'], dim=1), classification_target_params['filter_type'])
            print("regression_params",
                  output_dic['regression_params'], regression_target_parameters_tensor)

        end = time.time()
        print("batch processing time:", end - start)


def train(model, data_loader, optimiser_arg, device_arg, epochs):
    for i in range(epochs):
        print(f"Epoch {i + 1}")
        train_single_epoch(model, data_loader, optimiser_arg, device_arg)
        print("--------------------------------------\n")
    print("Finished training")


if __name__ == "__main__":
    device = helper.get_device()

    ai_synth_dataset = AiSynthDataset(csv_file=PARAMETERS_FILE,
                                      device_arg=device,
                                      dataset_mode=DATASET_MODE,
                                      )

    train_dataloader = create_data_loader(ai_synth_dataset, BATCH_SIZE)

    # construct model and assign it to device
    synth_net = SynthNetwork().to(device)

    # initialize optimizer
    optimiser = torch.optim.Adam(synth_net.parameters(), lr=LEARNING_RATE)

    # train model
    print("Training model with LOSS_MODE: ", LOSS_MODE)
    train(synth_net, train_dataloader, optimiser, device, EPOCHS)

    # save model
    torch.save(synth_net.state_dict(), "../trained_models/synth_net.pth")
    print("Trained synth net saved at synth_net.pth")
