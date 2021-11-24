import torch
import matplotlib.pyplot as plt
from torch import nn
from config import BATCH_SIZE, EPOCHS, LEARNING_RATE, DEBUG_MODE, REGRESSION_LOSS_FACTOR, \
    SPECTROGRAM_LOSS_FACTOR, PRINT_TRAIN_STATS, LOSS_MODE, SAVE_MODEL_PATH, DATASET_MODE, DATASET_TYPE, SYNTH_TYPE, \
    SPECTROGRAM_LOSS_TYPE, LOAD_MODEL_PATH, USE_LOADED_MODEL, FREQ_PARAM_LOSS_TYPE, FREQ_MSE_LOSS_FACTOR, \
    LOG_SPECTROGRAM_MSE_LOSS, ONLY_OSC_DATASET
from ai_synth_dataset import AiSynthDataset, AiSynthSingleOscDataset
from config import TRAIN_PARAMETERS_FILE, TRAIN_AUDIO_DIR, OS, PLOT_SPEC
from synth_model import SynthNetwork
from sound_generator import SynthBasicFlow, SynthOscOnly
from synth_config import OSC_FREQ_LIST, OSC_FREQ_DIC_INV
import synth
import helper
import time
import os


def train_single_epoch(model, data_loader, optimizer_arg, device_arg):
    # Initializations
    criterion_spectrogram = nn.MSELoss()
    criterion_kl = nn.KLDivLoss()
    if FREQ_PARAM_LOSS_TYPE == 'CE':
        criterion_osc_freq = nn.CrossEntropyLoss()
    elif FREQ_PARAM_LOSS_TYPE == 'MSE':
        criterion_osc_freq = nn.MSELoss()
    normalizer = helper.Normalizer()
    torch.autograd.set_detect_anomaly(True)

    sum_epoch_loss = 0
    sum_epoch_accuracy = 0
    num_of_mini_batches = 0

    for signal_mel_spec, target_params_dic in data_loader:
        start = time.time()

        # set_to_none as advised in page 6:
        # https://tigress-web.princeton.edu/~jdh4/PyTorchPerformanceTuningGuide_GTC2021.pdf
        model.zero_grad(set_to_none=True)

        # -------------------------------------
        # -----------Run Model-----------------
        # -------------------------------------
        # signal_log_mel_spec.requires_grad = True

        if SYNTH_TYPE == 'OSC_ONLY':
            output_dic, osc_logits_pred = model(signal_mel_spec)
        else:
            output_dic = model(signal_mel_spec)

        # helper.map_classification_params_from_ints(predicted_dic)

        osc_target_id = target_params_dic['classification_params']['osc1_freq']
        osc_target = torch.tensor([OSC_FREQ_DIC_INV[x.item()] for x in osc_target_id], device=device_arg)
        osc_pred = output_dic['osc1_freq']

        if LOSS_MODE == 'PARAMETERS_ONLY':
            if FREQ_PARAM_LOSS_TYPE == 'CE':
                ce_loss = criterion_osc_freq(osc_logits_pred, osc_target_id)
                loss = ce_loss

            elif FREQ_PARAM_LOSS_TYPE == 'MSE':
                frequency_mse_loss = criterion_osc_freq(osc_pred, osc_target)

                criterion = helper.RMSLELoss()
                frequency_mse_loss = criterion(osc_pred, osc_target)

                # DEBUG print
                # for i in range(len(osc_target)):
                #     print(f"Predicted freq: {osc_pred[i].item()}, True freq: {osc_target[i].item()}")

                loss = FREQ_MSE_LOSS_FACTOR * frequency_mse_loss
            else:
                raise ValueError("Provided FREQ_PARAM_LOSS_TYPE is not recognized")

        if LOSS_MODE == 'FULL' or LOSS_MODE == 'SPECTROGRAM_ONLY':
            denormalized_output_dic = normalizer.denormalize(output_dic)
            if SYNTH_TYPE == 'OSC_ONLY':
                param_dict_to_synth = denormalized_output_dic

                # -------------------------------------
                # -----------Run Synth-----------------
                # -------------------------------------
                overall_synth_obj = SynthOscOnly(parameters_dict=param_dict_to_synth,
                                                 num_sounds=len(signal_mel_spec))

            elif SYNTH_TYPE == 'SYNTH_BASIC':
                param_dict_to_synth = helper.clamp_regression_params(denormalized_output_dic)

                # -------------------------------------
                # -----------Run Synth-----------------
                # -------------------------------------
                overall_synth_obj = SynthBasicFlow(parameters_dict=param_dict_to_synth,
                                                   num_sounds=len(signal_mel_spec))

            else:
                raise ValueError("Provided SYNTH_TYPE is not recognized")

            overall_synth_obj.signal = helper.move_to(overall_synth_obj.signal, device_arg)

            predicted_mel_spec_sound_signal = helper.mel_spectrogram_transform(overall_synth_obj.signal)
            predicted_mel_spec_sound_signal = helper.move_to(predicted_mel_spec_sound_signal, device_arg)

            signal_mel_spec = torch.squeeze(signal_mel_spec)
            predicted_mel_spec_sound_signal = torch.squeeze(predicted_mel_spec_sound_signal)

            if PLOT_SPEC:
                helper.plot_spectrogram(signal_mel_spec[0].cpu(),
                                        title=f"True MelSpectrogram (dB) for frequency {osc_target[0]}",
                                        ylabel='mel freq')

                helper.plot_spectrogram(predicted_mel_spec_sound_signal[0].cpu().detach().numpy(),
                                        title=f"Predicted MelSpectrogram (dB) {osc_pred[0]}",
                                        ylabel='mel freq')

            if LOG_SPECTROGRAM_MSE_LOSS:
                spectrogram_mse_loss = criterion_spectrogram(torch.log(predicted_mel_spec_sound_signal+1),
                                                             torch.log(signal_mel_spec+1))
            else:
                spectrogram_mse_loss = criterion_spectrogram(predicted_mel_spec_sound_signal, signal_mel_spec)

            lsd_loss = helper.lsd_loss(signal_mel_spec, predicted_mel_spec_sound_signal)

            kl_loss = criterion_kl(signal_mel_spec, predicted_mel_spec_sound_signal)

            # todo: remove the individual synth inference code
            # -----------------------------------------------
            # -----------Compute sound individually----------
            # -----------------------------------------------

            if False:
                loss_spectrogram_total = 0
                current_predicted_dic = {}
                for i in range(len(signal_mel_spec)):
                    for key, value in output_dic.items():
                        current_predicted_dic[key] = output_dic[key][i]

                    # Generate sound from predicted parameters
                    synth_obj = SynthBasicFlow(parameters_dict=current_predicted_dic)

                    synth_obj.signal = helper.move_to(synth_obj.signal, device_arg)
                    predicted_log_mel_spec_sound_signal = helper.log_mel_spec_transform(synth_obj.signal)

                    predicted_log_mel_spec_sound_signal = helper.move_to(predicted_log_mel_spec_sound_signal,
                                                                         device_arg)

                    signal_mel_spec = torch.squeeze(signal_mel_spec)
                    predicted_log_mel_spec_sound_signal = torch.squeeze(predicted_log_mel_spec_sound_signal)
                    target_log_mel_spec_sound_signal = signal_mel_spec[i]
                    current_loss_spectrogram = criterion_spectrogram(predicted_log_mel_spec_sound_signal,
                                                                     target_log_mel_spec_sound_signal)
                    loss_spectrogram_total = loss_spectrogram_total + current_loss_spectrogram

                loss_spectrogram_total = loss_spectrogram_total / len(signal_mel_spec)

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
                        torch.cat(
                            [regression_target_parameters_tensor, regression_target_parameters[key].unsqueeze(dim=1)],
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
                if SPECTROGRAM_LOSS_TYPE == 'MSE':
                    loss = SPECTROGRAM_LOSS_FACTOR * spectrogram_mse_loss
                elif SPECTROGRAM_LOSS_TYPE == 'LSD':
                    loss = SPECTROGRAM_LOSS_FACTOR * lsd_loss
                elif SPECTROGRAM_LOSS_TYPE == 'KL':
                    loss = kl_loss
                else:
                    raise ValueError("Provided LOSS_TYPE is not recognized")
                # loss.requires_grad = True

        num_of_mini_batches += 1
        sum_epoch_loss += loss.item()
        loss.backward()
        optimizer_arg.step()

        # Check Accuracy
        if LOSS_MODE == 'PARAMETERS_ONLY':
            if FREQ_PARAM_LOSS_TYPE == 'CE':
                correct = 0
                correct += (osc_logits_pred.argmax(1) == osc_target_id).type(torch.float).sum().item()
                accuracy = correct / len(osc_logits_pred)
                sum_epoch_accuracy += accuracy
            elif FREQ_PARAM_LOSS_TYPE == 'MSE':
                accuracy = helper.regression_freq_accuracy(output_dic, target_params_dic, device_arg)
                sum_epoch_accuracy += accuracy

            # Sanity check - generate synth sounds with the resulted frequencies and compare spectrograms
            freq_ints_dict = {'osc1_freq': osc_logits_pred.argmax(1)}
            freq_hz_dict = helper.map_classification_params_from_ints(freq_ints_dict)

            overall_synth_obj = SynthOscOnly(parameters_dict=freq_hz_dict,
                                             num_sounds=len(signal_mel_spec))

            overall_synth_obj.signal = helper.move_to(overall_synth_obj.signal, device_arg)

            predicted_mel_spec_sound_signal = helper.mel_spectrogram_transform(overall_synth_obj.signal)
            predicted_mel_spec_sound_signal = helper.move_to(predicted_mel_spec_sound_signal, device_arg)

            signal_mel_spec = torch.squeeze(signal_mel_spec)
            predicted_mel_spec_sound_signal = torch.squeeze(predicted_mel_spec_sound_signal)

            spectrogram_mse_loss = criterion_spectrogram(predicted_mel_spec_sound_signal, signal_mel_spec)

            if PLOT_SPEC:
                helper.plot_spectrogram(signal_mel_spec[0].cpu(),
                                        title=f"True MelSpectrogram (dB) for frequency {osc_target[0]}",
                                        ylabel='mel freq')

                helper.plot_spectrogram(predicted_mel_spec_sound_signal[0].cpu().detach().numpy(),
                                        title=f"Predicted MelSpectrogram (dB) {osc_pred[0]}",
                                        ylabel='mel freq')

                # for i in range(len(signal_mel_spec)):
                #     print("Spec diff:", criterion_spectrogram(predicted_mel_spec_sound_signal[i], signal_mel_spec[i]).item())

                print("Spec diff of graphs:",
                      criterion_spectrogram(predicted_mel_spec_sound_signal[0], signal_mel_spec[0]).item())

                print("Spec diff of graphs:",
                      criterion_spectrogram(torch.log(predicted_mel_spec_sound_signal[0]+1), torch.log(signal_mel_spec[0]+1)).item())

        if LOSS_MODE == 'FULL' or LOSS_MODE == 'SPECTROGRAM_ONLY':
            accuracy = helper.regression_freq_accuracy(output_dic, target_params_dic, device_arg)
            sum_epoch_accuracy += accuracy

        end = time.time()

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

            if LOSS_MODE == 'SPECTROGRAM_ONLY':
                print(
                    f"MSE loss: {round(loss.item(), 2)},\t"
                    f"accuracy: {round(accuracy * 100, 2)}%,\t"
                    f"batch processing time: {round(end - start, 2)}s")

            if LOSS_MODE == 'PARAMETERS_ONLY':
                print(
                    f"Frequency {FREQ_PARAM_LOSS_TYPE} loss: {round(loss.item(), 2)},\t\t"
                    f"Accuracy: {round(accuracy * 100, 2)}%,\t"
                    f"Batch processing time: {round(end - start, 2)}s, \t"
                    f"MSE of spectrograms: {round(SPECTROGRAM_LOSS_FACTOR * spectrogram_mse_loss.item(), 2)}")

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

    avg_epoch_loss = sum_epoch_loss / num_of_mini_batches
    avg_epoch_accuracy = sum_epoch_accuracy / num_of_mini_batches
    return avg_epoch_loss, avg_epoch_accuracy


def train(model, data_loader, optimiser_arg, device_arg, cur_epoch, num_epochs):
    model.train()
    path_parent = os.path.dirname(os.getcwd())
    loss_list = []
    accuracy_list = []
    for i in range(num_epochs):
        if not ONLY_OSC_DATASET or i % 100 == 0:
            print(f"Epoch {cur_epoch} start")

        avg_epoch_loss, avg_epoch_accuracy = train_single_epoch(model, data_loader, optimiser_arg, device_arg)
        cur_epoch += 1
        if not ONLY_OSC_DATASET or (i % 100 == 0 and i != 0):
            print("--------------------------------------")
            print(f"Epoch {cur_epoch} end")
            print(f"Average Epoch{cur_epoch} Loss:", round(avg_epoch_loss, 2))
            print(f"Average Epoch{cur_epoch} Accuracy: {round(avg_epoch_accuracy * 100, 2)}%")
            print("--------------------------------------\n")

            # save model checkpoint
            if OS == 'WINDOWS':
                model_checkpoint = path_parent + f"\\trained_models\\synth_net_epoch{cur_epoch}.pth"
                plot_path = path_parent + f"\\trained_models\\loss_graphs\\end_epoch{cur_epoch}_loss_graph.png"
                txt_path = path_parent + f"\\trained_models\\loss_list.txt"
            elif OS == 'LINUX':
                model_checkpoint = path_parent + f"/ai_synth/trained_models/synth_net_epoch{cur_epoch}.pth"
                plot_path = path_parent + f"/ai_synth/trained_models/loss_graphs/end_epoch{cur_epoch}_loss_graph.png"
                txt_path = path_parent + f"/ai_synth/trained_models/loss_list.txt"

            torch.save({
                'epoch': cur_epoch,
                'model_state_dict': synth_net.state_dict(),
                'optimizer_state_dict': optimiser_arg.state_dict(),
                'loss': avg_epoch_loss
            }, model_checkpoint)

            loss_list.append(avg_epoch_loss)
            accuracy_list.append(avg_epoch_accuracy * 100)
            plt.savefig(plot_path)

            text_file = open(txt_path, "w")
            for j in range(len(loss_list)):
                text_file.write("loss: " + str(loss_list[j]) + " " + "accuracy: " + str(accuracy_list[j]) + "\n")
            text_file.close()

    print("Finished training")


if __name__ == "__main__":
    device = helper.get_device()

    if SYNTH_TYPE == 'OSC_ONLY':
        ai_synth_dataset = AiSynthSingleOscDataset(DATASET_MODE,
                                                   DATASET_TYPE,
                                                   TRAIN_PARAMETERS_FILE,
                                                   TRAIN_AUDIO_DIR,
                                                   helper.mel_spectrogram_transform,
                                                   synth.SAMPLE_RATE,
                                                   device
                                                   )
    elif SYNTH_TYPE == 'SYNTH_BASIC':
        ai_synth_dataset = AiSynthDataset(DATASET_MODE,
                                          DATASET_TYPE,
                                          TRAIN_PARAMETERS_FILE,
                                          TRAIN_AUDIO_DIR,
                                          helper.mel_spectrogram_transform,
                                          synth.SAMPLE_RATE,
                                          device
                                          )

    train_dataloader = helper.create_data_loader(ai_synth_dataset, BATCH_SIZE)

    # construct model and assign it to device
    synth_net = SynthNetwork().to(device)

    # initialize optimizer
    optimizer = torch.optim.Adam(synth_net.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    print(f"Training model with: \n LOSS_MODE: {LOSS_MODE} \n SYNTH_TYPE: {SYNTH_TYPE}")
    if USE_LOADED_MODEL:
        checkpoint = torch.load(LOAD_MODEL_PATH)
        synth_net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        cur_epoch = checkpoint['epoch']
        loss = checkpoint['loss']
    else:
        cur_epoch = 0

    # train model
    train(synth_net, train_dataloader, optimizer, device, cur_epoch=cur_epoch, num_epochs=EPOCHS)

    path_parent = os.path.dirname(os.getcwd())

    # save model
    if OS == 'WINDOWS':
        saved_model_path = path_parent + "\\ai_synth\\trained_models\\trained_synth_net.pth"
    elif OS == 'LINUX':
        saved_model_path = path_parent + "/ai_synth/trained_models/trained_synth_net.pth"

    torch.save(synth_net.state_dict(), saved_model_path)
    print("Trained synth net saved at synth_net.pth")
