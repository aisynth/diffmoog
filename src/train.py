import torch
import matplotlib.pyplot as plt
from torch import nn
from config import BATCH_SIZE, EPOCHS, LEARNING_RATE, DEBUG_MODE, REGRESSION_LOSS_FACTOR, \
    SPECTROGRAM_LOSS_FACTOR, PRINT_TRAIN_STATS, ARCHITECTURE, SAVE_MODEL_PATH, DATASET_MODE, DATASET_TYPE, SYNTH_TYPE, \
    SPECTROGRAM_LOSS_TYPE, LOAD_MODEL_PATH, USE_LOADED_MODEL, FREQ_PARAM_LOSS_TYPE, FREQ_MSE_LOSS_FACTOR, \
    LOG_SPECTROGRAM_MSE_LOSS, ONLY_OSC_DATASET, CNN_NETWORK, TRANSFORM, MODEL_FREQUENCY_OUTPUT, \
    NUM_EPOCHS_TO_PRINT_STATS, PRINT_ACCURACY_STATS_MULTIPLE_EPOCHS
from ai_synth_dataset import AiSynthDataset, AiSynthSingleOscDataset
from config import TRAIN_PARAMETERS_FILE, TRAIN_AUDIO_DIR, OS, PLOT_SPEC
from synth_model import SmallSynthNetwork, BigSynthNetwork
from sound_generator import SynthBasicFlow, SynthOscOnly
from synth_config import OSC_FREQ_LIST, OSC_FREQ_DIC_INV
import synth
import helper
import time
import os


def train_single_epoch(model, data_loader, transform, optimizer_arg, device_arg):

    # Initializations
    criterion_spectrogram = nn.MSELoss()
    if FREQ_PARAM_LOSS_TYPE == 'CE':
        criterion_osc_freq = nn.CrossEntropyLoss()
    elif FREQ_PARAM_LOSS_TYPE == 'MSE':
        criterion_osc_freq = nn.MSELoss()
    normalizer = helper.Normalizer()
    torch.autograd.set_detect_anomaly(True)

    sum_epoch_loss = 0
    sum_epoch_accuracy = 0
    sum_stats = []
    for i in range(len(OSC_FREQ_LIST)):
        dict_record = {
            'frequency_id': i,
            'frequency(Hz)': round(OSC_FREQ_LIST[i], 3),
            'prediction_success': 0,
            'predicted_frequency': 0,
            'frequency_model_output': 0
        }
        sum_stats.append(dict_record)
    num_of_mini_batches = 0

    for transformed_signal, target_params_dic in data_loader:
        start = time.time()

        # set_to_none as advised in page 6:
        # https://tigress-web.princeton.edu/~jdh4/PyTorchPerformanceTuningGuide_GTC2021.pdf
        model.zero_grad(set_to_none=True)

        # -------------------------------------
        # -----------Run Model-----------------
        # -------------------------------------
        # signal_log_mel_spec.requires_grad = True

        if SYNTH_TYPE == 'OSC_ONLY':
            output_dic, osc_logits_pred = model(transformed_signal)
        else:
            output_dic = model(transformed_signal)

        # helper.map_classification_params_from_ints(predicted_dic)

        osc_target_id = target_params_dic['classification_params']['osc1_freq']
        osc_target = torch.tensor([OSC_FREQ_DIC_INV[x.item()] for x in osc_target_id], device=device_arg)
        osc_pred = output_dic['osc1_freq']

        if ARCHITECTURE == 'SPEC_NO_SYNTH':
            if SPECTROGRAM_LOSS_TYPE == 'MSE':
                transformed_signal = torch.squeeze(transformed_signal)
                spectrogram_mse_loss = criterion_spectrogram(osc_pred, transformed_signal)
                loss = SPECTROGRAM_LOSS_FACTOR * spectrogram_mse_loss

        elif ARCHITECTURE == 'PARAMETERS_ONLY':
            if FREQ_PARAM_LOSS_TYPE == 'CE':
                ce_loss = criterion_osc_freq(osc_logits_pred, osc_target_id)
                loss = ce_loss

                # Sanity check - generate synth sounds with the resulted frequencies and compare spectrograms
                freq_ints_dict = {'osc1_freq': osc_logits_pred.argmax(1)}
                freq_hz_dict = helper.map_classification_params_from_ints(freq_ints_dict)

                overall_synth_obj = SynthOscOnly(parameters_dict=freq_hz_dict, num_sounds=len(transformed_signal))

                overall_synth_obj.signal = helper.move_to(overall_synth_obj.signal, device_arg)

                predicted_transformed_signal = transform(overall_synth_obj.signal)
                predicted_transformed_signal = helper.move_to(predicted_transformed_signal, device_arg)

                transformed_signal = torch.squeeze(transformed_signal)
                predicted_transformed_signal = torch.squeeze(predicted_transformed_signal)

                spectrogram_mse_loss = criterion_spectrogram(predicted_transformed_signal, transformed_signal)

                if PLOT_SPEC:
                    for i in range(49):
                        helper.plot_spectrogram(transformed_signal[i].cpu(),
                                                title=f"True MelSpectrogram (dB) for frequency {osc_target[i]}",
                                                ylabel='mel freq')

                        helper.plot_spectrogram(predicted_transformed_signal[i].cpu().detach().numpy(),
                                                title=f"Predicted MelSpectrogram (dB) {osc_pred[i]}",
                                                ylabel='mel freq')

                    # for i in range(len(signal_mel_spec)):
                    #     print("Spec diff:", criterion_spectrogram(predicted_mel_spec_sound_signal[i], signal_mel_spec[i]).item())

                    print("Spec diff of graphs:",
                          criterion_spectrogram(predicted_transformed_signal[0], transformed_signal[0]).item())

                    print("Spec diff of graphs:",
                          criterion_spectrogram(torch.log(predicted_transformed_signal[0] + 1),
                                                torch.log(transformed_signal[0] + 1)).item())

            elif FREQ_PARAM_LOSS_TYPE == 'MSE':
                frequency_mse_loss = criterion_osc_freq(osc_pred, osc_target)

                # criterion = helper.RMSLELoss()
                # frequency_mse_loss = criterion(osc_pred, osc_target)

                # DEBUG print
                # for i in range(len(osc_target)):
                #     print(f"Predicted freq: {osc_pred[i].item()}, True freq: {osc_target[i].item()}")

                loss = FREQ_MSE_LOSS_FACTOR * frequency_mse_loss
            else:
                raise ValueError("Provided FREQ_PARAM_LOSS_TYPE is not recognized")

        # if MODEL_FREQUENCY_OUTPUT == 'SINGLE', Spectrograms are compared just as accuracy measure, without training
        if (ARCHITECTURE == 'FULL' or ARCHITECTURE == 'SPECTROGRAM_ONLY') or MODEL_FREQUENCY_OUTPUT == 'SINGLE' or MODEL_FREQUENCY_OUTPUT == 'WEIGHTED':

            denormalized_output_dic = normalizer.denormalize(output_dic)
            denormalized_output_dic['osc1_freq'] = torch.clamp(denormalized_output_dic['osc1_freq'], min=0, max=20000)

            if SYNTH_TYPE == 'OSC_ONLY':
                param_dict_to_synth = denormalized_output_dic

                # -------------------------------------
                # -----------Run Synth-----------------
                # -------------------------------------
                overall_synth_obj = SynthOscOnly(parameters_dict=param_dict_to_synth,
                                                 num_sounds=len(transformed_signal))

            elif SYNTH_TYPE == 'SYNTH_BASIC':
                param_dict_to_synth = helper.clamp_regression_params(denormalized_output_dic)

                # -------------------------------------
                # -----------Run Synth-----------------
                # -------------------------------------
                overall_synth_obj = SynthBasicFlow(parameters_dict=param_dict_to_synth,
                                                   num_sounds=len(transformed_signal))

            else:
                raise ValueError("Provided SYNTH_TYPE is not recognized")

            overall_synth_obj.signal = helper.move_to(overall_synth_obj.signal, device_arg)

            predicted_transformed_signal = transform(overall_synth_obj.signal)
            predicted_transformed_signal = helper.move_to(predicted_transformed_signal, device_arg)

            transformed_signal = torch.squeeze(transformed_signal)
            predicted_transformed_signal = torch.squeeze(predicted_transformed_signal)

            if PLOT_SPEC:
                for i in range(20, 49):
                    helper.plot_spectrogram(transformed_signal[i].cpu(),
                                            title=f"True MelSpectrogram (dB) for frequency {osc_target[i]}",
                                            ylabel='mel freq')

                    helper.plot_spectrogram(predicted_transformed_signal[i].cpu().detach().numpy(),
                                            title=f"Predicted MelSpectrogram (dB) {osc_pred[i]}",
                                            ylabel='mel freq')

            if LOG_SPECTROGRAM_MSE_LOSS:
                spectrogram_mse_loss = criterion_spectrogram(torch.log(predicted_transformed_signal+1),
                                                             torch.log(transformed_signal + 1))
            else:
                spectrogram_mse_loss = criterion_spectrogram(predicted_transformed_signal, transformed_signal)

            lsd_loss = helper.lsd_loss(transformed_signal, predicted_transformed_signal)

            kl_loss = helper.kullback_leibler(predicted_transformed_signal, transformed_signal)

            # todo: remove the individual synth inference code
            # -----------------------------------------------
            # -----------Compute sound individually----------
            # -----------------------------------------------

            if False:
                loss_spectrogram_total = 0
                current_predicted_dic = {}
                for i in range(len(transformed_signal)):
                    for key, value in output_dic.items():
                        current_predicted_dic[key] = output_dic[key][i]

                    # Generate sound from predicted parameters
                    synth_obj = SynthBasicFlow(parameters_dict=current_predicted_dic)

                    synth_obj.signal = helper.move_to(synth_obj.signal, device_arg)
                    predicted_log_mel_spec_sound_signal = transform(synth_obj.signal)

                    predicted_log_mel_spec_sound_signal = helper.move_to(predicted_log_mel_spec_sound_signal,
                                                                         device_arg)

                    transformed_signal = torch.squeeze(transformed_signal)
                    predicted_log_mel_spec_sound_signal = torch.squeeze(predicted_log_mel_spec_sound_signal)
                    target_log_mel_spec_sound_signal = transformed_signal[i]
                    current_loss_spectrogram = criterion_spectrogram(predicted_log_mel_spec_sound_signal,
                                                                     target_log_mel_spec_sound_signal)
                    loss_spectrogram_total = loss_spectrogram_total + current_loss_spectrogram

                loss_spectrogram_total = loss_spectrogram_total / len(transformed_signal)

            # -----------------------------------------------
            #          -----------End Part----------
            # -----------------------------------------------

            if ARCHITECTURE == "FULL":
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

            elif ARCHITECTURE == 'SPECTROGRAM_ONLY':
                if SPECTROGRAM_LOSS_TYPE == 'MSE':
                    loss = SPECTROGRAM_LOSS_FACTOR * spectrogram_mse_loss
                elif SPECTROGRAM_LOSS_TYPE == 'LSD':
                    loss = SPECTROGRAM_LOSS_FACTOR * lsd_loss
                elif SPECTROGRAM_LOSS_TYPE == 'KL':
                    loss = kl_loss
                else:
                    raise ValueError("Unknown LOSS_TYPE")
                # loss.requires_grad = True

        num_of_mini_batches += 1
        sum_epoch_loss += loss.item()
        loss.backward()
        optimizer_arg.step()

        # Check Accuracy
        if ARCHITECTURE == 'SPEC_NO_SYNTH' or \
                (ARCHITECTURE == 'PARAMETERS_ONLY' and MODEL_FREQUENCY_OUTPUT == 'LOGITS'):
            correct = 0
            correct += (osc_logits_pred.argmax(1) == osc_target_id).type(torch.float).sum().item()
            accuracy = correct / len(osc_logits_pred)
            sum_epoch_accuracy += accuracy

        if ARCHITECTURE == 'FULL' or ARCHITECTURE == 'SPECTROGRAM_ONLY' or \
                (ARCHITECTURE == 'PARAMETERS_ONLY' and FREQ_PARAM_LOSS_TYPE == 'MSE'):
            accuracy, stats = helper.regression_freq_accuracy(output_dic, target_params_dic, device_arg)
            sum_epoch_accuracy += accuracy
            for i in range(len(OSC_FREQ_LIST)):
                sum_stats[i]['prediction_success'] += stats[i]['prediction_success']
                sum_stats[i]['predicted_frequency'] += stats[i]['predicted_frequency']
                sum_stats[i]['frequency_model_output'] += stats[i]['frequency_model_output']
        else:
            ValueError("Unknown architecture")

        end = time.time()

        if PRINT_TRAIN_STATS:
            if ARCHITECTURE == 'FULL':
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

            elif ARCHITECTURE == 'SPECTROGRAM_ONLY' or ARCHITECTURE == 'SPEC_NO_SYNTH':
                print(
                    f"MSE loss: {round(loss.item(), 3)},\t"
                    f"accuracy: {round(accuracy * 100, 2)}%,\t"
                    f"batch processing time: {round(end - start, 2)}s")

            elif ARCHITECTURE == 'PARAMETERS_ONLY':
                print(
                    f"Frequency {FREQ_PARAM_LOSS_TYPE} loss: {round(loss.item(), 6)},\t\t"
                    f"Accuracy: {round(accuracy * 100, 2)}%,\t"
                    f"MSE of spectrograms: {round(SPECTROGRAM_LOSS_FACTOR * spectrogram_mse_loss.item(), 3)}, \t"
                    f"Batch processing time: {round(end - start, 2)}s")
            else:
                ValueError("Unknown architecture")

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
    avg_stats = sum_stats
    for i in range(len(OSC_FREQ_LIST)):
        avg_stats[i]['prediction_success'] = sum_stats[i]['prediction_success'] / num_of_mini_batches
        avg_stats[i]['predicted_frequency'] = sum_stats[i]['predicted_frequency'] / num_of_mini_batches
        avg_stats[i]['frequency_model_output'] = sum_stats[i]['frequency_model_output'] / num_of_mini_batches

    return avg_epoch_loss, avg_epoch_accuracy, avg_stats


def train(model, data_loader, transform, optimiser_arg, device_arg, cur_epoch, num_epochs):
    model.train()

    loss_list = []
    accuracy_list = []

    multi_epoch_avg_loss, multi_epoch_avg_accuracy, multi_epoch_stats = helper.reset_multi_epoch_stats()

    for i in range(num_epochs):
        if not ONLY_OSC_DATASET or i % 100 == 0:
            print(f"Epoch {cur_epoch} start")

        avg_epoch_loss, avg_epoch_accuracy, avg_epoch_stats = \
            train_single_epoch(model, data_loader, transform, optimiser_arg, device_arg)

        # Sum stats over multiple epochs
        multi_epoch_avg_loss += avg_epoch_loss
        multi_epoch_avg_accuracy += avg_epoch_accuracy
        for j in range(len(OSC_FREQ_LIST)):
            multi_epoch_stats[j]['prediction_success'] += avg_epoch_stats[j]['prediction_success']
            multi_epoch_stats[j]['predicted_frequency'] += avg_epoch_stats[j]['predicted_frequency']
            multi_epoch_stats[j]['frequency_model_output'] += avg_epoch_stats[j]['frequency_model_output']

        cur_epoch += 1

        if ONLY_OSC_DATASET and (i % NUM_EPOCHS_TO_PRINT_STATS == 0 and i != 0):

            # Average stats over multiple epochs
            multi_epoch_avg_loss /= NUM_EPOCHS_TO_PRINT_STATS
            multi_epoch_avg_accuracy /= NUM_EPOCHS_TO_PRINT_STATS
            for k in range(len(OSC_FREQ_LIST)):
                multi_epoch_stats[k]['prediction_success'] /= NUM_EPOCHS_TO_PRINT_STATS
                multi_epoch_stats[k]['predicted_frequency'] /= NUM_EPOCHS_TO_PRINT_STATS
                multi_epoch_stats[k]['frequency_model_output'] /= NUM_EPOCHS_TO_PRINT_STATS

                multi_epoch_stats[k]['prediction_success'] *= 100
                multi_epoch_stats[k]['predicted_frequency'] = round(multi_epoch_stats[k]['predicted_frequency'], 3)
                multi_epoch_stats[k]['frequency_model_output'] = round(multi_epoch_stats[k]['frequency_model_output'], 3)

            print("--------------------------------------\n")
            print(f"Epoch {cur_epoch} end")
            print(f"Average Loss for last {NUM_EPOCHS_TO_PRINT_STATS} epochs:", round(multi_epoch_avg_loss, 6))
            print(f"Average Accuracy for last {NUM_EPOCHS_TO_PRINT_STATS} epochs:"
                  f" {round(multi_epoch_avg_accuracy * 100, 2)}%\n")

            fmt = [
                ('Frequency ID', 'frequency_id', 13),
                ('Frequency (Hz)', 'frequency(Hz)', 17),
                ('AVG Prediction %', 'prediction_success', 20),
                ('AVG Predicted Frequency ', 'predicted_frequency', 20),
                ('AVG Frequency Model Out', 'frequency_model_output', 20),
            ]
            if PRINT_ACCURACY_STATS_MULTIPLE_EPOCHS:
                print(helper.TablePrinter(fmt, ul='=')(multi_epoch_stats))
            print("--------------------------------------\n")

            multi_epoch_avg_loss, multi_epoch_avg_accuracy, multi_epoch_stats = helper.reset_multi_epoch_stats()

            loss_list.append(avg_epoch_loss)
            accuracy_list.append(avg_epoch_accuracy * 100)

            # save model checkpoint
            helper.save_model(cur_epoch, model, optimiser_arg, avg_epoch_loss, loss_list, accuracy_list)

        elif not ONLY_OSC_DATASET:

            print("--------------------------------------")
            print(f"Epoch {cur_epoch} end")
            print(f"Average Epoch{cur_epoch} Loss:", round(avg_epoch_loss, 2))
            print(f"Average Epoch{cur_epoch} Accuracy: {round(avg_epoch_accuracy * 100, 2)}%")
            print("--------------------------------------\n")

            loss_list.append(avg_epoch_loss)
            accuracy_list.append(avg_epoch_accuracy * 100)

            # save model checkpoint
            helper.save_model(cur_epoch, model, optimiser_arg, avg_epoch_loss, loss_list, accuracy_list)

    print("Finished training")


if __name__ == "__main__":
    device = helper.get_device()

    if TRANSFORM == 'MEL_SPECTROGRAM':
        transform = helper.mel_spectrogram_transform
    elif TRANSFORM == 'SPECTROGRAM':
        transform = helper.spectrogram_transform
    else:
        transform = None
        ValueError(f"{TRANSFORM} is an unidentified TRANSFORM")

    if SYNTH_TYPE == 'OSC_ONLY':
        ai_synth_dataset = AiSynthSingleOscDataset(DATASET_MODE,
                                                   DATASET_TYPE,
                                                   TRAIN_PARAMETERS_FILE,
                                                   TRAIN_AUDIO_DIR,
                                                   transform,
                                                   synth.SAMPLE_RATE,
                                                   device
                                                   )
    elif SYNTH_TYPE == 'SYNTH_BASIC':
        ai_synth_dataset = AiSynthDataset(DATASET_MODE,
                                          DATASET_TYPE,
                                          TRAIN_PARAMETERS_FILE,
                                          TRAIN_AUDIO_DIR,
                                          transform,
                                          synth.SAMPLE_RATE,
                                          device
                                          )
    else:
        ai_synth_dataset = None
        ValueError(f"{SYNTH_TYPE} is an unidentified SYNTH_TYPE")

    train_dataloader = helper.create_data_loader(ai_synth_dataset, BATCH_SIZE)

    # construct model and assign it to device
    if CNN_NETWORK == 'SMALL':
        synth_net = SmallSynthNetwork().to(device)
    elif CNN_NETWORK == 'BIG':
        synth_net = BigSynthNetwork().to(device)
    else:
        synth_net = None
        ValueError(f"{CNN_NETWORK} is an unidentified CNN_NETWORK")

    # initialize optimizer
    optimizer = torch.optim.Adam(synth_net.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    print(f"Training model with: \n LOSS_MODE: {ARCHITECTURE} \n SYNTH_TYPE: {SYNTH_TYPE}")
    if USE_LOADED_MODEL:
        checkpoint = torch.load(LOAD_MODEL_PATH)
        synth_net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        cur_epoch = checkpoint['epoch']
        loss = checkpoint['loss']
    else:
        cur_epoch = 0

    # train model
    train(synth_net, train_dataloader, transform, optimizer, device, cur_epoch=cur_epoch, num_epochs=EPOCHS)

    path_parent = os.path.dirname(os.getcwd())

    # save model
    if OS == 'WINDOWS':
        saved_model_path = path_parent + "\\ai_synth\\trained_models\\trained_synth_net.pth"
    elif OS == 'LINUX':
        saved_model_path = path_parent + "/ai_synth/trained_models/trained_synth_net.pth"

    torch.save(synth_net.state_dict(), saved_model_path)
    print("Trained synth net saved at synth_net.pth")
