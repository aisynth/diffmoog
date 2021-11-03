import torch
import helper
from torch import nn
from synth_model import SynthNetwork
from ai_synth_dataset import AiSynthDataset
from sound_generator import SynthBasicFlow
from config import TEST_PARAMETERS_FILE, TEST_AUDIO_DIR, BATCH_SIZE, LEARNING_RATE, LOAD_MODEL_PATH, \
    REGRESSION_LOSS_FACTOR, SPECTROGRAM_LOSS_FACTOR, PRINT_TRAIN_STATS, OS
import synth
import os
import scipy.io.wavfile
import pandas as pd


def predict(model, test_data_loader, device_arg):
    with torch.no_grad():
        for signal_mel_spec, target_params_dic in test_data_loader:
            batch_size = signal_mel_spec.shape[0]
            signal_log_mel_spec = helper.amplitude_to_db_transform(signal_mel_spec)
            classification_target_params = target_params_dic['classification_params']
            regression_target_parameters = target_params_dic['regression_params']
            translated_classification_target_params = classification_target_params.copy()
            translated_regression_target_parameters = regression_target_parameters.copy()
            helper.map_classification_params_to_ints(translated_classification_target_params)
            normalizer = helper.Normalizer()
            normalizer.normalize(translated_regression_target_parameters)

            translated_classification_target_params = helper.move_to(translated_classification_target_params, device_arg)
            translated_regression_target_parameters = helper.move_to(translated_regression_target_parameters, device_arg)
            signal_log_mel_spec = helper.move_to(signal_log_mel_spec, device_arg)

            output_dic = model(signal_log_mel_spec)

            # Infer predictions
            predicted_dic = {}
            for param in synth.CLASSIFICATION_PARAM_LIST:
                predicted_dic[param] = torch.argmax(output_dic[param], dim=1)
            for index, param in enumerate(synth.REGRESSION_PARAM_LIST):
                predicted_dic[param] = output_dic['regression_params'][:, index]

            predicted_dic = helper.map_classification_params_from_ints(predicted_dic)

            normalizer.denormalize(predicted_dic)
            helper.clamp_regression_params(predicted_dic)

            # Init criteria
            criterion_osc1_freq = criterion_osc1_wave = criterion_lfo1_wave \
                = criterion_osc2_freq = criterion_osc2_wave = criterion_lfo2_wave \
                = criterion_filter_type \
                = nn.CrossEntropyLoss()
            criterion_regression_params = criterion_spectrogram = nn.MSELoss()

            loss_spectrogram_total = 0
            current_predicted_dic = {}
            predicted_params_list = []

            # todo: refactor code. try to implement SynthBasicFlow in matrix, to prevent for loop.
            #  compute all spectrogram loss all at once using mean function
            for i in range(batch_size):

                # todo: include the index of audio file in the dataset parameters
                # Infer original audio from
                original_dic = {}
                for param in synth.CLASSIFICATION_PARAM_LIST:
                    original_dic[param] = target_params_dic['classification_params'][param][i]
                for index, param in enumerate(synth.REGRESSION_PARAM_LIST):
                    original_dic[param] = target_params_dic['regression_params'][param][i]

                for key, value in original_dic.items():
                    if torch.is_tensor(original_dic[key]):
                        original_dic[key] = original_dic[key].item()
                    else:
                        original_dic[key] = original_dic[key]

                print(original_dic)

                for key, value in predicted_dic.items():
                    if torch.is_tensor(predicted_dic[key][i]):
                        current_predicted_dic[key] = predicted_dic[key][i].item()
                    else:
                        current_predicted_dic[key] = predicted_dic[key][i]
                print(current_predicted_dic)

                print("------original parameters--------")
                print('osc1_freq: ', classification_target_params['osc1_freq'][i].item())
                print('osc1_wave: ', classification_target_params['osc1_wave'][i])
                print('lfo1_wave: ', classification_target_params['lfo1_wave'][i])
                print('osc2_freq: ', classification_target_params['osc2_freq'][i].item())
                print('osc2_wave: ', classification_target_params['osc2_wave'][i])
                print('lfo2_wave: ', classification_target_params['lfo2_wave'][i])
                print('osc1_amp: ', regression_target_parameters['osc1_amp'][i].item())
                print('osc1_mod_index: ', regression_target_parameters['osc1_mod_index'][i].item())
                print('lfo1_freq: ', regression_target_parameters['lfo1_freq'][i].item())
                print('lfo1_phase: ', regression_target_parameters['lfo1_phase'][i].item())
                print('osc2_amp: ', regression_target_parameters['osc2_amp'][i].item())
                print('osc2_mod_index: ', regression_target_parameters['osc2_mod_index'][i].item())
                print('lfo2_freq: ', regression_target_parameters['lfo2_freq'][i].item())
                print('lfo2_phase: ', regression_target_parameters['lfo2_phase'][i].item())
                print('filter_freq: ', regression_target_parameters['filter_freq'][i].item())
                print('attack_t: ', regression_target_parameters['attack_t'][i].item())
                print('decay_t: ', regression_target_parameters['decay_t'][i].item())
                print('sustain_t: ', regression_target_parameters['sustain_t'][i].item())
                print('release_t: ', regression_target_parameters['release_t'][i].item())
                print('sustain_level: ', regression_target_parameters['sustain_level'][i].item())

                print("\n")
                print("------predictions--------")
                print('osc1_freq: ', current_predicted_dic['osc1_freq'])
                print('osc1_wave: ', current_predicted_dic['osc1_wave'])
                print('lfo1_wave: ', current_predicted_dic['lfo1_wave'])
                print('osc2_freq: ', current_predicted_dic['osc2_freq'])
                print('osc2_wave: ', current_predicted_dic['osc2_wave'])
                print('lfo2_wave: ', current_predicted_dic['lfo2_wave'])
                print('osc1_amp: ', current_predicted_dic['osc1_amp'])
                print('osc1_mod_index: ', current_predicted_dic['osc1_mod_index'])
                print('lfo1_freq: ', current_predicted_dic['lfo1_freq'])
                print('lfo1_phase: ', current_predicted_dic['lfo1_phase'])
                print('osc2_amp: ', current_predicted_dic['osc2_amp'])
                print('osc2_mod_index: ', current_predicted_dic['osc2_mod_index'])
                print('lfo2_freq: ', current_predicted_dic['lfo2_freq'])
                print('lfo2_phase: ', current_predicted_dic['lfo2_phase'])
                print('filter_freq: ', current_predicted_dic['filter_freq'])
                print('attack_t: ', current_predicted_dic['attack_t'])
                print('decay_t: ', current_predicted_dic['decay_t'])
                print('sustain_t: ', current_predicted_dic['sustain_t'])
                print('release_t: ', current_predicted_dic['release_t'])
                print('sustain_level: ', current_predicted_dic['sustain_level'])
                print("\n")

                orig_synth_obj = SynthBasicFlow(parameters_dict=original_dic)
                pred_synth_obj = SynthBasicFlow(parameters_dict=current_predicted_dic)

                # save synth audio signal output
                orig_audio = orig_synth_obj.signal
                pred_audio = pred_synth_obj.signal
                orig_file_name = f"sound{i}_orig.wav"
                pred_file_name = f"sound{i}_pred.wav"
                path_parent = os.path.dirname(os.getcwd())
                if OS == 'WINDOWS':
                    orig_audio_path = path_parent + f"\\dataset\\test\\inference_wav_files\\original\\{orig_file_name}"
                    pred_audio_path = path_parent + f"\\dataset\\test\\inference_wav_files\\predicted\\{pred_file_name}"
                elif OS == 'LINUX':
                    orig_audio_path = path_parent + f"/dataset/test/inference_wav_files/original/{orig_file_name}"
                    pred_audio_path = path_parent + f"/dataset/test/inference_wav_files/predicted/{pred_file_name}"
                orig_audio = orig_audio.detach().cpu().numpy()
                pred_audio = pred_audio.detach().cpu().numpy()
                scipy.io.wavfile.write(orig_audio_path, 44100, orig_audio)
                scipy.io.wavfile.write(pred_audio_path, 44100, pred_audio)

                # save synth parameters output
                predicted_parameters = pred_synth_obj.params_dict
                predicted_params_list.append(predicted_parameters)

                predicted_mel_spec_sound_signal = helper.mel_spectrogram_transform(pred_synth_obj.signal)
                predicted_log_mel_spec_sound_signal = helper.amplitude_to_db_transform(predicted_mel_spec_sound_signal)

                predicted_log_mel_spec_sound_signal = helper.move_to(predicted_log_mel_spec_sound_signal, device_arg)
                signal_log_mel_spec = torch.squeeze(signal_log_mel_spec)

                current_loss_spectrogram = criterion_spectrogram(predicted_log_mel_spec_sound_signal,
                                                                 signal_log_mel_spec[i])
                loss_spectrogram_total = loss_spectrogram_total + current_loss_spectrogram

            dataframe = pd.DataFrame(predicted_params_list)
            path_parent = os.path.dirname(os.getcwd())
            if OS == 'WINDOWS':
                parameters_path = path_parent + "\\dataset\\test\\inference_wav_files\\dataset.csv"
            elif OS == 'LINUX':
                parameters_path = path_parent + "/dataset/test/inference_wav_files/dataset.csv"
            dataframe.to_csv(parameters_path)

            loss_spectrogram_total = loss_spectrogram_total / batch_size

            loss_osc1_freq = criterion_osc1_freq(output_dic['osc1_freq'], translated_classification_target_params['osc1_freq'])
            loss_osc1_wave = criterion_osc1_wave(output_dic['osc1_wave'], translated_classification_target_params['osc1_wave'])
            loss_lfo1_wave = criterion_lfo1_wave(output_dic['lfo1_wave'], translated_classification_target_params['lfo1_wave'])
            loss_osc2_freq = criterion_osc2_freq(output_dic['osc2_freq'], translated_classification_target_params['osc2_freq'])
            loss_osc2_wave = criterion_osc2_wave(output_dic['osc2_wave'], translated_classification_target_params['osc2_wave'])
            loss_lfo2_wave = criterion_lfo2_wave(output_dic['lfo2_wave'], translated_classification_target_params['lfo2_wave'])
            loss_filter_type = \
                criterion_filter_type(output_dic['filter_type'], translated_classification_target_params['filter_type'])

            # todo: refactor code. the code gets dictionary of tensors (regression_target_parameters) and return 2d tensor
            regression_target_parameters_tensor = torch.empty((len(translated_regression_target_parameters['osc1_amp']), 1))
            regression_target_parameters_tensor = helper.move_to(regression_target_parameters_tensor, device_arg)
            for key, value in translated_regression_target_parameters.items():
                regression_target_parameters_tensor = \
                    torch.cat([regression_target_parameters_tensor, translated_regression_target_parameters[key].unsqueeze(dim=1)],
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

            if PRINT_TRAIN_STATS:
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


if __name__ == "__main__":
    # load back the model
    device = helper.get_device()
    synth_net = SynthNetwork().to(device)
    optimizer = torch.optim.Adam(synth_net.parameters(), lr=LEARNING_RATE)

    checkpoint = torch.load(LOAD_MODEL_PATH)
    synth_net.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    print(loss)

    synth_net.eval()

    # import test dataset
    ai_synth_test_dataset = AiSynthDataset(TEST_PARAMETERS_FILE,
                                           TEST_AUDIO_DIR,
                                           helper.mel_spectrogram_transform,
                                           synth.SAMPLE_RATE,
                                           device)

    test_dataloader = helper.create_data_loader(ai_synth_test_dataset, BATCH_SIZE)

    predict(synth_net, test_dataloader, device)

