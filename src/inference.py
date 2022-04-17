import matplotlib
from torchaudio.transforms import Spectrogram

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from torch.utils.tensorboard import SummaryWriter

import helper
from model import BigSynthNetwork
from ai_synth_dataset import AiSynthDataset, create_data_loader
from synth.synth_architecture import SynthModular, SynthModularCell
from config import Config, SynthConfig, DatasetConfig, ModelConfig
import scipy.io.wavfile
import pandas as pd
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pathlib import Path
import librosa
import librosa.display
import numpy as np
from tqdm import tqdm

summary_writer = SummaryWriter()
spectrogram = Spectrogram(n_fft=256)


def predict(model,
            test_data_loader,
            transform,
            device_arg,
            cfg: Config,
            synth_cfg: SynthConfig,
            dataset_cfg: DatasetConfig):
    with torch.no_grad():
        modular_synth = SynthModular(synth_cfg=synth_cfg,
                                     sample_rate=cfg.sample_rate,
                                     signal_duration_sec=cfg.signal_duration_sec,
                                     num_sounds=1,
                                     device=device_arg,
                                     preset=synth_cfg.preset
                                     )

        predicted_params_list = []
        step = 0
        with tqdm(test_data_loader, unit="batch") as tepoch:
            for signals, target_params_dic, signals_indices in tepoch:
                signals = helper.move_to(signals, device_arg)
                normalizer = helper.Normalizer(cfg.signal_duration_sec, synth_cfg)

                transformed_signal = transform(signals)

                output_dic = model(transformed_signal)

                # Infer predictions
                denormalized_output_dict = normalizer.denormalize(output_dic)
                predicted_param_dict = helper.clamp_regression_params(denormalized_output_dict, synth_cfg, cfg)

                update_params = []
                for index, operation_dict in predicted_param_dict.items():
                    synth_modular_cell = SynthModularCell(index=index, parameters=operation_dict['params'])
                    update_params.append(synth_modular_cell)

                modular_synth.update_cells(update_params)
                modular_synth.generate_signal(num_sounds=len(transformed_signal))

                if cfg.spectrogram_loss_type == 'MULTI-SPECTRAL':
                    multi_spec_loss = helper.SpectralLoss(cfg=cfg,
                                                          loss_type=cfg.multi_spectral_loss_type,
                                                          mag_weight=cfg.multi_spectral_mag_weight,
                                                          delta_time_weight=cfg.multi_spectral_delta_time_weight,
                                                          delta_freq_weight=cfg.multi_spectral_delta_freq_weight,
                                                          cumsum_freq_weight=cfg.multi_spectral_cumsum_freq_weight,
                                                          logmag_weight=cfg.multi_spectral_logmag_weight,
                                                          device=device_arg)
                    signals = signals.squeeze()
                    loss = multi_spec_loss.call(signals, modular_synth.signal, summary_writer, step)
                else:
                    ValueError("SYNTH_TYPE 'MODULAR' supports only SPECTROGRAM_LOSS_TYPE of type 'MULTI-SPECTRAL'")

                for i in range(len(signals)):

                    # save synth parameters output
                    params_dict = {}
                    for layer in range(synth_cfg.num_layers):
                        for channel in range(synth_cfg.num_channels):
                            cell = modular_synth.architecture[channel][layer]
                            if cell.operation is not None:
                                operation = cell.operation
                            else:
                                operation = 'None'
                            if cell.parameters is not None:
                                parameters = cell.parameters
                                parameters_of_current_signal = {}
                                for key in parameters:
                                    if parameters[key].shape[1] <= 1:
                                        a = parameters[key][i].item()
                                        parameters_of_current_signal[key] = a
                                    else:
                                        if key == 'waveform':
                                            waveform = synth_cfg.wave_type_dic_inv[parameters[key][i].argmax().item()]
                                            parameters_of_current_signal[key] = waveform
                                        if key == 'filter_type':
                                            filter_type = synth_cfg.filter_type_dic_inv[
                                                parameters[key][i].argmax().item()]
                                            parameters_of_current_signal[key] = filter_type

                            else:
                                parameters_of_current_signal = 'None'
                            params_dict[cell.index] = {'operation': operation,
                                                       'parameters': parameters_of_current_signal}
                    predicted_params_list.append(params_dict)

                    signal_index = signals_indices[i]

                    orig_audio = signals[i]
                    pred_audio = modular_synth.signal[i]
                    orig_audio_np = orig_audio.detach().cpu().numpy()
                    pred_audio_np = pred_audio.detach().cpu().numpy()

                    orig_audio_transformed = librosa.feature.melspectrogram(y=orig_audio_np,
                                                                            sr=cfg.sample_rate,
                                                                            n_fft=1024,
                                                                            hop_length=512,
                                                                            n_mels=64)
                    orig_audio_transformed_db = librosa.power_to_db(orig_audio_transformed, ref=np.max)
                    pred_audio_transformed = librosa.feature.melspectrogram(y=pred_audio_np,
                                                                            sr=cfg.sample_rate,
                                                                            n_fft=1024,
                                                                            hop_length=512,
                                                                            n_mels=64)
                    pred_audio_transformed_db = librosa.power_to_db(pred_audio_transformed, ref=np.max)

                    # save synth audio signal output
                    pred_file_name = f"sound_{signal_index}_pred.wav"
                    pred_audio_path = dataset_cfg.inference_audio_dir.joinpath(pred_file_name)
                    scipy.io.wavfile.write(pred_audio_path, cfg.sample_rate, pred_audio_np)

                    # plot original vs predicted signal
                    plt.figure(figsize=[30, 20])
                    plt.ion()
                    plt.subplot(2, 2, 1)
                    plt.title(f"original audio")
                    plt.ylim([-1, 1])
                    plt.plot(orig_audio_np)
                    plt.subplot(2, 2, 2)
                    plt.ylim([-1, 1])
                    plt.title("predicted audio")
                    plt.plot(pred_audio_np)
                    plt.subplot(2, 2, 3)
                    librosa.display.specshow(orig_audio_transformed_db, sr=cfg.sample_rate, hop_length=512,
                                             x_axis='time', y_axis='mel')
                    plt.colorbar(format='%+2.0f dB')
                    plt.subplot(2, 2, 4)
                    librosa.display.specshow(pred_audio_transformed_db, sr=cfg.sample_rate, hop_length=512,
                                             x_axis='time', y_axis='mel')
                    plt.colorbar(format='%+2.0f dB')
                    plt.ioff()
                    plots_path = dataset_cfg.inference_plots_dir.joinpath(f"sound{signal_index}_plots.png")
                    plt.savefig(plots_path)

                    step += 1

        predicted_params_dataframe = pd.DataFrame(predicted_params_list)
        dataset_dir_path = Path(__file__).parent.parent.joinpath('dataset', 'test')
        parameters_pickle_path = dataset_dir_path.joinpath("predicted_params_dataset.pkl")
        parameters_csv_path = dataset_dir_path.joinpath("predicted_params_dataset.csv")
        predicted_params_dataframe.to_pickle(str(parameters_pickle_path))
        predicted_params_dataframe.to_csv(parameters_csv_path)


def visualize_signal_prediction(orig_audio, pred_audio, orig_params, pred_params,
                                db=False):

    orig_audio_np = orig_audio.detach().cpu().numpy()
    pred_audio_np = pred_audio.detach().cpu().numpy()

    orig_spectograms = [spectrogram(orig_audio.cpu())]
    pred_spectograms = [spectrogram(pred_audio.cpu())]

    orig_spectograms_np = [orig_spectogram.detach().cpu().numpy() for orig_spectogram in orig_spectograms]
    pred_spectograms_np = [pred_spectogram.detach().cpu().numpy() for pred_spectogram in pred_spectograms]

    if db:
        orig_spectograms_np = [librosa.power_to_db(orig_spectogram_np) for orig_spectogram_np in orig_spectograms_np]
        pred_spectograms_np = [librosa.power_to_db(pred_spectogram_np) for pred_spectogram_np in pred_spectograms_np]

    # plot original vs predicted signal
    n_rows = len(orig_spectograms_np) + 1
    fig, ax = plt.subplots(n_rows, 2, figsize=(15, 10))

    canvas = FigureCanvasAgg(fig)

    orig_params_str = '\n'.join([f'{k}: {v}' for k, v in orig_params.items()])
    ax[0][0].set_title(f"original audio\n{orig_params_str}", fontsize=10)
    ax[0][0].set_ylim([-1, 1])
    ax[0][0].plot(orig_audio_np)

    pred_params_str = '\n'.join([f'{k}: {v}' for k, v in pred_params.items()])
    ax[0][1].set_title(f"predicted audio\n{pred_params_str}", fontsize=10)
    ax[0][1].set_ylim([-1, 1])
    ax[0][1].plot(pred_audio_np)

    for i in range(1, n_rows):
        ax[i][0].imshow(orig_spectograms_np[i-1], origin='lower', aspect='auto')
        ax[i][1].imshow(pred_spectograms_np[i-1], origin='lower', aspect='auto')

    canvas.draw()
    s, (width, height) = canvas.print_to_buffer()

    # Option 2a: Convert to a NumPy array.
    X = np.fromstring(s, np.uint8).reshape((height, width, 4))[:, :, :3]

    plt.close('all')

    return X


def run():
    cfg = Config()
    synth_cfg = SynthConfig()
    dataset_cfg = DatasetConfig()
    device = helper.get_device()

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            description='Train AI Synth')
    parser.add_argument('-g', '--gpu_index', help='index of gpu (if exist, torch indexing) -1 for cpu',
                        type=int, default=0)
    parser.add_argument('-t', '--transform', choices=['mel', 'spec'],
                        help='mel: Mel Spectrogram, spec: Spectrogram', default='mel')
    args = parser.parse_args()

    transforms = {'mel': helper.mel_spectrogram_transform(cfg.sample_rate).to(device),
                  'spec': helper.spectrogram_transform().to(device)}
    transform = transforms[args.transform]

    # load back the model
    synth_net = BigSynthNetwork(synth_cfg=synth_cfg, device=device).to(device)
    checkpoint = torch.load(Config.load_model_path)
    synth_net.load_state_dict(checkpoint['model_state_dict'])

    synth_net.eval()

    # import test dataset
    ai_synth_test_dataset = AiSynthDataset(DatasetConfig.test_parameters_file,
                                           DatasetConfig.test_audio_dir,
                                           device)
    test_dataloader = create_data_loader(ai_synth_test_dataset, ModelConfig.batch_size, ModelConfig.num_workers)

    predict(model=synth_net,
            test_data_loader=test_dataloader,
            transform=transform,
            device_arg=device,
            cfg=cfg,
            synth_cfg=synth_cfg,
            dataset_cfg=dataset_cfg,)


if __name__ == "__main__":
    run()
