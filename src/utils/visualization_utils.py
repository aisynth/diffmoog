import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

import numpy as np
from torchaudio.transforms import Spectrogram, MelSpectrogram, AmplitudeToDB

matplotlib.use('Agg')


spectrogram = Spectrogram(n_fft=1024)
amp_to_db = AmplitudeToDB()


def visualize_signal_prediction(orig_audio, pred_audio, orig_params, pred_params,
                                db=False):
    orig_audio_np = orig_audio.detach().cpu().numpy()
    pred_audio_np = pred_audio.detach().cpu().numpy()

    if db:
        orig_spectrograms = [amp_to_db(spectrogram(orig_audio.cpu()))]
        pred_spectrograms = [amp_to_db(spectrogram(pred_audio.cpu()))]
    else:
        orig_spectrograms = [spectrogram(orig_audio.cpu())]
        pred_spectrograms = [spectrogram(pred_audio.cpu())]

    orig_spectrograms_np = [orig_spectrogram.detach().cpu().numpy() for orig_spectrogram in orig_spectrograms]
    pred_spectrograms_np = [pred_spectrogram.detach().cpu().numpy() for pred_spectrogram in pred_spectrograms]

    # plot original vs predicted signal
    n_rows = len(orig_spectrograms_np) + 1
    fig, ax = plt.subplots(n_rows, 2, figsize=(20, 12))

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
        ax[i][0].imshow(orig_spectrograms_np[i - 1], origin='lower', aspect='auto')
        ax[i][1].imshow(pred_spectrograms_np[i - 1], origin='lower', aspect='auto')

    canvas.draw()
    s, (width, height) = canvas.print_to_buffer()

    # Option 2a: Convert to a NumPy array.
    X = np.fromstring(s, np.uint8).reshape((height, width, 4))[:, :, :3]

    plt.close('all')

    return X