import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader
from config import BATCH_SIZE, EPOCHS, LEARNING_RATE, DEBUG_MODE
from ai_synth_dataset import AiSynthDataset
from config import SAMPLE_RATE, PARAMETERS_FILE, AUDIO_DIR
from synth_model import SynthNetwork
from sound_generator import SynthBasicFlow


def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    return train_dataloader


def train_single_epoch(model, data_loader, optimiser, device):
    for audio_mel_spectrogram, classification_target_params, regression_target_parameters in data_loader:

        audio_mel_spectrogram = move_to(audio_mel_spectrogram, device)
        classification_target_params = move_to(classification_target_params, device)
        regression_target_parameters = move_to(regression_target_parameters, device)

        # calculate loss
        # todo: force the model to predict values from defined ranges
        predicted_params = model(audio_mel_spectrogram)
        # todo: refactor code. extract predicted params
        predicted_dic = {
            'osc1_freq': torch.argmax(predicted_params['osc1_freq'], dim=1),
            'osc1_wave': torch.argmax(predicted_params['osc1_wave'], dim=1),
            'lfo1_wave': torch.argmax(predicted_params['lfo1_wave'], dim=1),
            'osc2_freq': torch.argmax(predicted_params['osc1_freq'], dim=1),
            'osc2_wave': torch.argmax(predicted_params['osc1_wave'], dim=1),
            'lfo2_wave': torch.argmax(predicted_params['lfo1_wave'], dim=1),
            'filter_type': torch.argmax(predicted_params['filter_type'], dim=1),
            'osc1_amp': predicted_params['regression_params'][:, 0],
            'osc1_mod_index': predicted_params['regression_params'][:, 1],
            'lfo1_freq': predicted_params['regression_params'][:, 2],
            'lfo1_phase': predicted_params['regression_params'][:, 3],
            'osc2_amp': predicted_params['regression_params'][:, 4],
            'osc2_mod_index': predicted_params['regression_params'][:, 5],
            'lfo2_freq': predicted_params['regression_params'][:, 6],
            'lfo2_phase': predicted_params['regression_params'][:, 7],
            'filter_freq': predicted_params['regression_params'][:, 8],
            'attack_t': predicted_params['regression_params'][:, 9],
            'decay_t': predicted_params['regression_params'][:, 10],
            'sustain_t': predicted_params['regression_params'][:, 11],
            'release_t': predicted_params['regression_params'][:, 12],
            'sustain_level': predicted_params['regression_params'][:, 13]
        }
        criterion_spectrogram = nn.MSELoss()
        loss_spectrogram = 0
        current_predicted_dic = {}
        # todo: refactor code. try to implement SynthBasicFlow in matrix, to prevent for loop. redefine range
        #  with general batch size. compute all spectrogram loss all at once using mean function
        for i in range(predicted_params['osc1_freq'].shape[0]):
            for key, value in predicted_dic.items():
                current_predicted_dic[key] = predicted_dic[key][i]
            synth_obj = SynthBasicFlow(current_predicted_dic)
            # todo: refactor code. define this transformation in config or other general file
            mel_spectrogram = torchaudio.transforms.MelSpectrogram(
                sample_rate=SAMPLE_RATE,
                n_fft=1024,
                hop_length=512,
                n_mels=64
            )
            predicted_mel_spec_sound_signal = mel_spectrogram(synth_obj.audio)
            predicted_mel_spec_sound_signal = move_to(predicted_mel_spec_sound_signal, device)
            # todo: refactor code. use unsqueeze instead of '0'.
            current_loss_spectrogram = criterion_spectrogram(predicted_mel_spec_sound_signal, audio_mel_spectrogram[i][0])
            loss_spectrogram = loss_spectrogram + current_loss_spectrogram

        loss_spectrogram = loss_spectrogram / predicted_params['osc1_freq'].shape[0]

        # todo: use for loop, prevent code duplication
        loss_osc1_freq = nn.CrossEntropyLoss()
        loss_osc1_wave = nn.CrossEntropyLoss()
        loss_lfo1_wave = nn.CrossEntropyLoss()
        loss_osc2_freq = nn.CrossEntropyLoss()
        loss_osc2_wave = nn.CrossEntropyLoss()
        loss_lfo2_wave = nn.CrossEntropyLoss()
        loss_filter_type = nn.CrossEntropyLoss()
        loss_regression_params = nn.MSELoss()

        loss_osc1_freq = loss_osc1_freq(predicted_params['osc1_freq'], classification_target_params['osc1_freq'])
        loss_osc1_wave = loss_osc1_wave(predicted_params['osc1_wave'], classification_target_params['osc1_wave'])
        loss_lfo1_wave = loss_lfo1_wave(predicted_params['lfo1_wave'], classification_target_params['lfo1_wave'])
        loss_osc2_freq = loss_osc2_freq(predicted_params['osc2_freq'], classification_target_params['osc2_freq'])
        loss_osc2_wave = loss_osc2_wave(predicted_params['osc2_wave'], classification_target_params['osc2_wave'])
        loss_lfo2_wave = loss_lfo2_wave(predicted_params['lfo2_wave'], classification_target_params['lfo2_wave'])
        loss_filter_type = \
            loss_filter_type(predicted_params['filter_type'], classification_target_params['filter_type'])

        # todo: refactor code. the code gets dictionary of tensors (regression_target_parameters) and return 2d tensor
        regression_target_parameters_tensor = torch.empty((len(regression_target_parameters['osc1_amp']), 1))
        regression_target_parameters_tensor = move_to(regression_target_parameters_tensor, device)
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
            loss_regression_params(predicted_params['regression_params'], regression_target_parameters_tensor)

        loss = loss_classification_params + loss_regression_params + loss_spectrogram

        # backpropogate error and update wights
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    print(f"loss: {loss.item()}")
    if DEBUG_MODE:
        print("osc1_freq",
              torch.argmax(predicted_params['osc1_freq'], dim=1), classification_target_params['osc1_freq'])
        print("osc1_wave",
              torch.argmax(predicted_params['osc1_wave'], dim=1), classification_target_params['osc1_wave'])
        print("lfo1_wave",
              torch.argmax(predicted_params['lfo1_wave'], dim=1), classification_target_params['lfo1_wave'])
        print("osc2_freq",
              torch.argmax(predicted_params['osc2_freq'], dim=1), classification_target_params['osc2_freq'])
        print("osc2_wave",
              torch.argmax(predicted_params['osc2_wave'], dim=1), classification_target_params['osc2_wave'])
        print("lfo2_wave",
              torch.argmax(predicted_params['lfo2_wave'], dim=1), classification_target_params['lfo2_wave'])
        print("filter_type",
              torch.argmax(predicted_params['filter_type'], dim=1), classification_target_params['filter_type'])
        print("regression_params",
              predicted_params['regression_params'], regression_target_parameters_tensor)


def train(model, data_loader, optimiser, device, epochs):
    for i in range(epochs):
        print(f"Epoch {i + 1}")
        train_single_epoch(model, data_loader, optimiser, device)
        print("--------------------------------------")
    print("Finished training")


def move_to(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            res[k] = move_to(v, device)
        return res
    elif isinstance(obj, list):
        res = []
        for v in obj:
            res.append(move_to(v, device))
        return res
    else:
        raise TypeError("Invalid type for move_to")


if __name__ == "__main__":

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device {device}")

    # instantiating our dataset object and create data loader
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    ai_synth_dataset = AiSynthDataset(PARAMETERS_FILE, AUDIO_DIR, mel_spectrogram, SAMPLE_RATE, device)
    train_dataloader = create_data_loader(ai_synth_dataset, BATCH_SIZE)

    # construct model and assign it to device
    synth_net = SynthNetwork().to(device)

    # initialize optimizer
    optimiser = torch.optim.Adam(synth_net.parameters(), lr=LEARNING_RATE)

    # train model
    train(synth_net, train_dataloader, optimiser, device, EPOCHS)

    # save model
    torch.save(synth_net.state_dict(), "synth_net.pth")
    print("Trained synth net saved at synth_net.pth")
