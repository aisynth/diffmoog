import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader
from config import BATCH_SIZE, EPOCHS, LEARNING_RATE
from ai_synth_dataset import AiSynthDataset
from config import SAMPLE_RATE, PARAMETERS_FILE, AUDIO_DIR
from synth_model import SynthNetwork


def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    return train_dataloader


def train_single_epoch(model, data_loader, loss_fn, optimiser, device):
    for audio_mel_spectrogram, target_parameters in data_loader:
        audio_mel_spectrogram = move_to(audio_mel_spectrogram, device)
        print(target_parameters)
        target_parameters = move_to(target_parameters, device)
        # input, target_parameters = input.to(device), target_parameters.to(device)

        # calculate loss
        predicted_parameters = model(audio_mel_spectrogram)
        print(predicted_parameters)
        print("----------------------------------------------------------------------------------------")
        print(target_parameters)

        loss_osc1_freq = nn.CrossEntropyLoss()
        loss_osc1_wave = nn.CrossEntropyLoss()
        loss_lfo1_wave = nn.CrossEntropyLoss()
        loss_osc2_freq = nn.CrossEntropyLoss()
        loss_osc2_wave = nn.CrossEntropyLoss()
        loss_lfo2_wave = nn.CrossEntropyLoss()
        loss_filter_type = nn.CrossEntropyLoss()
        loss_regression_params = nn.MSELoss()

        loss_osc1_freq = loss_osc1_freq(predicted_parameters['osc1_freq'], target_parameters['osc1_freq'])
        loss_osc1_wave = loss_osc1_wave(predicted_parameters['osc1_wave'], target_parameters['osc1_wave'])
        loss_lfo1_wave = loss_lfo1_wave(predicted_parameters['lfo1_wave'], target_parameters['lfo1_wave'])
        loss_osc2_freq = loss_osc2_freq(predicted_parameters['osc2_freq'], target_parameters['osc2_freq'])
        loss_osc2_wave = loss_osc2_wave(predicted_parameters['osc2_wave'], target_parameters['osc2_wave'])
        loss_lfo2_wave = loss_lfo2_wave(predicted_parameters['lfo2_wave'], target_parameters['lfo2_wave'])
        loss_filter_type = loss_filter_type(predicted_parameters['filter_type'], target_parameters['filter_type'])


        loss_regression_params = loss_regression_params(predicted_parameters['regression_params'],)
        # loss = loss_fn(predicted_parameters, target_parameters)

        # backpropogate error and update wights
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    print(f"loss: {loss.item()}")


def train(model, data_loader, loss_fn, optimiser, device, epochs):
    for i in range(epochs):
        print(f"Epoch {i + 1}")
        train_single_epoch(model, data_loader, loss_fn, optimiser, device)
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

    # initialize loss function + optimizer
    # todo: delete loss pass as argument
    # loss_fn = nn.CrossEntropyLoss()
    loss_fn = None
    optimiser = torch.optim.Adam(synth_net.parameters(), lr=LEARNING_RATE)

    # train model
    train(synth_net, train_dataloader, loss_fn, optimiser, device, EPOCHS)

    # save model
    torch.save(synth_net.state_dict(), "synth_net.pth")
    print("Trained synth net saved at synth_net.pth")
